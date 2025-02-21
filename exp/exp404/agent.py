from typing import Any
from pathlib import Path
from collections import deque

import numpy as np
import torch
from lightning import seed_everything
from lux.utils import (
    State,
    Action,
    GlobalState,
    HiddenState,
    EpisodeStore,
    in_map,
    extract_state,
    get_valid_sap_map,
    extract_global_state,
    get_valid_policy_map,
)
from lux.models import LuxUNetModel
from lux.params import EnvParams
from scipy.special import softmax


class Config:
    seed: int = 2025
    # 確率的な行動を取るかどうか
    # stochastic: bool = True  # Falseにするとargmaxで行動を選択する
    res: bool = True
    n_stack: int = 4

    checkpoint_path: Path = Path(__file__).parent / "output/best_model.ckpt"


class ILAgent:
    def __init__(self, env_cfg: EnvParams, checkpoint_path: Path, n_stack: int, res: bool = True) -> None:
        self.model = LuxUNetModel(
            state_space_size=len(State),
            global_state_space_size=len(GlobalState),
            action_space_size=len(Action),
            hidden_state_space_size=len(HiddenState),
            n_stack=n_stack,
            res=res,
        )
        ckpt = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.player = None
        self.env_cfg = env_cfg
        # n_stack分のstateを保持するqueue
        self.stack_states = deque(maxlen=n_stack)
        self.stack_global_states = deque(maxlen=n_stack)
        for i in range(n_stack):
            self.stack_states.append(np.zeros((len(State), 24, 24)))
            self.stack_global_states.append(np.zeros(len(GlobalState)))

    def predict(self, obs: dict[str, Any], team_id: int, episode_store: EpisodeStore) -> tuple[np.ndarray, np.ndarray]:
        state = extract_state(obs, team_id, episode_store)
        global_state = extract_global_state(obs, team_id, self.env_cfg)
        self.stack_states.append(state)
        self.stack_global_states.append(global_state)
        states = {
            "state": torch.from_numpy(np.stack(list(self.stack_states), axis=0)).unsqueeze(0).float(),
            "global_state": torch.from_numpy(np.stack(list(self.stack_global_states), axis=0)).unsqueeze(0).float(),
        }

        with torch.no_grad():
            output = self.model(states)
            policy_map = output["policy"].squeeze().numpy()

        policy_map = get_legal_policy(obs, policy_map, team_id, episode_store)
        point_map = state[State.POINTS]

        return policy_map, point_map


def get_legal_policy(
    obs: dict[str, Any], policy_map: np.ndarray, team_id: int, episode_store: EpisodeStore
) -> np.ndarray:
    legal_action_map = get_valid_policy_map(obs, team_id, episode_store)
    action_mask_map = np.ones_like(policy_map) * 1e32
    action_mask_map[legal_action_map > 0] = 0  # legal actionは0でそれ以外は1e32
    # 無効な行動は負の大きな値になるためsoftmax後は0になる。その上で再度無効な行動を0にする
    policy_map = softmax(policy_map - action_mask_map, axis=0) * (action_mask_map == 0) * 1
    return policy_map


def get_legal_sap_policy(
    obs: dict[str, Any], sap_map: np.ndarray, team_id: int, episode_store: EpisodeStore
) -> np.ndarray:
    legal_sap_map = get_valid_sap_map(obs, team_id, episode_store)
    # 無効な場所は0にする
    sap_map *= legal_sap_map
    return sap_map


cfg = Config()
seed_everything(cfg.seed, workers=True)
imitation_model = ILAgent(EnvParams, cfg.checkpoint_path, cfg.n_stack, cfg.res)


class Agent:
    def __init__(self, player: str, env_cfg: EnvParams) -> None:
        torch.set_num_threads(1)
        self.cfg = Config()
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        # np.random.seed(self.cfg.seed)
        self.env_cfg = env_cfg
        self.episode_store = EpisodeStore(self.team_id, env_cfg)
        self.prev_opp_unit_positions = []

    def _can_sap(
        self,
        unit_pos: tuple[int, int],
        opp_unit_positions: list[tuple[int, int]],
        point_map: np.ndarray,
        sapped_points: set,
    ) -> tuple[bool, tuple[Action, int, int]]:
        # 範囲内にいる敵ユニットを取得
        nearby_enemy_unit_ids = get_nearby_enemy_unit_ids(unit_pos, opp_unit_positions, self.env_cfg["unit_sap_range"])
        # nearby_enemy_unit_idsからsapped_pointsに含まれるユニットを除外
        nearby_enemy_unit_ids = [
            unit_id for unit_id in nearby_enemy_unit_ids if opp_unit_positions[unit_id] not in sapped_points
        ]
        if len(nearby_enemy_unit_ids) > 0:
            sap_pos = opp_unit_positions[np.random.choice(nearby_enemy_unit_ids)]
            # 敵ユニットが2ステップ以上動いていない場合はsapする
            if point_map[sap_pos[1], sap_pos[0]] == 1 or sap_pos in self.prev_opp_unit_positions:
                dx, dy = calc_relative_pos(unit_pos, sap_pos)
                return True, [Action.SAP, dx, dy]
            else:
                # 敵ユニットの隣接セルがポイント位置であればそこに移動すると考える。
                nearby_point_positions = get_nearby_point_positions(sap_pos, point_map)
                if len(nearby_point_positions) > 0:
                    sap_pos = nearby_point_positions[np.random.choice(len(nearby_point_positions))]
                    dx, dy = calc_relative_pos(unit_pos, sap_pos)
                    return True, [Action.SAP, dx, dy]
        return False, [Action.CENTER, 0, 0]

    # 確率に応じた重み付きラウンドロビンで、ユニット数分の行動順序（リスト）を作成する関数
    def weighted_round_robin(self, candidates: list[tuple[int, float]], total: int) -> list[int]:
        total_weight = sum(weight for _, weight in candidates)
        # 各候補の現在の値を初期化
        current = {action: 0.0 for action, _ in candidates}
        ordering = []
        for _ in range(total):
            # 各候補の現在値に重みを加算
            for action, weight in candidates:
                current[action] += weight
            # 現在値が最大の候補を選ぶ
            chosen = max(candidates, key=lambda x: current[x[0]])[0]
            ordering.append(chosen)
            # 選ばれた候補から全候補の重み合計を引く
            current[chosen] -= total_weight
        return ordering

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # マッチごとにリセットされる要素をリセット
        if obs["match_steps"] == 0:
            self.episode_store.reset()
        else:
            self.episode_store.update(obs)
        policy_map, point_map = imitation_model.predict(obs, self.team_id, self.episode_store)

        unit_mask = np.array(obs["units_mask"][self.team_id])  # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id])  # shape (max_units, 2)
        available_unit_ids = np.where(unit_mask)[0]

        opp_unit_positions = [tuple(pos) for pos in obs["units"]["position"][self.opp_team_id] if pos[0] != -1]
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        # unit ids range from 0 to max_units - 1
        pos_to_unit_id = dict()

        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            x, y = unit_pos
            if (x, y) in pos_to_unit_id:
                pos_to_unit_id[(x, y)].append(unit_id)
            else:
                pos_to_unit_id[(x, y)] = [unit_id]

        sapped_points = set()

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        for unit_pos, unit_ids in pos_to_unit_id.items():
            x, y = unit_pos
            policy = policy_map[:, y, x].copy()

            # unit_idの分だけpolicyの候補を決めておく
            action_candidates = []
            # まず、上位の行動を3つ取得
            for _ in range(3):
                action = policy.argmax()
                if action == Action.SAP and not get_valid_sap_map(obs, self.team_id, self.episode_store)[y, x]:
                    policy[action] = -1
                    continue
                action_candidates.append((action, policy[action]))
                policy[action] = -1

            # 確率に応じて行動を取得.高いもの先に来るようにしている. A:0.4, B:0.1 の場合、ユニット数が5ならA,A,B,A,Aのようになるはず
            total_units = len(unit_ids)
            ordering = self.weighted_round_robin(action_candidates, total_units)
            # SAPがだめなとき用に確率が高い順に行動を取得
            sap_alt = action_candidates[0][0] if action_candidates[0][0] != Action.SAP else action_candidates[1][0]
            assert len(ordering) == total_units
            for unit_id, act in zip(unit_ids, ordering):
                if act == Action.SAP:
                    can_sap, sap_action = self._can_sap(unit_pos, opp_unit_positions, point_map, sapped_points)
                    if can_sap:
                        sapped_points.add(tuple(unit_pos))
                        actions[unit_id] = sap_action
                    else:
                        actions[unit_id] = [sap_alt, 0, 0]
                else:
                    actions[unit_id] = [act, 0, 0]

        # 敵ユニットの位置を更新
        self.prev_opp_unit_positions = opp_unit_positions
        return actions


# 相対位置を計算
def calc_relative_pos(base_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    return target_pos - base_pos


# マスの半径kマス以内に該当するかどうか
def is_within_k_tiles(base_pos: np.ndarray, target_pos: np.ndarray, k: int) -> bool:
    return np.abs(base_pos[0] - target_pos[0]) <= k and np.abs(base_pos[1] - target_pos[1]) <= k


# 隣接するマスにあるポイントマスを取得
def get_nearby_point_positions(pos: np.ndarray, point_map: np.ndarray, k: int = 1) -> list[np.ndarray]:
    # posを中心にkマス以内のマスを取得
    nearby_positions = []
    up_pos = (pos[0], pos[1] - k)
    if in_map(up_pos) and point_map[up_pos[1], up_pos[0]] == 1:
        nearby_positions.append(up_pos)
    down_pos = (pos[0], pos[1] + k)
    if in_map(down_pos) and point_map[down_pos[1], down_pos[0]] == 1:
        nearby_positions.append(down_pos)
    left_pos = (pos[0] - k, pos[1])
    if in_map(left_pos) and point_map[left_pos[1], left_pos[0]] == 1:
        nearby_positions.append(left_pos)
    right_pos = (pos[0] + k, pos[1])
    if in_map(right_pos) and point_map[right_pos[1], right_pos[0]] == 1:
        nearby_positions.append(right_pos)
    return nearby_positions


# 自身の周囲kタイル以内にいる敵ユニットを抽出
def get_nearby_enemy_unit_ids(
    unit_pos: tuple[int, int], opp_unit_positions: list[tuple[int, int]], k: int
) -> list[int]:
    return [unit_id for unit_id, pos in enumerate(opp_unit_positions) if is_within_k_tiles(unit_pos, pos, k)]
