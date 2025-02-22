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
    stochastic: bool = True  # Falseにするとargmaxで行動を選択する
    res: bool = True
    n_stack: int = 4
    tta: bool = False  # 手元の検証では悪化する。入替のバグがありそう

    checkpoint_path: Path = Path(__file__).parent / "output/best_model.ckpt"


class ILAgent:
    def __init__(
        self, env_cfg: EnvParams, checkpoint_path: Path, n_stack: int, res: bool = True, tta: bool = True
    ) -> None:
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
        self.tta = tta
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
        states = np.stack(list(self.stack_states), axis=0)
        global_states = np.stack(list(self.stack_global_states), axis=0)

        # batch方向にstack
        if self.tta:
            states = self.tta_for_state(states)
            global_states = np.stack([global_states for _ in range(states.shape[0])], axis=0)
            states = torch.from_numpy(states).float()
            global_states = torch.from_numpy(global_states).float()
        else:
            # batchの次元を追加
            states = torch.from_numpy(states).unsqueeze(0).float()
            global_states = torch.from_numpy(global_states).unsqueeze(0).float()

        features = {
            "state": states,
            "global_state": global_states,
        }

        with torch.no_grad():
            output = self.model(features)
            policy_map = output["policy"].squeeze().numpy()

        if self.tta:
            policy_map = self.tta_for_policy_map(policy_map)

        policy_map = get_legal_policy(obs, policy_map, team_id, episode_store)
        point_map = state[State.POINTS]

        return policy_map, point_map

    def tta_for_state(self, state: np.ndarray) -> np.ndarray:
        tta_states = []
        tta_states.append(state.copy())
        # 上下を入れ替えている
        tta_states.append(np.flip(state, axis=2).copy())
        # 左右を入れ替えている
        tta_states.append(np.flip(state, axis=3).copy())
        # 90度回転
        tta_states.append(np.rot90(state, axes=(2, 3)).copy())
        return np.stack(tta_states, axis=0)

    def switch_action(self, policy_map: np.ndarray, indices: list[int]) -> np.ndarray:
        return policy_map[indices, :, :]

    def tta_for_policy_map(self, policy_map: np.ndarray) -> np.ndarray:
        """
        policy_map: (num_action, 24, 24)
        """
        # center up, right, down, left, sap
        # 上下を入れ替えている
        policy_map[1] = self.switch_action(
            np.flip(policy_map[1], axis=1),
            [Action.CENTER, Action.DOWN, Action.RIGHT, Action.UP, Action.LEFT, Action.SAP],
        )
        # 左右を入れ替えている
        policy_map[2] = self.switch_action(
            np.flip(policy_map[2], axis=2),
            [Action.CENTER, Action.UP, Action.LEFT, Action.DOWN, Action.RIGHT, Action.SAP],
        )
        # 90度回転(left - down - right - up)
        policy_map[3] = self.switch_action(
            np.rot90(policy_map[1], axes=(2, 1)),
            [Action.CENTER, Action.RIGHT, Action.UP, Action.LEFT, Action.DOWN, Action.SAP],
        )
        return policy_map.mean(axis=0)


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
imitation_model = ILAgent(EnvParams, cfg.checkpoint_path, cfg.n_stack, cfg.res, cfg.tta)


class Agent:
    def __init__(self, player: str, env_cfg: EnvParams) -> None:
        self.cfg = Config()
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        # np.random.seed(self.cfg.seed)
        self.env_cfg = env_cfg
        self.episode_store = EpisodeStore(self.team_id, env_cfg)
        self.prev_opp_unit_positions = []

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
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            x, y = unit_pos
            policy = policy_map[:, y, x]

            while True:
                if cfg.stochastic:
                    if policy.sum() == 0:
                        actions[unit_id] = [Action.CENTER, 0, 0]
                        break

                    policy = policy / policy.sum()
                    action = np.random.choice(range(6), p=policy)
                else:
                    action = policy.argmax()

                # print(policy, file=sys.stderr)
                if action == Action.SAP:
                    # 範囲内にいる敵ユニットを取得
                    nearby_enemy_unit_ids = get_nearby_enemy_unit_ids(
                        unit_pos, opp_unit_positions, self.env_cfg["unit_sap_range"]
                    )
                    if len(nearby_enemy_unit_ids) > 0:
                        sap_pos = opp_unit_positions[np.random.choice(nearby_enemy_unit_ids)]
                        # 敵ユニットが2ステップ以上動いていない場合はsapする
                        if point_map[sap_pos[1], sap_pos[0]] == 1 or sap_pos in self.prev_opp_unit_positions:
                            dx, dy = calc_relative_pos(unit_pos, sap_pos)
                            actions[unit_id] = [Action.SAP, dx, dy]
                            break
                        else:
                            # 敵ユニットの隣接セルがポイント位置であればそこに移動すると考える。
                            nearby_point_positions = get_nearby_point_positions(sap_pos, point_map)
                            if len(nearby_point_positions) > 0:
                                sap_pos = nearby_point_positions[np.random.choice(len(nearby_point_positions))]
                                dx, dy = calc_relative_pos(unit_pos, sap_pos)
                                actions[unit_id] = [Action.SAP, dx, dy]
                                break

                    policy[Action.SAP] = 0
                else:
                    actions[unit_id] = [action, 0, 0]
                    break
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
