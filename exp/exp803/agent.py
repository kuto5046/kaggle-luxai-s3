from typing import Any
from pathlib import Path
from collections import deque

import numpy as np
import torch
import networkx as nx
from lightning import seed_everything
from lux.utils import (
    State,
    Action,
    GlobalState,
    HiddenState,
    EpisodeStore,
    in_map,
    calc_next_pos,
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
    stochastic: bool = False  # Falseにするとargmaxで行動を選択する
    res: bool = True
    n_stack: int = 4
    # 同じマスに複数のユニットが移動する場合のペナルティ、0=重複を許可(greedy)、1=重複を禁止
    overlap_penalty: float = 1.0

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
        if torch.cuda.is_available():
            self.model.cuda()
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

        # 自陣が(0, 0)になるようにstateを反転
        do_flip = team_id == 1
        if do_flip:
            states["state"] = torch.flip(states["state"], [3, 4])

        with torch.no_grad():
            if torch.cuda.is_available():
                states = {k: v.cuda() for k, v in states.items()}
            output = self.model(states)
            if torch.cuda.is_available():
                output = {k: v.cpu() for k, v in output.items()}
            policy_map = output["policy"].squeeze().numpy()

        if do_flip:
            policy_map = np.flip(policy_map, axis=(1, 2)).copy()
            policy_map[Action.UP], policy_map[Action.DOWN] = (
                policy_map[Action.DOWN].copy(),
                policy_map[Action.UP].copy(),
            )
            policy_map[Action.LEFT], policy_map[Action.RIGHT] = (
                policy_map[Action.RIGHT].copy(),
                policy_map[Action.LEFT].copy(),
            )

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

        if len(available_unit_ids) > 1 and self.cfg.overlap_penalty > 0:
            actions = self._assign_actions_with_flow(
                available_unit_ids, unit_positions, policy_map, point_map, obs, opp_unit_positions
            )
        else:
            actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
            self._assign_greedy_actions(
                actions, available_unit_ids, unit_positions, policy_map, point_map, obs, opp_unit_positions
            )

        self.prev_opp_unit_positions = opp_unit_positions
        return actions

    def _assign_actions_with_flow(
        self, available_unit_ids, unit_positions, policy_map, point_map, obs, opp_unit_positions
    ):
        """最小費用流問題としてグリッドへの割り当てを解く"""
        # start_time = time.time()  # 開始時間を記録

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=np.int32)

        # 各ユニットの可能なアクションと移動後の位置を列挙
        unit_actions = []
        all_next_positions = set()

        for unit_id in available_unit_ids:
            unit_pos = tuple(unit_positions[unit_id])  # numpy配列をtupleに変換
            x, y = unit_pos
            policy = policy_map[:, y, x].copy()
            policy /= policy.sum()

            valid_actions = []
            for action in range(len(Action)):
                if action == Action.SAP:
                    ok = False
                    nearby_enemies = get_nearby_enemy_unit_ids(
                        unit_pos, opp_unit_positions, self.env_cfg["unit_sap_range"]
                    )
                    if len(nearby_enemies) > 0:
                        sap_pos = opp_unit_positions[np.random.choice(nearby_enemies)]
                        # 敵ユニットが2ステップ以上動いていない場合はsapする
                        if point_map[sap_pos[1], sap_pos[0]] == 1 or sap_pos in self.prev_opp_unit_positions:
                            ok = True
                        else:
                            # 敵ユニットの隣接セルがポイント位置であればそこに移動すると考える。
                            nearby_point_positions = get_nearby_point_positions(sap_pos, point_map)
                            if len(nearby_point_positions) > 0:
                                ok = True
                    if not ok:
                        continue
                    next_pos = unit_pos  # SAPは現在位置として扱う
                else:
                    next_pos = tuple(calc_next_pos(unit_pos, action))  # numpy配列をtupleに変換
                    if not in_map(next_pos):
                        continue

                # score = -np.log(policy[action] + 1e-10)
                score = 1.0 - policy[action]
                valid_actions.append({"action_id": action, "next_pos": next_pos, "score": score})
                all_next_positions.add(next_pos)

            unit_actions.append(valid_actions)

        # フローネットワークの構築
        G = nx.DiGraph()

        n_units = len(available_unit_ids)

        source = "source"
        sink = "sink"
        G.add_node(source, demand=-n_units)  # ソースから n_units 分のフローを流す
        G.add_node(sink, demand=n_units)  # シンクで n_units 分のフローを受け取る

        # ユニットノードの追加
        for i, unit_id in enumerate(available_unit_ids):
            unit_node = f"unit_{i}"
            G.add_node(unit_node, demand=0)  # 中継ノードなのでdemandは0
            G.add_edge(source, unit_node, capacity=1, weight=0)

        # 位置ノードの追加
        for pos in all_next_positions:
            pos_node = f"pos_{pos[0]}_{pos[1]}"
            G.add_node(pos_node, demand=0)  # 中継ノードなのでdemandは0
            # 重複ペナルティを調整するためにcapacityを設定
            if self.cfg.overlap_penalty == 1:
                capacity = 1  # 完全に重複を禁止
                G.add_edge(pos_node, sink, capacity=capacity, weight=0)
            else:
                capacity = len(available_unit_ids)  # 重複を許可
                for _ in range(capacity):  # 重複するごとにoverlap_penaltyを加算
                    G.add_edge(pos_node, sink, capacity=1, weight=self.cfg.overlap_penalty)
        # ユニットから位置へのエッジを追加
        for i, unit_actions_list in enumerate(unit_actions):
            unit_node = f"unit_{i}"
            for action in unit_actions_list:
                pos = action["next_pos"]
                pos_node = f"pos_{pos[0]}_{pos[1]}"
                base_cost = action["score"]
                G.add_edge(unit_node, pos_node, capacity=1, weight=base_cost)

        # 最小費用流を解く
        try:
            flow_dict = nx.min_cost_flow(G)
        except nx.NetworkXUnfeasible:
            # フローが見つからない場合は貪欲な割り当てにフォールバック
            raise ValueError("フローが見つからない")
            self._assign_greedy_actions(
                actions, available_unit_ids, unit_positions, policy_map, point_map, obs, opp_unit_positions
            )
            return actions

        # フローから行動を抽出
        for i, unit_id in enumerate(available_unit_ids):
            unit_node = f"unit_{i}"
            unit_flow = flow_dict[unit_node]
            # 選択された位置を見つける
            selected_action = None
            for action in unit_actions[i]:
                pos = action["next_pos"]
                pos_node = f"pos_{pos[0]}_{pos[1]}"
                if pos_node in unit_flow and unit_flow[pos_node] > 0:
                    selected_action = action
                    break

            if selected_action is None:
                actions[unit_id] = [Action.CENTER, 0, 0]
            elif selected_action["action_id"] == Action.SAP:
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
                        continue
                    else:
                        # 敵ユニットの隣接セルがポイント位置であればそこに移動すると考える。
                        nearby_point_positions = get_nearby_point_positions(sap_pos, point_map)
                        if len(nearby_point_positions) > 0:
                            sap_pos = nearby_point_positions[np.random.choice(len(nearby_point_positions))]
                            dx, dy = calc_relative_pos(unit_pos, sap_pos)
                            actions[unit_id] = [Action.SAP, dx, dy]
                            continue
                raise ValueError("SAP: 敵ユニットが見つからない")
            else:
                actions[unit_id] = [selected_action["action_id"], 0, 0]

        # end_time = time.time()  # 終了時間を記録
        # print(f"Flow assignment took {(end_time - start_time) * 1000:.1f} ms")  # ミリ秒単位で表示

        return actions

    def _assign_greedy_actions(
        self, actions, available_unit_ids, unit_positions, policy_map, point_map, obs, opp_unit_positions
    ):
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
