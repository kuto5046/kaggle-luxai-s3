from typing import Any
from pathlib import Path
from collections import deque
import time
import sys
from heapq import heappop, heappush  # for dijkstra in MinimumCostFlow


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
    calc_next_pos,
    extract_state,
    get_valid_sap_map,
    extract_global_state,
    get_valid_policy_map,
)
from lux.models import LuxUNetModel, LuxConvLSTMModel
from lux.params import EnvParams
from scipy.special import softmax


class Config:
    seed: int = 2025
    # 確率的な行動を取るかどうか
    stochastic: bool = True  # Falseにするとargmaxで行動を選択する
    # model
    n_stack: int = 8
    num_layers: int = 3
    hidden_dim: int = 64
    kernel_size: int = 5
    num_repeats: int = 3

    overlap_penalty: float = 2.0

    checkpoint_path: Path = Path(__file__).parent / "output/best_model.ckpt"


###########################################################################
# MinCostFlow algorithm (Primal Dual with Dijkstra)                       #
# ----------------------------------------------------------------------- #


# verified http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=4675288#1
class MinimumCostFlow:
    inf = 1000000000000000000

    def __init__(self, n):
        self.n = n
        self.edges = [[] for i in range(n)]

    def add_edge(self, f, t, capacity, cost, action):
        self.edges[f].append([t, capacity, cost, len(self.edges[t]), action])
        self.edges[t].append([f, 0, -cost, len(self.edges[f]) - 1, -1])  # reverse edge

    def flow(self, s, t, flow, timeout=0.1):
        n = self.n
        g = self.edges
        inf = MinimumCostFlow.inf

        prevv = [0 for i in range(n)]
        preve = [0 for i in range(n)]
        h = [0 for i in range(n)]
        dist = [inf for i in range(n)]

        res = 0
        start_time = time.time()

        while flow != 0:
            if time.time() - start_time > timeout:
                return -1

            dist = [inf for i in range(n)]
            dist[s] = 0
            que = [(0, s)]

            while que:
                if time.time() - start_time > timeout:
                    return -1

                c, v = heappop(que)
                if dist[v] < c:
                    continue
                r0 = dist[v] + h[v]
                for i, e in enumerate(g[v]):
                    w, cap, cost, _, _ = e
                    if cap > 0 and r0 + cost - h[w] < dist[w]:
                        r = r0 + cost - h[w]
                        dist[w] = r
                        prevv[w] = v
                        preve[w] = i
                        heappush(que, (r, w))

            if dist[t] == inf:
                return -1

            for i in range(n):
                h[i] += dist[i]

            d = flow
            v = t
            while v != s:
                d = min(d, g[prevv[v]][preve[v]][1])
                v = prevv[v]
            flow -= d
            res += d * h[t]
            v = t
            while v != s:
                e = g[prevv[v]][preve[v]]
                e[1] -= d
                g[v][e[3]][1] += d
                v = prevv[v]
        return res


class ILAgent:
    def __init__(self, env_cfg: EnvParams, checkpoint_path: Path, n_stack: int) -> None:
        self.model = LuxConvLSTMModel(
            state_space_size=len(State),
            global_state_space_size=len(GlobalState),
            action_space_size=len(Action),
            hidden_state_space_size=len(HiddenState),
            num_layers=cfg.num_layers,
            hidden_dim=cfg.hidden_dim,
            kernel_size=cfg.kernel_size,
            num_repeats=cfg.num_repeats,
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
        global_state = extract_global_state(obs, team_id, self.env_cfg, episode_store)
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
        self.prev_actions = None

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # マッチごとにリセットされる要素をリセット
        if obs["match_steps"] == 0:
            self.episode_store.reset()
        else:
            self.episode_store.update(obs, self.prev_actions)
        policy_map, point_map = imitation_model.predict(obs, self.team_id, self.episode_store)

        unit_mask = np.array(obs["units_mask"][self.team_id])  # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id])  # shape (max_units, 2)
        available_unit_ids = np.where(unit_mask)[0]

        opp_unit_positions = [tuple(pos) for pos in obs["units"]["position"][self.opp_team_id] if pos[0] != -1]

        if len(available_unit_ids) > 1 and self.cfg.overlap_penalty > 0:
            actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
            self._assign_actions_with_flow(
                actions, available_unit_ids, unit_positions, policy_map, point_map, obs, opp_unit_positions
            )
        else:
            actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
            self._assign_greedy_actions(
                actions, available_unit_ids, unit_positions, policy_map, point_map, obs, opp_unit_positions
            )

        self.prev_opp_unit_positions = opp_unit_positions
        self.prev_actions = actions
        return actions

    def _assign_actions_with_flow(
        self, actions, available_unit_ids, unit_positions, policy_map, point_map, obs, opp_unit_positions
    ):
        """最小費用流問題としてグリッドへの割り当てを解く"""
        # self._assign_greedy_actions(
        #         actions, available_unit_ids, unit_positions, policy_map, point_map, obs, opp_unit_positions
        #     )
        # return
        # start_time = time.time()  # 開始時間を記録

        # 各ユニットの可能なアクションと移動後の位置を列挙
        unit_actions = []
        all_next_positions = set()

        for unit_id in available_unit_ids:
            current_pos = unit_positions[unit_id]
            unit_pos = (int(current_pos[0]), int(current_pos[1]))  # numpy配列をintのtupleに確実に変換
            x, y = unit_pos
            policy = policy_map[:, y, x].copy()
            if policy.sum() > 0:  # policyの合計が0の場合はスキップ
                policy /= policy.sum()
            else:
                # policyが全て0の場合はCENTERのみを有効にする
                policy = np.zeros_like(policy)
                policy[Action.CENTER] = 1.0

            valid_actions = []
            for action in range(len(Action)):
                sap_pos_relative = None
                if action == Action.SAP:
                    # 範囲内にいる敵ユニットを取得
                    nearby_enemy_unit_ids = get_nearby_enemy_unit_ids(
                        unit_pos, opp_unit_positions, self.env_cfg["unit_sap_range"]
                    )
                    if len(nearby_enemy_unit_ids) > 0:
                        sap_pos = opp_unit_positions[np.random.choice(nearby_enemy_unit_ids)]
                        # 敵ユニットが2ステップ以上動いていない場合はsapする
                        if point_map[sap_pos[1], sap_pos[0]] == 1 or sap_pos in self.prev_opp_unit_positions:
                            sap_pos_relative = calc_relative_pos(np.array(unit_pos), np.array(sap_pos))
                        else:
                            # 敵ユニットの隣接セルがポイント位置であればそこに移動すると考える。
                            nearby_point_positions = get_nearby_point_positions(sap_pos, point_map)
                            if len(nearby_point_positions) > 0:
                                sap_pos = nearby_point_positions[np.random.choice(len(nearby_point_positions))]
                                sap_pos_relative = calc_relative_pos(np.array(unit_pos), np.array(sap_pos))
                    next_pos = unit_pos  # SAPは現在位置として扱う
                    if sap_pos_relative is None:
                        continue
                else:
                    next_pos = calc_next_pos(unit_pos, action)
                    if not in_map(next_pos):
                        continue

                # score = -np.log(policy[action] + 1e-10)
                score = 1.0 - policy[action]
                valid_actions.append(
                    {"action_id": action, "next_pos": next_pos, "score": score, "sap_pos": sap_pos_relative}
                )
                all_next_positions.add(next_pos)

            unit_actions.append(valid_actions)

        # フローネットワークの構築
        n_units = len(available_unit_ids)
        n_cells = len(all_next_positions)
        flow = MinimumCostFlow(2 + n_units + 2 * n_cells)
        nodes = {}
        nodes["source"] = 0
        nodes["sink"] = 1
        for i in range(n_units):
            nodes[f"unit_{i}"] = 2 + i
        for i, (x, y) in enumerate(all_next_positions):
            nodes[f"pos_{x}_{y}"] = 2 + n_units + i
            nodes[f"pos_{x}_{y}_additional"] = 2 + n_units + n_cells + i

        for i, unit_id in enumerate(available_unit_ids):
            flow.add_edge(nodes["source"], nodes[f"unit_{i}"], capacity=1, cost=0, action=-1)
        for pos in all_next_positions:
            pos_node = nodes[f"pos_{pos[0]}_{pos[1]}"]
            pos_node_additional = nodes[f"pos_{pos[0]}_{pos[1]}_additional"]
            flow.add_edge(pos_node, nodes["sink"], capacity=1, cost=0, action=-1)
            additional_capacity = max(len(available_unit_ids) - 1, 1)  # 最低でも1の容量を確保
            flow.add_edge(
                pos_node_additional,
                nodes["sink"],
                capacity=additional_capacity,
                cost=self.cfg.overlap_penalty,
                action=-1,
            )

        for i, unit_actions_list in enumerate(unit_actions):
            unit_node = nodes[f"unit_{i}"]
            for action in unit_actions_list:
                pos = action["next_pos"]
                base_cost = action["score"]
                pos_node = nodes[f"pos_{pos[0]}_{pos[1]}"]
                pos_node_additional = nodes[f"pos_{pos[0]}_{pos[1]}_additional"]
                flow.add_edge(unit_node, pos_node, capacity=1, cost=base_cost, action=action)
                flow.add_edge(unit_node, pos_node_additional, capacity=1, cost=base_cost, action=action)
        # try:
        flow_result = flow.flow(nodes["source"], nodes["sink"], len(available_unit_ids))
        # except Exception as e:
        #     print(f"フロー計算エラー: {e}")
        #     self._assign_greedy_actions(actions, available_unit_ids, unit_positions, policy_map, point_map, obs, opp_unit_positions)
        #     return
        if flow_result == -1:
            print("flow=-1", file=sys.stderr)
            self._assign_greedy_actions(
                actions, available_unit_ids, unit_positions, policy_map, point_map, obs, opp_unit_positions
            )
            return

        # フローから行動を抽出
        for i, unit_id in enumerate(available_unit_ids):
            unit_node = nodes[f"unit_{i}"]
            unit_pos = unit_positions[unit_id]
            selected_action = None
            for edge in flow.edges[unit_node]:
                if edge[1] == 0 and edge[4] != -1:
                    selected_action = edge[4]
                    break
            if selected_action is None:
                actions[unit_id] = [Action.CENTER, 0, 0]
            elif selected_action["action_id"] == Action.SAP:
                actions[unit_id] = [Action.SAP, selected_action["sap_pos"][0], selected_action["sap_pos"][1]]
            else:
                actions[unit_id] = [selected_action["action_id"], 0, 0]

        # for unit_id in available_unit_ids:
        #     print(actions[unit_id], file=sys.stderr)
        # end_time = time.time()  # 終了時間を記録
        # print(f"Flow assignment took {(end_time - start_time) * 1000:.1f} ms")  # ミリ秒単位で表示

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
