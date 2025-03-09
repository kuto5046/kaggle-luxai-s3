import time
from heapq import heappop, heappush  # for dijkstra in MinimumCostFlow
from typing import Any
from pathlib import Path
from collections import deque

import numpy as np
import torch
from lightning import seed_everything
from lux.utils import (
    is_kaggle_environment,
    State,
    Action,
    GlobalState,
    EpisodeStore,
    in_map,
    calc_next_pos,
    extract_state,
    calc_relative_pos,
    get_valid_sap_map,
    extract_global_state,
    get_valid_policy_map,
    get_nearby_enemy_unit_ids,
    get_nearby_point_positions,
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
    # 同じマスに複数のユニットが移動する場合のペナルティ、0=重複を許可(greedy)、1=重複を禁止
    overlap_penalty: float = 2.0

    tta: bool = True
    debug: bool = False

    checkpoint_path: Path = Path(__file__).parent / "output/best_model_exp629.ckpt"


###########################################################################
# MinCostFlow algorithm (Primal Dual with Dijkstra)                       #
# ----------------------------------------------------------------------- #


# verified http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=4675288#1
class MinimumCostFlow:
    inf = 10**18  # 十分大きな値

    def __init__(self, n):
        self.n = n
        self.edges = [[] for i in range(n)]

    def add_edge(self, f, t, capacity, cost, action):
        self.edges[f].append([t, capacity, cost, len(self.edges[t]), action])
        self.edges[t].append([f, 0, -cost, len(self.edges[f]) - 1, -1])  # reverse edge

    def flow(self, s, t, flow, timeout=1.0):
        n = self.n
        g = self.edges
        inf = MinimumCostFlow.inf

        prevv = [0 for i in range(n)]
        preve = [0 for i in range(n)]
        h = [0 for i in range(n)]
        dist = [inf for i in range(n)]

        res = 0
        start_time = time.time()
        eps = 1e-9  # 浮動小数点誤差対策用の許容値

        while flow > 0:
            if time.time() - start_time > timeout:
                return -2

            dist = [inf for i in range(n)]
            dist[s] = 0
            que = [(0, s)]

            while que:
                if time.time() - start_time > timeout:
                    return -2

                c, v = heappop(que)
                if dist[v] < c - eps:
                    continue
                # 隣接する各辺を緩和
                for i, e in enumerate(g[v]):
                    w, cap, cost, _, _ = e
                    if cap > 0:
                        r = dist[v] + cost + h[v] - h[w]
                        if r < dist[w] - eps:
                            dist[w] = r
                            prevv[w] = v
                            preve[w] = i
                            heappush(que, (r, w))

            # t への到達がなければ経路が存在しない
            if dist[t] == inf:
                return -1

            # 到達可能なノードのみ h を更新する
            for i in range(n):
                if dist[i] < inf:
                    h[i] += dist[i]

            # この経路で流せる最大の流量 d を求める
            d = flow
            v = t
            while v != s:
                d = min(d, g[prevv[v]][preve[v]][1])
                v = prevv[v]
            flow -= d
            res += d * h[t]

            # 経路に沿って辺の容量を更新
            v = t
            while v != s:
                e = g[prevv[v]][preve[v]]
                e[1] -= d
                g[v][e[3]][1] += d
                v = prevv[v]
        return res


class ILAgent:
    def __init__(self, env_cfg: EnvParams, checkpoint_path: Path, n_stack: int, res: bool = True) -> None:
        self.model = LuxUNetModel(
            state_space_size=len(State),
            global_state_space_size=len(GlobalState),
            action_space_size=len(Action),
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

    def transpose_state(self, state: torch.Tensor) -> torch.Tensor:
        assert state.dim() == 5
        state = state.clone().detach()
        return torch.permute(state, [0, 1, 2, 4, 3])

    def transpose_global_state(self, global_state: torch.Tensor) -> torch.Tensor:
        global_state = global_state.clone().detach()
        return global_state

    def transpose_policy(self, policy: torch.Tensor) -> torch.Tensor:
        assert policy.dim() == 4
        policy = policy.clone().detach()
        policy = torch.permute(policy, [0, 1, 3, 2])
        policy[:, Action.UP], policy[:, Action.LEFT] = policy[:, Action.LEFT].clone(), policy[:, Action.UP].clone()
        policy[:, Action.DOWN], policy[:, Action.RIGHT] = (
            policy[:, Action.RIGHT].clone(),
            policy[:, Action.DOWN].clone(),
        )
        return policy

    def transpose_sap(self, sap: torch.Tensor) -> torch.Tensor:
        assert sap.dim() == 4
        sap = sap.clone().detach()
        return torch.permute(sap, [0, 1, 3, 2])

    def predict(
        self, obs: dict[str, Any], team_id: int, episode_store: EpisodeStore, cfg: Config
    ) -> tuple[np.ndarray, np.ndarray]:
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
            if cfg.tta:
                states["state"] = torch.cat([states["state"], self.transpose_state(states["state"])], dim=0)
                states["global_state"] = torch.cat(
                    [states["global_state"], self.transpose_global_state(states["global_state"])], dim=0
                )
            if torch.cuda.is_available():
                states = {k: v.cuda() for k, v in states.items()}
            output = self.model(states)
            if torch.cuda.is_available():
                output = {k: v.cpu() for k, v in output.items()}
            if cfg.tta:
                assert output["policy"].shape[0] == 2
                assert output["sap"].shape[0] == 2
                # output["policy"] = (output["policy"][:1] + self.transpose_policy(output["policy"][1:2])) / 2
                # output["sap"] = (output["sap"][:1] + self.transpose_sap(output["sap"][1:2])) / 2
                weight_low_policy = 0.5
                weight_low_sap = 0.5
                output_policy_maximum = torch.maximum(
                    output["policy"][:1], self.transpose_policy(output["policy"][1:2])
                )
                output_policy_minimum = torch.minimum(
                    output["policy"][:1], self.transpose_policy(output["policy"][1:2])
                )
                output["policy"] = (
                    output_policy_maximum + (output_policy_minimum - output_policy_maximum) * weight_low_policy
                )
                output_sap_maximum = torch.maximum(output["sap"][:1], self.transpose_sap(output["sap"][1:2]))
                output_sap_minimum = torch.minimum(output["sap"][:1], self.transpose_sap(output["sap"][1:2]))
                output["sap"] = output_sap_maximum + (output_sap_minimum - output_sap_maximum) * weight_low_sap
            if do_flip:
                output["sap"] = torch.flip(output["sap"], [-2, -1])
            policy_map = output["policy"].squeeze().numpy()
            sap = torch.sigmoid(output["sap"]).squeeze().numpy()
            sap_available_area = state[State.SAP_AVAILABLE_AREA]
            sap = sap * sap_available_area

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

        return policy_map, point_map, sap


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


def use_cpp_flow():
    # 提出環境ではpythonのflow計算を使う想定. kaggle環境でpybind周りをなんとかすれば使えるはずだが未整備
    # debugモードではpythonのflow計算との比較を行うため使用している
    return not is_kaggle_environment() or Config().debug


def use_py_flow():
    return is_kaggle_environment() or Config().debug


class SingleSapInfo:
    def __init__(
        self,
        sap_pos: list[tuple[int, int]],
        sap_policy: list[float],
        next_policy: float,
        next_action,
        unit_id: int,
        unit_pos: tuple[int, int],
    ) -> None:
        self.sap_pos = sap_pos  # sapの位置
        self.sap_policy = sap_policy  # sapの各位置のpolicy
        self.next_policy = next_policy  # sap以外の行動のpolicyのうちの最大値
        self.next_action = next_action  # sap以外の行動のうちの最大値の行動
        self.unit_id = unit_id
        self.unit_pos = unit_pos


cfg = Config()
seed_everything(cfg.seed, workers=True)
imitation_model = ILAgent(EnvParams, cfg.checkpoint_path, cfg.n_stack, cfg.res)
if use_cpp_flow():
    import min_cost_flow


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
        policy_map, _point_map, sap_map = imitation_model.predict(obs, self.team_id, self.episode_store, self.cfg)

        unit_mask = np.array(obs["units_mask"][self.team_id])  # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id])  # shape (max_units, 2)
        available_unit_ids = np.where(unit_mask)[0]

        opp_unit_positions = [tuple(pos) for pos in obs["units"]["position"][self.opp_team_id] if pos[0] != -1]

        # sapの位置を決定 -> 残りの行動をフローで決定
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        self._assign_greedy_actions_sap(actions, available_unit_ids, unit_positions, policy_map, sap_map)
        sapped_unit_next_pos_set = set()
        for unit_id in available_unit_ids:
            action = actions[unit_id]
            if action[0] == Action.SAP:
                sapped_unit_next_pos = (unit_positions[unit_id][0], unit_positions[unit_id][1])
                sapped_unit_next_pos_set.add(sapped_unit_next_pos)

        # available_unit_idsからsapしたユニットを除外
        next_available_unit_ids = [unit_id for unit_id in available_unit_ids if actions[unit_id][0] != Action.SAP]

        self._assign_actions_with_flow(
            actions, next_available_unit_ids, unit_positions, policy_map, sapped_unit_next_pos_set
        )

        self.prev_opp_unit_positions = opp_unit_positions
        self.prev_actions = actions
        return actions

    def _assign_actions_with_flow(
        self, actions, available_unit_ids, unit_positions, policy_map, sapped_unit_next_pos_set
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
                    continue
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
        if use_py_flow():
            flow = MinimumCostFlow(2 + n_units + 2 * n_cells)
        if use_cpp_flow():
            mcf = min_cost_flow.MinimumCostFlow(2 + n_units + 2 * n_cells)

        nodes = {}
        nodes["source"] = 0
        nodes["sink"] = 1
        for i in range(n_units):
            nodes[f"unit_{i}"] = 2 + i
        for i, (x, y) in enumerate(all_next_positions):
            nodes[f"pos_{x}_{y}"] = 2 + n_units + i
            nodes[f"pos_{x}_{y}_additional"] = 2 + n_units + n_cells + i

        for i, unit_id in enumerate(available_unit_ids):
            if use_py_flow():
                flow.add_edge(nodes["source"], nodes[f"unit_{i}"], capacity=1, cost=0, action=-1)
            if use_cpp_flow():
                mcf.add_edge(nodes["source"], nodes[f"unit_{i}"], 1, 0, -1)
        for pos in all_next_positions:
            pos_node = nodes[f"pos_{pos[0]}_{pos[1]}"]
            pos_node_additional = nodes[f"pos_{pos[0]}_{pos[1]}_additional"]
            dec = 0
            if pos not in sapped_unit_next_pos_set:
                if use_py_flow():
                    flow.add_edge(pos_node, nodes["sink"], capacity=1, cost=0, action=-1)
                if use_cpp_flow():
                    mcf.add_edge(pos_node, nodes["sink"], 1, 0, -1)
                dec = 1
            additional_capacity = max(len(available_unit_ids) - dec, 1)  # 最低でも1の容量を確保
            if use_py_flow():
                flow.add_edge(
                    pos_node_additional,
                    nodes["sink"],
                    capacity=additional_capacity,
                    cost=self.cfg.overlap_penalty,
                    action=-1,
                )
            if use_cpp_flow():
                mcf.add_edge(
                    pos_node_additional,
                    nodes["sink"],
                    additional_capacity,
                    self.cfg.overlap_penalty,
                    -1,
                )

        for i, unit_actions_list in enumerate(unit_actions):
            unit_node = nodes[f"unit_{i}"]
            for action in unit_actions_list:
                pos = action["next_pos"]
                base_cost = action["score"]
                pos_node = nodes[f"pos_{pos[0]}_{pos[1]}"]
                pos_node_additional = nodes[f"pos_{pos[0]}_{pos[1]}_additional"]
                if use_py_flow():
                    flow.add_edge(unit_node, pos_node, capacity=1, cost=base_cost, action=action)
                    flow.add_edge(unit_node, pos_node_additional, capacity=1, cost=base_cost, action=action)
                if use_cpp_flow():
                    mcf.add_edge(unit_node, pos_node, 1, base_cost, action)
                    mcf.add_edge(unit_node, pos_node_additional, 1, base_cost, action)

        if use_py_flow():
            flow_result = flow.flow(nodes["source"], nodes["sink"], len(available_unit_ids))
            assert flow_result != -1, "Flow calculation failed"
            assert flow_result != -2, "Flow calculation timeout"
        if use_cpp_flow():
            mcflow_result = mcf.flow(nodes["source"], nodes["sink"], len(available_unit_ids))
            assert mcflow_result != -1, "Flow calculation failed"
            assert mcflow_result != -2, "Flow calculation timeout"

        if use_py_flow() and use_cpp_flow():
            for i, unit_id in enumerate(available_unit_ids):
                for original_edge, edge in zip(flow.edges[unit_id], mcf.edges[unit_id]):
                    assert original_edge[1] == edge.cap
                    assert original_edge[4] == edge.action

        # フローから行動を抽出
        for i, unit_id in enumerate(available_unit_ids):
            unit_node = nodes[f"unit_{i}"]
            unit_pos = unit_positions[unit_id]
            selected_action = None
            if use_cpp_flow():
                for edge in mcf.edges[unit_node]:
                    if edge.cap == 0 and edge.action != -1:
                        selected_action = edge.action
                        break
            elif use_py_flow():
                for edge in flow.edges[unit_node]:
                    if edge[1] == 0 and edge[4] != -1:
                        selected_action = edge[4]
                        break
            if selected_action is None:
                actions[unit_id] = [Action.CENTER, 0, 0]
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

    def _assign_greedy_actions_sap(self, actions, available_unit_ids, unit_positions, policy_map, sap_map):
        sap_infos: list[SingleSapInfo] = []

        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            x, y = unit_pos
            policy = policy_map[:, y, x].copy()
            # print(f"{x=} {y=} {policy=}", file=sys.stderr)

            if self.cfg.stochastic:
                if policy.sum() == 0:
                    actions[unit_id] = [Action.CENTER, 0, 0]
                    continue

                policy = policy / policy.sum()
                action = np.random.choice(range(6), p=policy)
            else:
                action = policy.argmax()

            if action == Action.SAP:
                # 範囲内のpolicyをすべて取得
                sap_poses = []
                sap_policies = []
                for dx in range(-self.env_cfg["unit_sap_range"], self.env_cfg["unit_sap_range"] + 1):
                    for dy in range(-self.env_cfg["unit_sap_range"], self.env_cfg["unit_sap_range"] + 1):
                        sap_pos = (x + dx, y + dy)
                        if in_map(sap_pos):
                            sap_poses.append(sap_pos)
                            sap_policies.append(sap_map[sap_pos[1], sap_pos[0]] * policy[Action.SAP])
                policy[Action.SAP] = 0
                next_action = policy.argmax()
                next_action_info = {
                    "action_id": next_action,
                    "next_pos": unit_pos,
                    "score": 1.0 - policy[next_action],
                    "sap_pos": None,
                }
                next_policy = policy[next_action]
                sap_infos.append(
                    SingleSapInfo(sap_poses, sap_policies, next_policy, next_action_info, unit_id, unit_pos)
                )

        # SAPの処理. min_cost_flowでSAPの位置を決定
        if len(sap_infos) == 0:
            return

        n_units = len(sap_infos)
        sap_unit_ids = [sap_info.unit_id for sap_info in sap_infos]
        all_sap_positions = set()
        for sap_info in sap_infos:
            for sap_pos in sap_info.sap_pos:
                all_sap_positions.add(sap_pos)
        n_cells = len(all_sap_positions)
        if use_py_flow():
            flow = MinimumCostFlow(2 + n_units + n_cells + 1)
        if use_cpp_flow():
            mcf = min_cost_flow.MinimumCostFlow(2 + n_units + n_cells + 1)
        nodes = {}
        nodes["source"] = 0
        nodes["sink"] = 1
        nodes["unit_move"] = 2 + n_units + n_cells  # sap以外の行動のためのノード
        for i in range(n_units):
            nodes[f"unit_{i}"] = 2 + i
        for i, (x, y) in enumerate(all_sap_positions):
            nodes[f"pos_{x}_{y}"] = 2 + n_units + i
            # nodes[f"pos_{x}_{y}_additional"] = 2 + n_units + n_cells + i

        if use_py_flow():
            flow.add_edge(nodes["unit_move"], nodes["sink"], capacity=n_units, cost=0, action=-1)
        if use_cpp_flow():
            mcf.add_edge(nodes["unit_move"], nodes["sink"], n_units, 0, -1)
        for i, unit_id in enumerate(sap_unit_ids):
            if use_py_flow():
                flow.add_edge(nodes["source"], nodes[f"unit_{i}"], capacity=1, cost=0, action=-1)
            if use_cpp_flow():
                mcf.add_edge(nodes["source"], nodes[f"unit_{i}"], 1, 0, -1)
        for pos in all_sap_positions:
            pos_node = nodes[f"pos_{pos[0]}_{pos[1]}"]
            # pos_node_additional = nodes[f"pos_{pos[0]}_{pos[1]}_additional"]
            if use_py_flow():
                flow.add_edge(pos_node, nodes["sink"], capacity=1, cost=0, action=-1)
            if use_cpp_flow():
                mcf.add_edge(pos_node, nodes["sink"], 1, 0, -1)
            # additional_capacity = max(len(available_unit_ids) - 1, 1)  # 最低でも1の容量を確保
            # flow.add_edge(
            #     pos_node_additional,
            #     nodes["sink"],
            #     capacity=additional_capacity,
            #     cost=self.cfg.overlap_penalty,
            #     action=-1,
            # )

        for i, sap_info in enumerate(sap_infos):
            unit_node = nodes[f"unit_{i}"]
            if use_py_flow():
                flow.add_edge(
                    unit_node,
                    nodes["unit_move"],
                    capacity=1,
                    cost=sap_info.next_action["score"],
                    action=sap_info.next_action,
                )
            if use_cpp_flow():
                mcf.add_edge(
                    unit_node,
                    nodes["unit_move"],
                    1,
                    sap_info.next_action["score"],
                    sap_info.next_action,
                )
            for sap_pos, sap_policy in zip(sap_info.sap_pos, sap_info.sap_policy):
                sap_pos_relative = calc_relative_pos(sap_info.unit_pos, sap_pos)
                assert sap_policy >= 0 and sap_policy <= 1
                action = {
                    "action_id": Action.SAP,
                    "next_pos": sap_info.unit_pos,
                    "score": 1.0 - sap_policy,
                    "sap_pos": sap_pos_relative,
                }
                if use_py_flow():
                    flow.add_edge(
                        unit_node,
                        nodes[f"pos_{sap_pos[0]}_{sap_pos[1]}"],
                        capacity=1,
                        cost=action["score"],
                        action=action,
                    )
                if use_cpp_flow():
                    mcf.add_edge(unit_node, nodes[f"pos_{sap_pos[0]}_{sap_pos[1]}"], 1, action["score"], action)

        # try:
        if use_py_flow():
            flow_result = flow.flow(nodes["source"], nodes["sink"], len(sap_unit_ids))
            assert flow_result != -1, "Flow calculation failed"
            assert flow_result != -2, "Flow calculation timeout"
        if use_cpp_flow():
            mcflow_result = mcf.flow(nodes["source"], nodes["sink"], len(sap_unit_ids))
            assert mcflow_result != -1, "Flow calculation failed"
            assert mcflow_result != -2, "Flow calculation timeout"

        if use_py_flow() and use_cpp_flow():
            for i, sap_info in enumerate(sap_infos):
                for original_edge, edge in zip(flow.edges[i], mcf.edges[i]):
                    assert original_edge[1] == edge.cap
                    assert original_edge[4] == edge.action
        # except Exception as e:
        #     print(f"フロー計算エラー: {e}")
        #     self._assign_greedy_actions(actions, available_unit_ids, unit_positions, policy_map, point_map, obs, opp_unit_positions)
        #     return
        # if flow_result == -1:
        #     print("flow=-1", file=sys.stderr)
        #     for i, sap_info in enumerate(sap_infos):
        #         policy_map[Action.SAP, sap_info.unit_pos[1], sap_info.unit_pos[0]] = 0
        #     self._assign_greedy_actions(actions, sap_unit_ids, unit_positions, policy_map, point_map, obs, sap_map)
        #     return

        # フローから行動を抽出
        for i, unit_id in enumerate(sap_unit_ids):
            unit_node = nodes[f"unit_{i}"]
            unit_pos = unit_positions[unit_id]
            selected_action = None
            if use_cpp_flow():
                for edge in mcf.edges[unit_node]:
                    if edge.cap == 0 and edge.action != -1:
                        selected_action = edge.action
                        break
            elif use_py_flow():
                for edge in flow.edges[unit_node]:
                    if edge[1] == 0 and edge[4] != -1:
                        selected_action = edge[4]
                        break
            if selected_action["action_id"] == Action.SAP:
                actions[unit_id] = [Action.SAP, selected_action["sap_pos"][0], selected_action["sap_pos"][1]]
