from heapq import heappop, heappush  # for dijkstra in MinimumCostFlow
from typing import Any, NamedTuple, cast
from pathlib import Path
from collections import deque

import numpy as np
import torch
from lightning import seed_everything
from lux.utils import (
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
    get_valid_policy_per_unit,
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

    tta: bool = False

    checkpoint_path: Path = Path(__file__).parent / "output/best_model.ckpt"


# https://github.com/not522/ac-library-python/blob/master/atcoder/mincostflow.py


class MCFGraph:
    class Edge(NamedTuple):
        src: int
        dst: int
        cap: int
        flow: int
        cost: int

    class _Edge:
        def __init__(self, dst: int, cap: int, cost: int) -> None:
            self.dst = dst
            self.cap = cap
            self.cost = cost
            self.rev: MCFGraph._Edge | None = None

    def __init__(self, n: int) -> None:
        self._n = n
        self._g: list[list[MCFGraph._Edge]] = [[] for _ in range(n)]
        self._edges: list[MCFGraph._Edge] = []

    def add_edge(self, src: int, dst: int, cap: int, cost: int) -> int:
        assert 0 <= src < self._n
        assert 0 <= dst < self._n
        assert 0 <= cap
        m = len(self._edges)
        e = MCFGraph._Edge(dst, cap, cost)
        re = MCFGraph._Edge(src, 0, -cost)
        e.rev = re
        re.rev = e
        self._g[src].append(e)
        self._g[dst].append(re)
        self._edges.append(e)
        return m

    def get_edge(self, i: int) -> Edge:
        assert 0 <= i < len(self._edges)
        e = self._edges[i]
        re = cast(MCFGraph._Edge, e.rev)
        return MCFGraph.Edge(re.dst, e.dst, e.cap + re.cap, re.cap, e.cost)

    def edges(self) -> list[Edge]:
        return [self.get_edge(i) for i in range(len(self._edges))]

    def flow(self, s: int, t: int, flow_limit: int | None = None) -> tuple[int, int]:
        return self.slope(s, t, flow_limit)[-1]

    def slope(self, s: int, t: int, flow_limit: int | None = None) -> list[tuple[int, int]]:
        assert 0 <= s < self._n
        assert 0 <= t < self._n
        assert s != t
        if flow_limit is None:
            flow_limit = cast(int, sum(e.cap for e in self._g[s]))

        dual = [0] * self._n
        prev: list[tuple[int, MCFGraph._Edge] | None] = [None] * self._n

        def refine_dual() -> bool:
            pq = [(0, s)]
            visited = [False] * self._n
            dist: list[int | None] = [None] * self._n
            dist[s] = 0
            while pq:
                dist_v, v = heappop(pq)
                if visited[v]:
                    continue
                visited[v] = True
                if v == t:
                    break
                dual_v = dual[v]
                for e in self._g[v]:
                    w = e.dst
                    if visited[w] or e.cap == 0:
                        continue
                    reduced_cost = e.cost - dual[w] + dual_v
                    new_dist = dist_v + reduced_cost
                    dist_w = dist[w]
                    if dist_w is None or new_dist < dist_w:
                        dist[w] = new_dist
                        prev[w] = v, e
                        heappush(pq, (new_dist, w))
            else:
                return False
            dist_t = dist[t]
            for v in range(self._n):
                if visited[v]:
                    dual[v] -= cast(int, dist_t) - cast(int, dist[v])
            return True

        flow = 0
        cost = 0
        prev_cost_per_flow: int | None = None
        result = [(flow, cost)]
        while flow < flow_limit:
            if not refine_dual():
                break
            f = flow_limit - flow
            v = t
            while prev[v] is not None:
                u, e = cast(tuple[int, MCFGraph._Edge], prev[v])
                f = min(f, e.cap)
                v = u
            v = t
            while prev[v] is not None:
                u, e = cast(tuple[int, MCFGraph._Edge], prev[v])
                e.cap -= f
                assert e.rev is not None
                e.rev.cap += f
                v = u
            c = -dual[s]
            flow += f
            cost += f * c
            if c == prev_cost_per_flow:
                result.pop()
            result.append((flow, cost))
            prev_cost_per_flow = c
        return result


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
        return state.permute(0, 1, 2, 4, 3)

    def transpose_global_state(self, global_state: torch.Tensor) -> torch.Tensor:
        return global_state

    def transpose_policy(self, policy: torch.Tensor) -> torch.Tensor:
        assert policy.dim() == 4
        policy = policy.permute(0, 1, 3, 2)
        policy[:, Action.UP], policy[:, Action.LEFT] = policy[:, Action.LEFT].clone(), policy[:, Action.UP].clone()
        policy[:, Action.DOWN], policy[:, Action.RIGHT] = (
            policy[:, Action.RIGHT].clone(),
            policy[:, Action.DOWN].clone(),
        )
        return policy

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
                output["policy"] = (output["policy"][:1] + self.transpose_policy(output["policy"][1:])) / 2
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

        policy_per_unit = get_legal_policy_per_unit(obs, policy_map, team_id, episode_store)
        point_map = state[State.POINTS]

        return policy_per_unit, point_map, sap


def get_legal_policy_per_unit(
    obs: dict[str, Any], policy_map: np.ndarray, team_id: int, episode_store: EpisodeStore
) -> np.ndarray:
    # shape (EnvParams.max_units, len(Action)) 無効な行動は0その他は1のマスク
    legal_action_per_unit = get_valid_policy_per_unit(obs, team_id, episode_store)
    policy_per_unit = np.zeros_like(legal_action_per_unit, dtype=float)
    unit_positions = np.array(obs["units"]["position"][team_id])
    for unit_id in range(legal_action_per_unit.shape[0]):
        for action_id in range(legal_action_per_unit.shape[1]):
            if legal_action_per_unit[unit_id, action_id] > 0:
                x, y = unit_positions[unit_id]
                policy_per_unit[unit_id, action_id] = policy_map[action_id, y, x]
            else:
                policy_per_unit[unit_id, action_id] = -1e32
        policy_per_unit[unit_id] = softmax(policy_per_unit[unit_id])
    return policy_per_unit


def get_legal_sap_policy(
    obs: dict[str, Any], sap_map: np.ndarray, team_id: int, episode_store: EpisodeStore
) -> np.ndarray:
    legal_sap_map = get_valid_sap_map(obs, team_id, episode_store)
    # 無効な場所は0にする
    sap_map *= legal_sap_map
    return sap_map


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
        policy_per_unit, _point_map, sap_map = imitation_model.predict(obs, self.team_id, self.episode_store, self.cfg)

        unit_mask = np.array(obs["units_mask"][self.team_id])  # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id])  # shape (max_units, 2)
        available_unit_ids = np.where(unit_mask)[0]

        opp_unit_positions = [tuple(pos) for pos in obs["units"]["position"][self.opp_team_id] if pos[0] != -1]

        # sapの位置を決定 -> 残りの行動をフローで決定
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        self._assign_greedy_actions_sap(actions, available_unit_ids, unit_positions, policy_per_unit, sap_map)
        sapped_unit_next_pos_set = set()
        for unit_id in available_unit_ids:
            action = actions[unit_id]
            if action[0] == Action.SAP:
                sapped_unit_next_pos = (unit_positions[unit_id][0], unit_positions[unit_id][1])
                sapped_unit_next_pos_set.add(sapped_unit_next_pos)

        # available_unit_idsからsapしたユニットを除外
        next_available_unit_ids = [unit_id for unit_id in available_unit_ids if actions[unit_id][0] != Action.SAP]

        self._assign_actions_with_flow(
            actions, next_available_unit_ids, unit_positions, policy_per_unit, sapped_unit_next_pos_set
        )

        self.prev_opp_unit_positions = opp_unit_positions
        self.prev_actions = actions
        return actions

    def _assign_actions_with_flow(
        self, actions, available_unit_ids, unit_positions, policy_per_unit, sapped_unit_next_pos_set
    ):
        """
        新しいMCFGraphライブラリを用いて、グリッドへの割り当てを最小費用流問題として解く。
        各ユニットの可能なアクションをエッジとして追加し、後で流量が使われたエッジからアクションを抽出する。
        """
        # 各ユニットの可能なアクションと移動後の位置を列挙
        unit_actions = []
        all_next_positions = set()

        for unit_id in available_unit_ids:
            current_pos = unit_positions[unit_id]
            unit_pos = (int(current_pos[0]), int(current_pos[1]))
            x, y = unit_pos
            policy = policy_per_unit[unit_id].copy()
            if policy.sum() > 0:
                policy = policy / policy.sum()
            else:
                # policyが全て0の場合はCENTERのみを有効にする
                policy = np.zeros_like(policy)
                policy[Action.CENTER] = 1.0

            valid_actions = []
            for action in range(len(Action)):
                # SAPはここでは処理しない
                if action == Action.SAP:
                    continue
                if policy[action] == 0:
                    continue
                next_pos = calc_next_pos(unit_pos, action)
                if not in_map(next_pos):
                    continue
                # 確率が高いほどコストは低くなる（1 - probability）
                score = 1.0 - policy[action]
                valid_actions.append(
                    {
                        "action_id": action,
                        "next_pos": next_pos,
                        "score": score,
                        "sap_pos": None,  # 今回は使わない
                    }
                )
                all_next_positions.add(next_pos)
            unit_actions.append(valid_actions)

        # フローネットワークの構築
        n_units = len(available_unit_ids)
        n_cells = len(all_next_positions)
        total_nodes = 2 + n_units + 2 * n_cells  # source, sink, unitノード、2種類の位置ノード
        flow = MCFGraph(total_nodes)

        nodes = {}
        nodes["source"] = 0
        nodes["sink"] = 1
        for i in range(n_units):
            nodes[f"unit_{i}"] = 2 + i
        # all_next_positions は順序が不定なためソートしておくと安定
        sorted_positions = sorted(all_next_positions)
        for i, (x, y) in enumerate(sorted_positions):
            nodes[f"pos_{x}_{y}"] = 2 + n_units + i
            nodes[f"pos_{x}_{y}_additional"] = 2 + n_units + n_cells + i

        # source -> unitノード
        for i, _ in enumerate(available_unit_ids):
            flow.add_edge(nodes["source"], nodes[f"unit_{i}"], cap=1, cost=0)

        # 位置ノード -> sink
        for pos in sorted_positions:
            pos_node = nodes[f"pos_{pos[0]}_{pos[1]}"]
            pos_node_additional = nodes[f"pos_{pos[0]}_{pos[1]}_additional"]
            if pos not in sapped_unit_next_pos_set:
                flow.add_edge(pos_node, nodes["sink"], cap=1, cost=0)
            # 重なりが発生した場合のペナルティ
            additional_capacity = max(n_units - 1, 1)
            flow.add_edge(pos_node_additional, nodes["sink"], cap=additional_capacity, cost=self.cfg.overlap_penalty)

        # unitノード -> 位置ノード：各ユニットから可能な行動に応じたエッジを追加
        # エッジに対応するアクションを保持するためのマッピング
        action_map = {}
        for i, unit_actions_list in enumerate(unit_actions):
            unit_node = nodes[f"unit_{i}"]
            for act in unit_actions_list:
                pos = act["next_pos"]
                base_cost = act["score"]
                pos_node = nodes[f"pos_{pos[0]}_{pos[1]}"]
                pos_node_additional = nodes[f"pos_{pos[0]}_{pos[1]}_additional"]
                # エッジを2種類追加し、それぞれに対応するアクション情報を保存
                edge_idx1 = flow.add_edge(unit_node, pos_node, cap=1, cost=base_cost)
                action_map[(unit_node, pos_node, edge_idx1)] = act
                edge_idx2 = flow.add_edge(unit_node, pos_node_additional, cap=1, cost=base_cost)
                action_map[(unit_node, pos_node_additional, edge_idx2)] = act

        # 最小費用流を計算（流量はユニット数とする）
        flow_result = flow.flow(nodes["source"], nodes["sink"], len(available_unit_ids))
        # flow_resultは (total_flow, total_cost) のタプル
        if flow_result[0] != len(available_unit_ids):
            # 全ユニットの割り当てが得られなかった場合、デフォルトでCENTERアクションを割り当てる
            for unit_id in available_unit_ids:
                actions[unit_id] = [Action.CENTER, 0, 0]
            return

        # 各ユニットについて、流量が流れたエッジから採用したアクションを決定する
        for i, unit_id in enumerate(available_unit_ids):
            unit_node = nodes[f"unit_{i}"]
            selected_action = None
            # 新ライブラリでは内部グラフは flow._g に保持されている（内部実装に依存します）
            for edge in flow._g[unit_node]:
                # 元々の容量が1なら、flowが使われた場合は edge.cap が 0 になっているはず
                if edge.cap == 0:
                    # key: (unit_node, edge.dst, edge index)
                    # edgeのインデックスは、action_mapのキーに含めた値と一致させる必要があるが、
                    # ここでは一意に (unit_node, edge.dst) で探索します（複数存在する場合は最初のものを採用）
                    for key, act in action_map.items():
                        if key[0] == unit_node and key[1] == edge.dst:
                            selected_action = act
                            break
                    if selected_action is not None:
                        break
            if selected_action is None:
                actions[unit_id] = [Action.CENTER, 0, 0]
            else:
                actions[unit_id] = [selected_action["action_id"], 0, 0]

    def _assign_greedy_actions_sap(self, actions, available_unit_ids, unit_positions, policy_per_unit, sap_map):
        """
        新しいMCFGraphライブラリを用いて、SAPアクションの位置割り当てを最小費用流問題として解く。
        各ユニットについて、SAPアクションを選択した場合、その周辺の候補位置とsap_mapのスコアから
        最適なSAP位置を決定する。
        """
        sap_infos: list[SingleSapInfo] = []

        # 各ユニットごとにSAP候補情報を収集
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            x, y = unit_pos
            policy = policy_per_unit[unit_id].copy()

            if self.cfg.stochastic:
                if policy.sum() == 0:
                    actions[unit_id] = [Action.CENTER, 0, 0]
                    continue
                policy = policy / policy.sum()
                action = np.random.choice(range(6), p=policy)
            else:
                action = policy.argmax()

            if action == Action.SAP:
                sap_poses = []
                sap_policies = []
                # 指定範囲内の各位置についてsap_mapのスコアを取得
                for dx in range(-self.env_cfg["unit_sap_range"], self.env_cfg["unit_sap_range"] + 1):
                    for dy in range(-self.env_cfg["unit_sap_range"], self.env_cfg["unit_sap_range"] + 1):
                        sap_pos = (x + dx, y + dy)
                        if in_map(sap_pos):
                            sap_poses.append(sap_pos)
                            sap_policies.append(sap_map[sap_pos[1], sap_pos[0]] * policy[Action.SAP])
                # SAP候補を除いた上で、他の行動の中から次の行動を選択
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

        # SAP候補がないユニットがなければ終了
        if len(sap_infos) == 0:
            return

        n_units = len(sap_infos)
        sap_unit_ids = [sap_info.unit_id for sap_info in sap_infos]
        all_sap_positions = set()
        for sap_info in sap_infos:
            for sap_pos in sap_info.sap_pos:
                all_sap_positions.add(sap_pos)
        n_cells = len(all_sap_positions)

        # ノード数: source, sink, unitノード(n_units), 位置ノード(n_cells), および unit_move ノード
        total_nodes = 2 + n_units + n_cells + 1
        flow = MCFGraph(total_nodes)
        nodes = {}
        nodes["source"] = 0
        nodes["sink"] = 1
        nodes["unit_move"] = 2 + n_units + n_cells  # SAP以外の行動への割り当て用
        for i in range(n_units):
            nodes[f"unit_{i}"] = 2 + i
        # 位置ノードはソートして安定した順序にする
        sorted_positions = sorted(all_sap_positions)
        for i, (px, py) in enumerate(sorted_positions):
            nodes[f"pos_{px}_{py}"] = 2 + n_units + i

        # エッジに付加するアクション情報を保持する辞書（キー: (src, dst, edge_idx)）
        action_map = {}

        # ノード間のエッジ追加
        # (1) SAP以外の行動用ノードから sink へ
        flow.add_edge(nodes["unit_move"], nodes["sink"], cap=n_units, cost=0)
        # (2) source から各 unit ノードへ
        for i, _ in enumerate(sap_unit_ids):
            flow.add_edge(nodes["source"], nodes[f"unit_{i}"], cap=1, cost=0)
        # (3) 各位置ノードから sink へ
        for pos in sorted_positions:
            pos_node = nodes[f"pos_{pos[0]}_{pos[1]}"]
            flow.add_edge(pos_node, nodes["sink"], cap=1, cost=0)

        # (4) 各ユニットごとに、SAP候補エッジを追加
        for i, sap_info in enumerate(sap_infos):
            unit_node = nodes[f"unit_{i}"]
            # まず、unit_node から unit_move へのエッジ（SAPを選ばなかった場合の代替）
            edge_idx = flow.add_edge(unit_node, nodes["unit_move"], cap=1, cost=sap_info.next_action["score"])
            action_map[(unit_node, nodes["unit_move"], edge_idx)] = sap_info.next_action
            # 次に、各SAP候補位置へのエッジを追加
            for sap_pos, sap_policy in zip(sap_info.sap_pos, sap_info.sap_policy):
                sap_pos_relative = calc_relative_pos(sap_info.unit_pos, sap_pos)
                # スコアはsap_policyに応じて決定（確率が高いほどコストは低くなる）
                action_dict = {
                    "action_id": Action.SAP,
                    "next_pos": sap_info.unit_pos,
                    "score": 1.0 - sap_policy,
                    "sap_pos": sap_pos_relative,
                }
                pos_node = nodes[f"pos_{sap_pos[0]}_{sap_pos[1]}"]
                edge_idx = flow.add_edge(unit_node, pos_node, cap=1, cost=action_dict["score"])
                action_map[(unit_node, pos_node, edge_idx)] = action_dict

        # 最小費用流を計算（流量はSAPを選択したユニット数）
        flow_result = flow.flow(nodes["source"], nodes["sink"], len(sap_unit_ids))
        # flow_resultは (total_flow, total_cost) のタプル
        if flow_result[0] != len(sap_unit_ids):
            # すべてのユニットに割り当てができなかった場合は、デフォルトで CENTER を割り当てる
            for unit_id in sap_unit_ids:
                actions[unit_id] = [Action.CENTER, 0, 0]
            return

        # 各ユニットについて、流れが流れた（すなわち容量が0になった）エッジから採用アクションを抽出する
        for i, unit_id in enumerate(sap_unit_ids):
            unit_node = nodes[f"unit_{i}"]
            selected_action = None
            # action_map のキーから、unit_node から出ているエッジで、対応するエッジが飽和しているものを探す
            for (src, dst, edge_idx), act in action_map.items():
                if src == unit_node:
                    # 新ライブラリでは、該当エッジの残容量が0ならそのエッジは使用済み
                    if flow._edges[edge_idx].cap == 0:
                        selected_action = act
                        break
            if selected_action is not None and selected_action["action_id"] == Action.SAP:
                actions[unit_id] = [Action.SAP, selected_action["sap_pos"][0], selected_action["sap_pos"][1]]
