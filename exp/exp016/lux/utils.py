from enum import IntEnum, auto
from typing import Any

import flax
import numpy as np
import torch

from .params import EnvParams


class State(IntEnum):
    TILE_TYPE = 0  # 0スタート
    ENERGY = auto()
    SENSOR_MASK = auto()
    RELICS = auto()
    POINTS = auto()  # relic nodes周辺のポイントを獲得できるノード
    OWN_UNIT_COUNT = auto()
    OWN_UNIT_ENERGY = auto()
    # OWN_UNIT_MASK = auto()
    OPP_UNIT_COUNT = auto()
    OPP_UNIT_ENERGY = auto()
    # OPP_UNIT_MASK = auto()
    VISIT_COUNT = auto()


class GlobalState(IntEnum):
    MATCH_STEPS = 0
    MATCH_COUNT = auto()
    TEAM_POINTS = auto()
    TEAM_WINS = auto()
    # 環境パラメータからepisodeごとに取得できる
    UNIT_MOVE_COST = auto()
    UNIT_SAP_COST = auto()
    UNIT_SAP_RANGE = auto()
    UNIT_SENSOR_RANGE = auto()


class HiddenState(IntEnum):
    OWN_UNIT_COUNT = 0
    OPP_UNIT_COUNT = auto()
    POINTS = auto()
    ENERGY = auto()


# episodeごとに変動する環境パラメータ
class HiddenGlobalState(IntEnum):
    NEBULA_TILE_VISION_REDUCTION = 0
    NEBULA_TILE_ENERGY_REDUCTION = auto()
    UNIT_SAP_DROPOFF_FACTOR = auto()
    UNIT_ENERGY_VOID_FACTOR = auto()
    NEBULA_TILE_DRIFT_SPEED = auto()
    ENERGY_NODE_DRIFT_SPEED = auto()
    ENERGY_NODE_DRIFT_MAGNITUDE = auto()


class Action(IntEnum):
    CENTER = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4
    SAP = 5


class TileType(IntEnum):
    UKNOWN = -1
    EMPTY = 0
    NEBULA = 1
    ASTEROID = 2


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


class EpisodeStore:
    def __init__(
        self, episode_id: int, target_team_id: int, env_cfg: dict | EnvParams, validation: bool = False
    ) -> None:
        self._init_low_prob = 0.1  # マップ全体に設定されるpoint発生確率
        self._init_high_prob = 0.5  # 可能性があるところに設定されるpoint発生確率
        self._relic_map = np.zeros((EnvParams.map_height, EnvParams.map_width), dtype=np.float32)
        self._point_map = np.ones((EnvParams.map_height, EnvParams.map_width), dtype=np.float32) * self._init_low_prob
        self._visit_count = np.zeros((EnvParams.map_height, EnvParams.map_width), dtype=np.int32)
        self._target_team_id = target_team_id
        self._relic_nodes = set()
        self._is_popup_relic_in_this_match = False
        self.validation = validation
        self.episode_id = episode_id

        if isinstance(env_cfg, EnvParams):
            env_cfg = flax.serialization.to_state_dict(env_cfg)

        self.unit_move_cost = env_cfg["unit_move_cost"]
        self.unit_sap_cost = env_cfg["unit_sap_cost"]
        self.reset()

    def reset(self) -> None:
        # matchが切り替わったらリセットする
        self._prev_points = 0
        self._current_points = 0
        self._is_popup_relic_in_this_match = False

        if not self._is_finished_relic_search():
            # ないと判定されているところも発生する可能性があるため-1にする
            self._point_map = np.where(self._point_map == 0, self._init_low_prob, self._point_map)

    @property
    def point(self) -> int:
        # pointが減ることはないので必ず0以上を返す
        return self._current_points - self._prev_points

    @property
    def relic_map(self) -> np.ndarray:
        return self._relic_map.copy()

    @property
    def point_map(self) -> np.ndarray:
        return self._point_map.copy()

    @property
    def visit_count(self) -> np.ndarray:
        return self._visit_count.copy()

    def update(self, obs: dict[str, Any]) -> None:
        self._update_relic_map(obs)
        self._update_visit_count(obs)
        self._update_points(obs)
        self._update_point_map(obs)

    def _is_finished_relic_search(self) -> bool:
        return self._relic_map.sum() == EnvParams.max_relic_nodes

    def _update_relic_map(self, obs: dict[str, Any]) -> None:
        # # relicの情報を記録する関数
        relic_nodes = {(x, y) for x, y in obs["relic_nodes"] if x != -1 and y != -1}
        new_relic_nodes = relic_nodes - self._relic_nodes
        if len(new_relic_nodes) == 0:
            return

        # 既知のrelic_nodesの範囲の場合Trueにする
        old_point_candidate_mask = np.zeros((EnvParams.map_height, EnvParams.map_width), dtype=np.bool_)
        for x, y in self._relic_nodes:
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    if in_map((nx, ny)):
                        old_point_candidate_mask[ny, nx] = True

        for x, y in new_relic_nodes:
            self._relic_nodes.add((x, y))
            self._relic_map[y, x] = 1
            ox, oy = get_opposite(x, y)
            self._relic_nodes.add((ox, oy))
            self._relic_map[oy, ox] = 1

            # relic_nodesの周囲5マスのpointの下限を0.5にする
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    # 既知のrelic_nodesの範囲&0より大きい場合はスキップ
                    if in_map((nx, ny)):
                        self._point_map[ny, nx] = max(self._point_map[ny, nx], self._init_high_prob)
                        ox, oy = get_opposite(nx, ny)
                        self._point_map[oy, ox] = self._point_map[ny, nx]

        # ここに到達するということは新しいrelic_nodesが発生しているということ
        self._is_popup_relic_in_this_match = True
        # このマッチではrelic_nodesはこれ以上発生しないので可能性が0のところは0にする
        self._point_map = np.where(
            (self._point_map < self._init_high_prob) & (~old_point_candidate_mask), 0, self._point_map
        )

    def _update_visit_count(self, obs: dict[str, Any]) -> None:
        # TODO: gtと一致しないため正確でない
        # 自チームのunitの情報を記録する
        own_unit_positions = np.array(obs["units"]["position"][self._target_team_id])
        for unit_id in range(EnvParams.max_units):
            pos = own_unit_positions[unit_id]
            if pos[0] != -1 and pos[1] != -1:
                self._visit_count[pos[1], pos[0]] += 1 / 100

    def _update_points(self, obs: dict[str, Any]) -> None:
        self._prev_points = self._current_points
        self._current_points = obs["team_points"][self._target_team_id]

    def _extract_unknown_point_positions(
        self, unit_positions: np.ndarray, unit_energies: np.ndarray
    ) -> tuple[set[tuple[int, int]], int]:
        # unit_positionsのうちpointが未知の位置のみ抽出する
        unknown_point_positions = (
            set()
        )  # unitが重複している場合ポイントは１つしか入らないため重複を除外するためにsetを使う
        known_point_positions = set()
        known_point = 0

        for (x, y), energy in zip(unit_positions, unit_energies):
            if x == -1 and y == -1:
                continue
            # 敵からのsapで負のエネルギーになってる場合はポイントは獲得されず次のステップでrestartになる
            if energy < 0:
                continue
            # known_point_positionsに含まれている場合はスキップ(unitが同じセルに重複して存在するケース)
            if (x, y) in known_point_positions:
                continue
            # ポイントセルとして確定している場合は既知ポイントとしてカウント
            if self._point_map[y, x] == 1:
                known_point += 1
                known_point_positions.add((x, y))
                continue
            unknown_point_positions.add((x, y))
        return unknown_point_positions, known_point

    # relic_nodesがこのマッチで確定しているか
    def _is_finished_relic_search_in_this_match(self, match_steps: int) -> bool:
        # relic_nodesが全て見つかっているか、今回のマッチではすでにrelic_nodesが発生しているか、この試合ではrelic_nodesが発生しないことが確定していればpointを確定できる
        return match_steps > 50 or self._is_popup_relic_in_this_match or self._is_finished_relic_search()

    def _update_point_map(self, obs: dict[str, Any]) -> None:
        """
        各マッチ0~50stepの間でrelic_nodesが発生する(しない場合もある)
        - max_relic_nodesは6なので全て見つけたら確定する
        - 50step以降はrelic_nodesは発生しないので探索をやめる
        - matchが切り替わったらrelic_nodesが発生する可能性があれば-1にして探索するようにする。
        """
        # 必ず_update_points, _update_actionsを先に呼び出すこと
        unit_positions = np.array(obs["units"]["position"][self._target_team_id])  # (max_units, 2)
        unit_energies = np.array(obs["units"]["energy"][self._target_team_id])  # (max_units, 1)
        unknown_point_positions, known_point = self._extract_unknown_point_positions(unit_positions, unit_energies)
        unknown_point = self.point - known_point
        # print(
        #     f"{obs['steps']=} {self.point=} {unknown_point=} {self._point_map[y, x]=} {get_opposite(x, y)=} {unknown_point_positions=}"
        # )

        if len(unknown_point_positions) == 0:
            return

        if self._is_finished_relic_search_in_this_match(obs["match_steps"]):
            if self.validation:
                assert unknown_point >= 0

            # 未知のユニット位置で得られるポイントがユニット数と同じなら100%の確率でそこにポイントがあると考える
            if len(unknown_point_positions) == unknown_point:
                posterior_values = np.ones(len(unknown_point_positions))
            # 未知のユニット位置で得られるポイントが0ならそこでポイントが得られないと考える
            elif unknown_point == 0:
                posterior_values = np.zeros(len(unknown_point_positions))

            elif 0 < unknown_point < len(unknown_point_positions):
                prior_probs = np.array([self._point_map[y, x] for x, y in unknown_point_positions])
                likelihood = unknown_point / len(unknown_point_positions)
                posterior_values = bayesian_update(prior_probs, likelihood, unknown_point)
            elif self.validation:
                raise ValueError(
                    f"{self.episode_id=} {obs['steps']=} {unknown_point=} {self.point=} {known_point=} {len(unknown_point_positions)=} is invalid"
                )
            else:
                posterior_values = np.ones(len(unknown_point_positions))

            for i, (x, y) in enumerate(unknown_point_positions):
                self._point_map[y, x] = posterior_values[i]
                ox, oy = get_opposite(x, y)

                self._point_map[oy, ox] = self._point_map[y, x]


def extract_hidden_state(gt_obs: dict[str, Any], target_team_id: int) -> np.ndarray:
    """
    - 自/敵unitの位置と数
    - relic pointの位置
    - energy分布
    """
    state_space_size: int = len(HiddenState)
    state_map = np.zeros((state_space_size, EnvParams.map_width, EnvParams.map_height), dtype=np.float32)

    # state
    for team_id in range(2):
        unit_positions = np.array(gt_obs["units"]["position"][team_id])  # (max_units, 2)
        for unit_id in range(EnvParams.max_units):
            x, y = unit_positions[unit_id]

            if team_id == target_team_id:
                state_map[HiddenState.OWN_UNIT_COUNT, y, x] += 1 / EnvParams.max_units
            else:
                state_map[HiddenState.OPP_UNIT_COUNT, y, x] += 1 / EnvParams.max_units

    state_map[HiddenState.POINTS] = get_gt_point_map(gt_obs)

    state_map[HiddenState.ENERGY] = np.array(gt_obs["map_features"]["energy"]).T / 10  # (24, 24)
    return state_map


def bayesian_update(prior_probs: np.ndarray, likelihood: float, unknown_point: float, eps: float = 1e-13) -> np.ndarray:
    # 対数空間での計算（0を避けるためepsを足す）
    post_probs = np.exp(np.log(prior_probs + eps) + np.log(likelihood + eps))
    post_probs /= post_probs.sum()  # 正規化

    # unknown_pointに応じた期待値分布
    values = post_probs * unknown_point

    # 各位置の値が1を超えた場合、超過分を1未満の箇所に再分配する
    while np.any(values > 1):
        total_excess = np.maximum(values - 1, 0).sum()
        values = np.minimum(values, 1)  # 1を超える部分は1に固定
        mask = values < 1  # 1未満の箇所
        if mask.any():
            # 1未満の箇所に現在の値に比例して余剰分を分配
            weights = values[mask] / values[mask].sum()
            values[mask] += total_excess * weights
    return values


def get_gt_point_map(gt_obs: dict[str, Any]) -> np.ndarray:
    relic_map = np.zeros((24, 24))
    for node_pos, _point_local_map in zip(gt_obs["relic_nodes"], gt_obs["relic_node_configs"]):
        x, y = node_pos
        point_local_map = np.array(_point_local_map, dtype=np.int8)
        h, w = point_local_map.shape
        # assert h == w == 5
        for i, dx in enumerate(range(-2, 3)):
            for j, dy in enumerate(range(-2, 3)):
                nx, ny = x + dx, y + dy
                if in_map((nx, ny)) and relic_map[ny, nx] != 1:  # 1だったら上書きしない
                    relic_map[ny, nx] = point_local_map[i, j]
    return relic_map


def get_opposite(x: int, y: int) -> tuple[int, int]:
    # Returns the mirrored point across the diagonal
    return EnvParams.map_height - y - 1, EnvParams.map_width - x - 1


def mirroring(map2d: np.ndarray, null_value: float = -1.0) -> np.ndarray:
    """
    (24,24)のマップを想定
    -1以外で埋められている部分を反転結果で埋める
    """
    for y in range(EnvParams.map_height):
        for x in range(EnvParams.map_width):
            if map2d[y, x] != null_value:
                ox, oy = get_opposite(x, y)
                map2d[oy, ox] = map2d[y, x]
    return map2d


def extract_gt_state(obs: dict[str, Any], target_team_id: int) -> np.ndarray:
    state_space_size: int = len(State)
    enemy_team_id = 1 - target_team_id
    state_map = np.zeros((state_space_size, EnvParams.map_width, EnvParams.map_height), dtype=np.float32)

    # state
    # map state
    state_map[State.TILE_TYPE] = np.array(obs["map_features"]["tile_type"]).T  # (24, 24)

    # energy nodesの位置は未知(tileのenergyはvisionで観測可能)
    state_map[State.ENERGY] = np.array(obs["map_features"]["energy"]).T / 10  # (24, 24)

    state_map[State.SENSOR_MASK] = np.array(obs["vision_power_map"][target_team_id]).T

    for x, y in obs["relic_nodes"]:
        state_map[State.RELICS, y, x] = 1
    state_map[State.POINTS] = get_gt_point_map(obs)

    # unit state
    for team_id in range(2):
        # 敵チームの情報はvision内にいない限り見れない
        unit_energies = np.array(obs["units"]["energy"][team_id])  # (max_units, 1)
        unit_positions = np.array(obs["units"]["position"][team_id])  # (max_units, 2)
        unit_masks = np.array(obs["units_mask"][team_id])  # (max_units, )
        for unit_id in range(EnvParams.max_units):
            unit_energy = unit_energies[unit_id]
            x, y = unit_positions[unit_id]
            unit_mask = unit_masks[unit_id] * 1
            if x == -1 and y == -1:
                continue

            # 味方同士は重複可能なのでincrementする（敵との重複はないため打ち消し合うことはないはず）
            if team_id == target_team_id:
                # 重複はそんなに発生しないだろうということで正規化はしない
                state_map[State.OWN_UNIT_COUNT, y, x] += 1
                state_map[State.OWN_UNIT_ENERGY, y, x] += unit_energy / EnvParams.init_unit_energy
                state_map[State.OWN_UNIT_MASK, y, x] = unit_mask
            else:
                state_map[State.OPP_UNIT_COUNT, y, x] += 1
                state_map[State.OPP_UNIT_ENERGY, y, x] += unit_energy / EnvParams.init_unit_energy
                state_map[State.OPP_UNIT_MASK, y, x] = unit_mask

    # game state
    state_map[State.MATCH_STEPS] = obs["match_steps"] / EnvParams.max_steps_in_match  # そのマッチの進行度
    state_map[State.MATCH_COUNT] = obs["steps"] // EnvParams.max_steps_in_match  # 何試合目か
    state_map[State.TEAM_POINTS] = (obs["team_points"][target_team_id] - obs["team_points"][enemy_team_id]) / 100
    state_map[State.TEAM_WINS] = (
        obs["team_wins"][target_team_id] - obs["team_wins"][enemy_team_id]
    ) / EnvParams.match_count_per_episode
    return state_map


def extract_state(obs: dict[str, Any], target_team_id: int, episode_store: EpisodeStore) -> np.ndarray:
    state_space_size: int = len(State)
    # enemy_team_id = 1 - target_team_id
    state_map = np.zeros((state_space_size, EnvParams.map_width, EnvParams.map_height), dtype=np.float32)

    # state
    # map state
    state_map[State.TILE_TYPE] = np.array(obs["map_features"]["tile_type"]).T
    state_map[State.TILE_TYPE] = mirroring(state_map[State.TILE_TYPE], null_value=-1)
    # energy nodesの位置は未知(tileのenergyはvisionで観測可能) energy系は正規化の分母をinit_unit_energyにする
    state_map[State.ENERGY] = np.array(obs["map_features"]["energy"]).T / EnvParams.init_unit_energy
    state_map[State.ENERGY] = mirroring(state_map[State.ENERGY], null_value=-0.1)
    state_map[State.SENSOR_MASK] = np.array(obs["sensor_mask"]).T

    state_map[State.RELICS] = episode_store.relic_map
    state_map[State.POINTS] = episode_store.point_map
    state_map[State.VISIT_COUNT] = episode_store.visit_count

    # unit state
    for team_id in range(2):
        # 敵チームの情報はvision内にいない限り見れない
        unit_energies = np.array(obs["units"]["energy"][team_id])  # (max_units, 1)
        unit_positions = np.array(obs["units"]["position"][team_id])  # (max_units, 2)
        unit_masks = np.array(obs["units_mask"][team_id])  # (max_units, )
        if team_id != target_team_id:
            # sensor_maskが1(見える範囲)の場合は0にする。それ以外は0.5
            state_map[State.OPP_UNIT_COUNT] = 0.5 / EnvParams.max_units
            state_map[State.OPP_UNIT_COUNT] *= 1 - state_map[State.SENSOR_MASK]

        # available_unit_ids = np.where(unit_masks)[0]
        for unit_id in range(EnvParams.max_units):
            unit_energy = unit_energies[unit_id]
            x, y = unit_positions[unit_id]
            unit_mask = unit_masks[unit_id] * 1
            if x == -1 and y == -1:
                continue
            # 味方同士は重複可能なのでincrementする（敵との重複はないため打ち消し合うことはないはず）
            if team_id == target_team_id:
                # 重複はそんなに発生しないだろうということで正規化はしない
                state_map[State.OWN_UNIT_COUNT, y, x] += 1 / EnvParams.max_units
                state_map[State.OWN_UNIT_ENERGY, y, x] += unit_energy / EnvParams.init_unit_energy
                # state_map[State.OWN_UNIT_MASK, y, x] = unit_mask
            else:
                state_map[State.OPP_UNIT_COUNT, y, x] += 1 / EnvParams.max_units
                state_map[State.OPP_UNIT_ENERGY, y, x] += unit_energy / EnvParams.init_unit_energy
                # state_map[State.OPP_UNIT_MASK, y, x] = unit_mask
    return state_map


def extract_global_state(obs: dict[str, Any], target_team_id: int, env_params: EnvParams) -> np.ndarray:
    enemy_team_id = 1 - target_team_id
    global_states = np.zeros((len(GlobalState),), dtype=np.float32)
    # game state
    global_states[GlobalState.MATCH_STEPS] = obs["match_steps"] / env_params.max_steps_in_match  # そのマッチの進行度
    global_states[GlobalState.MATCH_COUNT] = (
        obs["steps"] // (env_params.max_steps_in_match + 1)
    ) / env_params.match_count_per_episode  # 何試合目か
    global_states[GlobalState.TEAM_POINTS] = (
        obs["team_points"][target_team_id] - obs["team_points"][enemy_team_id]
    ) / 100
    global_states[GlobalState.TEAM_WINS] = (
        obs["team_wins"][target_team_id] - obs["team_wins"][enemy_team_id]
    ) / env_params.match_count_per_episode

    global_states[GlobalState.UNIT_MOVE_COST] = env_params.unit_move_cost / env_params.init_unit_energy
    global_states[GlobalState.UNIT_SAP_COST] = env_params.unit_sap_cost / env_params.init_unit_energy
    global_states[GlobalState.UNIT_SAP_RANGE] = env_params.unit_sap_range
    global_states[GlobalState.UNIT_SENSOR_RANGE] = env_params.unit_sensor_range
    return global_states


def extract_hidden_global_state(env_params: dict[str, Any]) -> np.ndarray:
    hidden_global_states = np.zeros((len(HiddenGlobalState),), dtype=np.float32)
    hidden_global_states[HiddenGlobalState.NEBULA_TILE_VISION_REDUCTION] = env_params.nebula_tile_vision_reduction
    hidden_global_states[HiddenGlobalState.NEBULA_TILE_ENERGY_REDUCTION] = env_params.nebula_tile_energy_reduction
    hidden_global_states[HiddenGlobalState.UNIT_SAP_DROPOFF_FACTOR] = env_params.unit_sap_dropoff_factor
    hidden_global_states[HiddenGlobalState.UNIT_ENERGY_VOID_FACTOR] = env_params.unit_energy_void_factor
    hidden_global_states[HiddenGlobalState.NEBULA_TILE_DRIFT_SPEED] = env_params.nebula_tile_drift_speed
    hidden_global_states[HiddenGlobalState.ENERGY_NODE_DRIFT_SPEED] = env_params.energy_node_drift_speed
    hidden_global_states[HiddenGlobalState.ENERGY_NODE_DRIFT_MAGNITUDE] = env_params.energy_node_drift_magnitude
    return hidden_global_states


def extract_action(actions: dict[str, Any], obs: dict[str, Any], target_team_id: int) -> tuple[np.ndarray, np.ndarray]:
    action_map = np.zeros((EnvParams.map_width, EnvParams.map_height), dtype=np.float32)
    # unit state
    unit_masks = np.array(obs["units_mask"][target_team_id])  # (max_units, )
    unit_positions = np.array(obs["units"]["position"][target_team_id])  # (max_units, 2)

    available_unit_ids = np.where(unit_masks)[0]
    for unit_id in available_unit_ids:
        x, y = unit_positions[unit_id]
        action_map[y, x] = actions[unit_id][0]

    return action_map


def get_valid_policy_map(obs: dict[str, Any], team_id: int, env_cfg: EnvParams) -> np.ndarray:
    validate_policy_map = np.zeros((len(Action), EnvParams.map_width, EnvParams.map_height), dtype=np.float32)
    tile_type_map = np.array(obs["map_features"]["tile_type"]).T  # (24, 24)
    available_unit_ids = np.where(obs["units_mask"][team_id])[0]
    for unit_id in available_unit_ids:
        pos = tuple(obs["units"]["position"][team_id][unit_id])
        x, y = pos
        energy = obs["units"]["energy"][team_id][unit_id]

        validate_policy_map[:, y, x] = 1  # 行動は一旦全て有効化

        for dir in [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]:
            if not can_move(pos, energy, dir, tile_type_map, env_cfg.unit_move_cost):
                validate_policy_map[dir, y, x] = 0

        if not can_sap(energy, env_cfg.unit_sap_cost):
            validate_policy_map[Action.SAP, y, x] = 0
    return validate_policy_map


def calc_next_pos(pos: tuple[int, int], action: Action) -> tuple[int, int]:
    x, y = pos
    if action == Action.CENTER:
        return x, y
    elif action == Action.UP:
        return x, y - 1
    elif action == Action.RIGHT:
        return x + 1, y
    elif action == Action.DOWN:
        return x, y + 1
    elif action == Action.LEFT:
        return x - 1, y
    elif action == Action.SAP:
        return x, y
    else:
        raise ValueError(f"Invalid action: {action}")


def in_map(pos: tuple[int, int]) -> bool:
    x, y = pos
    return 0 <= x < EnvParams.map_width and 0 <= y < EnvParams.map_height


def can_move(pos: tuple[int, int], energy: int, dir: int, tile_type_map: np.ndarray, unit_move_cost: int) -> bool:
    """
    以下の条件のどれかに該当する場合、Falseを返す。全ての条件を満たす場合、Trueを返す。
    1.グリッド範囲外
    2.保持しているエネルギーより移動コストの方が大きい
    3.ASTEROID_TILEタイルが存在する
    自チームと重なるのは今回は問題ないらしい

    """
    next_pos = calc_next_pos(pos, dir)
    nx, ny = next_pos
    if not in_map(next_pos):
        return False
    if energy < unit_move_cost:
        return False
    if tile_type_map[ny, nx] == TileType.ASTEROID:
        return False
    return True


def can_sap(energy: int, unit_sap_cost: int):
    return energy >= unit_sap_cost
