from enum import IntEnum, auto
from typing import Any

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
    OWN_UNIT_MASK = auto()
    OPP_UNIT_COUNT = auto()
    OPP_UNIT_ENERGY = auto()
    OPP_UNIT_MASK = auto()
    MATCH_STEPS = auto()
    MATCH_COUNT = auto()
    TEAM_POINTS = auto()
    TEAM_WINS = auto()


class HiddenState(IntEnum):
    # 分類として扱いたいので全てbinaryで表現する
    OWN_UNIT = 0
    OPP_UNIT = auto()
    POINTS = auto()


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
    def __init__(self, target_team_id: int, env_cfg: dict) -> None:
        self._relic_map = np.zeros((EnvParams.map_height, EnvParams.map_width), dtype=np.float32)
        # self._relic_nodes = None # TODO: relic_nodesを全て発見したらその情報を使ってpoint_mapを更新する(ポイントが絶対に存在しないところがわかる)
        self._point_map = np.ones((EnvParams.map_height, EnvParams.map_width), dtype=np.float32) * -1
        self._target_team_id = target_team_id
        self.unit_move_cost = env_cfg["unit_move_cost"]
        self.unit_sap_cost = env_cfg["unit_sap_cost"]
        self.reset()

    def reset(self) -> None:
        self._own_unit_positions = np.ones((EnvParams.max_units, 2), dtype=np.int32) * -1
        self._own_unit_energies = np.ones(EnvParams.max_units, dtype=np.int32) * -1
        self._prev_unit_actions = np.zeros(EnvParams.max_units)  # 前のstepで移動したユニット
        self._prev_points = 0
        self._current_points = 0

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
    def own_unit_positions(self) -> np.ndarray:
        return self._own_unit_positions.copy()

    @property
    def own_unit_energies(self) -> np.ndarray:
        return self._own_unit_energies.copy()

    def update(self, obs: dict[str, Any], prev_actions: list[list[int]]) -> None:
        self._update_relic_map(obs)
        self._update_actions(prev_actions)
        self._update_own_units(obs)
        self._update_points(obs)
        self._update_point_map(obs)

    def _update_relic_map(self, obs: dict[str, Any]) -> None:
        # # relicの情報を記録する関数
        relic_nodes = obs["relic_nodes"]
        # relic_nodes_mask = obs['relic_nodes_mask']
        for x, y in relic_nodes:
            if x == -1 and y == -1:
                continue
            self._relic_map[y, x] = 1
            ox, oy = get_opposite(x, y)
            self._relic_map[oy, ox] = 1

    def _update_actions(self, prev_actions: dict[str, Any]) -> None:
        if len(prev_actions) == 0:
            return

        self._prev_unit_actions = np.zeros(EnvParams.max_units)
        # actionの情報を記録する関数
        for unit_id in range(EnvParams.max_units):
            self._prev_unit_actions[unit_id] = prev_actions[unit_id][0]

    def _update_own_units(self, obs: dict[str, Any]) -> None:
        # TODO: gtと一致しないため正確でない
        # 自チームのunitの情報を記録する
        own_unit_positions = np.array(obs["units"]["position"][self._target_team_id])
        own_unit_energies = np.array(obs["units"]["energy"][self._target_team_id])
        own_unit_masks = np.array(obs["units_mask"][self._target_team_id])
        tile_type_map = np.array(obs["map_features"]["tile_type"])
        energy_map = np.array(obs["map_features"]["energy"])
        for unit_id in range(EnvParams.max_units):
            pos = own_unit_positions[unit_id]
            unit_energy = own_unit_energies[unit_id]
            unit_mask = own_unit_masks[unit_id]
            map_energy = energy_map[pos[1], pos[0]]
            # 観測可能な場合は観測値をそのままの値を使う
            if unit_mask:
                self._own_unit_positions[unit_id] = pos
                self._own_unit_energies[unit_id] = unit_energy
                continue

            # ここからは現在のステップでは未観測のユニットを扱う
            prev_action = self._prev_unit_actions[unit_id]
            if prev_action == Action.CENTER:
                # 位置は変わらないので何もしない
                self._own_unit_energies[unit_id] += map_energy
            elif prev_action == Action.SAP:
                # 位置は変わらないので何もしない
                if self._own_unit_energies[unit_id] >= self.unit_sap_cost:  # TODO: ちゃんと判定すべき
                    self._own_unit_energies[unit_id] -= self.unit_sap_cost
                self._own_unit_energies[unit_id] += map_energy
            # 現在の位置が不明で、前のステップで移動してるユニットは位置推定を行う
            else:
                # 現在の1つ前の時点でのposを参照する
                prev_pos = self._own_unit_positions[unit_id]
                prev_unit_energy = self._own_unit_energies[unit_id]
                # 前のステップでも未観測の場合は分からないため何もしない
                if prev_pos[0] != -1 and can_move(
                    prev_pos, prev_unit_energy, prev_action, tile_type_map, self.unit_move_cost
                ):
                    self._own_unit_positions[unit_id] = calc_next_pos(prev_pos, prev_action)
                    self._own_unit_energies[unit_id] -= self.unit_move_cost
                    self._own_unit_energies[unit_id] += map_energy

            # マップのエネルギーが未知の場合ユニットのエネルギーも未知なので上書きする
            if map_energy == -1:
                self._own_unit_energies[unit_id] = -1

    def _update_points(self, obs: dict[str, Any]) -> None:
        self._prev_points = self._current_points
        self._current_points = obs["team_points"][self._target_team_id]

    def _update_point_map(self, obs: dict[str, Any]) -> None:
        """
        移動したユニットがいるかをまず考える(いない場合はpointは変動しない)

        """
        # 必ず_update_points, _update_actionsを先に呼び出すこと
        unit_positions = np.array(obs["units"]["position"][self._target_team_id])  # (max_units, 2)
        # unit_positionsのうちpoint_mapが未知の位置のみ抽出する
        unknown_positions = []
        known_point = 0
        for x, y in unit_positions:
            if x == -1 and y == -1:
                continue
            # 非ポイントと確定してる場合ポイント計算はしない
            if self._point_map[y, x] == 0:
                continue
            # ポイントセルとして確定している場合は既知ポイントとしてカウント
            if self._point_map[y, x] == 1:
                known_point += 1
                continue
            unknown_positions.append((x, y))

        if len(unknown_positions) == 0:
            return

        # ポイントが変わらない場合そのユニット位置はpointが発生していない
        if self.point == 0:
            for x, y in unknown_positions:
                self._point_map[y, x] = 0
                ox, oy = get_opposite(x, y)
                self._point_map[oy, ox] = 0
        elif 0 < self.point < EnvParams.max_units:
            # 確定しないところを抽出して按分した場合のmaxをみる
            prob = (self.point - known_point) / len(unknown_positions)
            for x, y in unknown_positions:
                ox, oy = get_opposite(x, y)
                # 過去にも確率値として計算されている場合もあるため最大値をその地点のポイント発生確率とする
                self._point_map[y, x] = max(self._point_map[y, x], prob, self._point_map[oy, ox])
                self._point_map[oy, ox] = self._point_map[y, x]

        # ポイントがmax_unitsに達した場合そのユニット位置はすべてpointが発生している
        elif self.point == EnvParams.max_units:
            for x, y in unknown_positions:
                self._point_map[y, x] = 1
                ox, oy = get_opposite(x, y)
                self._point_map[oy, ox] = 1
        else:
            raise ValueError(f"invalid point: {self.point}")


def extract_hidden_state(gt_obs: dict[str, Any], target_team_id: int) -> np.ndarray:
    """
    - 自/敵unitの位置
    - energy_nodesの位置
    - relic nodeの位置

    以下は難しい or 意義が薄いので一旦やらない
    - unitのエネルギー
    - mapのエネルギー (energy_nodesが分かればとりあえずはいいかな)
    - vision power map(これをboolにしたのが観測可能なsensor mask)
    """
    state_space_size: int = len(HiddenState)
    state_map = np.zeros((state_space_size, EnvParams.map_width, EnvParams.map_height), dtype=np.float32)

    # state
    for team_id in range(2):
        unit_positions = np.array(gt_obs["units"]["position"][team_id])  # (max_units, 2)
        for unit_id in range(EnvParams.max_units):
            x, y = unit_positions[unit_id]

            if team_id == target_team_id:
                state_map[HiddenState.OWN_UNIT, y, x] = 1
            else:
                state_map[HiddenState.OPP_UNIT, y, x] = 1

    state_map[HiddenState.POINTS] = get_gt_point_map(gt_obs)

    return state_map


def get_gt_point_map(gt_obs: dict[str, Any]) -> np.ndarray:
    relic_map = np.zeros((24, 24))
    for node_pos, _reward_map in zip(gt_obs["relic_nodes"], gt_obs["relic_node_configs"]):
        x, y = node_pos
        reward_map = np.array(_reward_map, dtype=bool)
        h, w = reward_map.shape

        # reward_mapの中心座標を計算
        center_y, center_x = h // 2, w // 2

        # mapに挿入する領域の開始・終了座標を計算
        start_y = max(0, y - center_y)
        end_y = min(24, y + (h - center_y))
        start_x = max(0, x - center_x)
        end_x = min(24, x + (w - center_x))

        # reward_mapの対応する部分を切り出す
        map_start_y = center_y - (y - start_y)
        map_end_y = center_y + (end_y - y)
        map_start_x = center_x - (x - start_x)
        map_end_x = center_x + (end_x - x)

        relic_map[start_y:end_y, start_x:end_x] = reward_map[map_start_y:map_end_y, map_start_x:map_end_x]
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
                state_map[State.OWN_UNIT_ENERGY, y, x] += unit_energy / EnvParams.max_unit_energy
                state_map[State.OWN_UNIT_MASK, y, x] = unit_mask
            else:
                state_map[State.OPP_UNIT_COUNT, y, x] += 1
                state_map[State.OPP_UNIT_ENERGY, y, x] += unit_energy / EnvParams.max_unit_energy
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
    enemy_team_id = 1 - target_team_id
    state_map = np.zeros((state_space_size, EnvParams.map_width, EnvParams.map_height), dtype=np.float32)

    # state
    # map state
    state_map[State.TILE_TYPE] = np.array(obs["map_features"]["tile_type"]).T  # (24, 24)
    state_map[State.TILE_TYPE] = mirroring(state_map[State.TILE_TYPE], null_value=-1)
    # energy nodesの位置は未知(tileのenergyはvisionで観測可能)
    state_map[State.ENERGY] = np.array(obs["map_features"]["energy"]).T / 10  # (24, 24)
    state_map[State.ENERGY] = mirroring(state_map[State.ENERGY], null_value=-0.1)
    state_map[State.SENSOR_MASK] = np.array(obs["sensor_mask"]).T

    state_map[State.RELICS] = episode_store.relic_map
    state_map[State.POINTS] = episode_store.point_map

    # unit state
    for team_id in range(2):
        if team_id == target_team_id:
            unit_positions = episode_store.own_unit_positions
            unit_energies = episode_store.own_unit_energies
        else:
            # 敵チームの情報はvision内にいない限り見れない
            unit_energies = np.array(obs["units"]["energy"][team_id])  # (max_units, 1)
            unit_positions = np.array(obs["units"]["position"][team_id])  # (max_units, 2)
        unit_masks = np.array(obs["units_mask"][team_id])  # (max_units, )
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
                state_map[State.OWN_UNIT_ENERGY, y, x] += unit_energy / EnvParams.max_unit_energy
                state_map[State.OWN_UNIT_MASK, y, x] = unit_mask
            else:
                state_map[State.OPP_UNIT_COUNT, y, x] += 1 / EnvParams.max_units
                state_map[State.OPP_UNIT_ENERGY, y, x] += unit_energy / EnvParams.max_unit_energy
                state_map[State.OPP_UNIT_MASK, y, x] = unit_mask

    # game state
    state_map[State.MATCH_STEPS] = obs["match_steps"] / EnvParams.max_steps_in_match  # そのマッチの進行度
    # 0-100は0, 101-201は1, ... としたい
    state_map[State.MATCH_COUNT] = (
        obs["steps"] // (EnvParams.max_steps_in_match + 1)
    ) / EnvParams.match_count_per_episode  # 何試合目か
    state_map[State.TEAM_POINTS] = (obs["team_points"][target_team_id] - obs["team_points"][enemy_team_id]) / 100
    state_map[State.TEAM_WINS] = (
        obs["team_wins"][target_team_id] - obs["team_wins"][enemy_team_id]
    ) / EnvParams.match_count_per_episode
    return state_map


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


def calc_next_pos(pos: tuple[int, int], dir: int) -> tuple[int, int]:
    x, y = pos
    if dir == Action.CENTER:
        return x, y
    elif dir == Action.UP:
        return x, y - 1
    elif dir == Action.RIGHT:
        return x + 1, y
    elif dir == Action.DOWN:
        return x, y + 1
    elif dir == Action.LEFT:
        return x - 1, y
    else:
        raise ValueError(f"Invalid direction: {dir}")


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
