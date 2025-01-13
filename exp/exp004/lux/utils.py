from enum import IntEnum
from typing import Any

import numpy as np
import torch

from lux.params import EnvParams


class State(IntEnum):
    TILE_TYPE = 0
    ENERGY = 1
    SENSOR_MASK = 2
    RELIC_NODE = 3
    RELIC_NODE_MASK = 4
    UNIT_COUNT = 5
    UNIT_ENERGY = 6
    MATCH_STEPS = 7
    MATCH_COUNT = 8
    TEAM_POINTS = 9
    TEAM_WINS = 10


class HiddenState(IntEnum):
    # 分類として扱いたいので全てbinaryで表現する
    OWN_UNIT = 0
    OPP_UNIT = 1
    ENERGY_NODE = 2
    RELIC_NODE = 3


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
    # unit state
    for team_id in range(2):
        unit_positions = np.array(gt_obs["units"]["position"][team_id])  # (max_units, 2)
        for unit_id in range(EnvParams.max_units):
            x, y = unit_positions[unit_id]

            if team_id == target_team_id:
                state_map[HiddenState.OWN_UNIT, y, x] = 1
            else:
                state_map[HiddenState.OPP_UNIT, y, x] = 1

    for x, y in gt_obs["energy_nodes"]:
        state_map[HiddenState.ENERGY_NODE, y, x] = 1

    for x, y in gt_obs["relic_nodes"]:
        state_map[HiddenState.RELIC_NODE, y, x] = 1

    return state_map


def extract_state(obs: dict[str, Any], target_team_id: int) -> np.ndarray:
    state_space_size: int = len(State)
    enemy_team_id = 1 - target_team_id
    state_map = np.zeros((state_space_size, EnvParams.map_width, EnvParams.map_height), dtype=np.float32)

    # state
    # map state
    state_map[State.TILE_TYPE] = np.array(obs["map_features"]["tile_type"]).T  # (24, 24)
    # energy nodesの位置は未知(tileのenergyはvisionで観測可能)
    state_map[State.ENERGY] = np.array(obs["map_features"]["energy"]).T / 10  # (24, 24)
    state_map[State.SENSOR_MASK] = np.array(obs["sensor_mask"][target_team_id]).T

    for k, (x, y) in enumerate(obs["relic_nodes"]):
        state_map[State.RELIC_NODE, y, x] = 1
        state_map[State.RELIC_NODE_MASK, y, x] = obs["relic_nodes_mask"][k]

    # unit state
    for team_id in range(2):
        # 敵チームの情報はvision内にいない限り見れない
        unit_energys = np.array(obs["units"]["energy"][team_id])  # (max_units, 1)
        unit_positions = np.array(obs["units"]["position"][team_id])  # (max_units, 2)
        unit_masks = np.array(obs["units_mask"][team_id])  # (max_units, )
        available_unit_ids = np.where(unit_masks)[0]
        for unit_id in available_unit_ids:
            unit_energy = unit_energys[unit_id]
            x, y = unit_positions[unit_id]
            # 味方同士は重複可能なのでincrementする（敵との重複はないため打ち消し合うことはないはず）
            if team_id == target_team_id:
                # 重複はそんなに発生しないだろうということで正規化はしない
                state_map[State.UNIT_COUNT, y, x] += 1
                state_map[State.UNIT_ENERGY, y, x] += unit_energy / EnvParams.max_unit_energy
            else:
                state_map[State.UNIT_COUNT, y, x] -= 1
                state_map[State.UNIT_ENERGY, y, x] -= unit_energy / EnvParams.max_unit_energy

    # game state
    state_map[State.MATCH_STEPS] = obs["match_steps"] / EnvParams.max_steps_in_match  # そのマッチの進行度
    state_map[State.MATCH_COUNT] = obs["steps"] // EnvParams.max_steps_in_match  # 何試合目か
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
    tile_type_map = np.array(obs["map_features"]["tile_type"])  # (24, 24)
    available_unit_ids = np.where(obs["units_mask"][team_id])[0]
    for unit_id in available_unit_ids:
        pos = tuple(obs["units"]["position"][team_id][unit_id])
        x, y = pos
        energy = obs["units"]["energy"][team_id][unit_id]

        validate_policy_map[:6, y, x] = 1  # 行動は一旦全て有効化
        for dir in range(1, 5):
            if not can_move(pos, energy, dir, tile_type_map, env_cfg.unit_move_cost):
                validate_policy_map[dir, y, x] = 0

        if not can_sap(energy):
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


def can_sap(energy: int):
    # 行動にはcostがかからない
    return True
    # return energy >= EnvParams.unit_sap_cost
