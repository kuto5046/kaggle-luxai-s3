from enum import Enum
from typing import Any

import numpy as np
import torch

from lux.params import EnvParams


class State(Enum):
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


class Action(Enum):
    CENTER = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4
    SAP = 5


class TileType(Enum):
    UKNOWN = -1
    EMPTY = 0
    NEBULA = 1
    ASTEROID = 2


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def extract_state(obs: dict[str, Any], target_team_id: int) -> np.ndarray:
    state_space_size: int = len(State)
    enemy_team_id = 1 - target_team_id
    state_map = np.zeros((state_space_size, EnvParams.map_width, EnvParams.map_height), dtype=np.float32)

    # state
    # map state
    state_map[State.TILE_TYPE.value] = np.array(obs["map_features"]["tile_type"])  # (24, 24)
    # energy nodesの位置は未知(tileのenergyはvisionで観測可能)
    state_map[State.ENERGY.value] = np.array(obs["map_features"]["energy"]) / 10  # (24, 24)
    state_map[State.SENSOR_MASK.value] = np.array(obs["sensor_mask"][target_team_id])

    for k, (i, j) in enumerate(obs["relic_nodes"]):
        state_map[State.RELIC_NODE.value, i, j] = 1
        state_map[State.RELIC_NODE_MASK.value, i, j] = obs["relic_nodes_mask"][k]

    # unit state
    for team_id in range(2):
        # 敵チームの情報はvision内にいない限り見れない
        unit_energys = np.array(obs["units"]["energy"][team_id])  # (max_units, 1)
        unit_positions = np.array(obs["units"]["position"][team_id])  # (max_units, 2)
        unit_masks = np.array(obs["units_mask"][team_id])  # (max_units, )
        available_unit_ids = np.where(unit_masks)[0]
        for unit_id in available_unit_ids:
            unit_energy = unit_energys[unit_id]
            pos = unit_positions[unit_id]
            # 味方同士は重複可能なのでincrementする（敵との重複はないため打ち消し合うことはないはず）
            if team_id == target_team_id:
                # 重複はそんなに発生しないだろうということで正規化はしない
                state_map[State.UNIT_COUNT.value, pos[0], pos[1]] += 1
                state_map[State.UNIT_ENERGY.value, pos[0], pos[1]] += unit_energy / EnvParams.max_unit_energy
            else:
                state_map[State.UNIT_COUNT.value, pos[0], pos[1]] -= 1
                state_map[State.UNIT_ENERGY.value, pos[0], pos[1]] -= unit_energy / EnvParams.max_unit_energy

    # game state
    state_map[State.MATCH_STEPS.value] = obs["match_steps"] / EnvParams.max_steps_in_match  # そのマッチの進行度
    state_map[State.MATCH_COUNT.value] = obs["steps"] // EnvParams.max_steps_in_match  # 何試合目か
    state_map[State.TEAM_POINTS.value] = (obs["team_points"][target_team_id] - obs["team_points"][enemy_team_id]) / 100
    state_map[State.TEAM_WINS.value] = (
        obs["team_wins"][target_team_id] - obs["team_wins"][enemy_team_id]
    ) / EnvParams.match_count_per_episode
    return state_map


def extract_action(actions: dict[str, Any], obs: dict[str, Any], target_team_id: int) -> tuple[np.ndarray, np.ndarray]:
    action_map = np.zeros((EnvParams.map_width, EnvParams.map_height), dtype=np.float32)
    # unit state
    unit_masks = np.array(obs["units_mask"][target_team_id])  # (max_units, )
    unit_positions = np.array(obs["units"]["position"][target_team_id])  # (max_units, 2)
    unavailable_unit_ids = np.where(~unit_masks)[0]
    for unit_id in unavailable_unit_ids:
        # mask=Falseされているところは必ずaction=0になってるはず
        assert actions[unit_id][0] == 0

    available_unit_ids = np.where(unit_masks)[0]
    for unit_id in available_unit_ids:
        pos = unit_positions[unit_id]
        action_map[pos[0], pos[1]] = actions[unit_id][0]

    return action_map


def get_valid_policy_map(obs: dict[str, Any], team_id: int) -> np.ndarray:
    validate_policy_map = np.zeros((len(Action), EnvParams.map_width, EnvParams.map_height), dtype=np.float32)
    tile_type_map = np.array(obs["map_features"]["tile_type"])  # (24, 24)
    available_unit_ids = np.where(obs["units_mask"][team_id])[0]
    for unit_id in available_unit_ids:
        pos = tuple(obs["units"]["position"][team_id][unit_id])
        energy = obs["units"]["energy"][team_id][unit_id]
        # mask = obs["units_mask"][team_id][unit_idx]  # Trueの場合見えない
        validate_policy_map[:5, pos[0], pos[1]] = 1  # 移動行動は一旦全て有効化
        for dir in range(1, 5):
            if not can_move(pos, energy, dir, tile_type_map):
                validate_policy_map[dir, pos[0], pos[1]] = 0
    return validate_policy_map


def calc_next_pos(pos: tuple[int, int], dir: int) -> tuple[int, int]:
    x, y = pos
    if dir == Action.CENTER.value:
        return x, y
    elif dir == Action.UP.value:
        return x, y - 1
    elif dir == Action.RIGHT.value:
        return x + 1, y
    elif dir == Action.DOWN.value:
        return x, y + 1
    elif dir == Action.LEFT.value:
        return x - 1, y
    else:
        raise ValueError(f"Invalid direction: {dir}")


def in_map(pos: tuple[int, int]) -> bool:
    x, y = pos
    return 0 <= x < EnvParams.map_width and 0 <= y < EnvParams.map_height


def can_move(pos: tuple[int, int], energy: int, dir: int, tile_type_map: np.ndarray) -> bool:
    """
    以下の条件のどれかに該当する場合、Falseを返す。全ての条件を満たす場合、Trueを返す。
    1.グリッド範囲外
    2.保持しているエネルギーより移動コストの方が大きい
    3.ASTEROID_TILEタイルが存在する
    自チームと重なるのは今回は問題ないらしい
    """
    next_pos = calc_next_pos(pos, dir)
    if not in_map(next_pos):
        return False
    if energy < EnvParams.unit_move_cost:
        return False
    if tile_type_map[next_pos[0], next_pos[1]] == TileType.ASTEROID.value:
        return False
    return True
