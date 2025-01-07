from enum import Enum
from typing import Any

import numpy as np

from lux.params import EnvParams


class Action(Enum):
    CENTER = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4
    SAP = 5


def extract_state(step_info: dict[str, Any], target_team_idx: int, state_space_size: int = 13) -> np.ndarray:
    opponent_team_idx = 1 - target_team_idx
    state_map = np.zeros((state_space_size, EnvParams.map_width, EnvParams.map_height), dtype=np.float32)
    obs = step_info[0]["info"]["replay"]["observations"][0]

    # state
    # map state
    state_map[0] = np.array(obs["map_features"]["tile_type"])  # (24, 24)
    state_map[1] = np.array(obs["map_features"]["energy"]) / 10  # (24, 24)

    for i, j in obs["energy_nodes"]:
        state_map[2, i, j] = 1

    for i, j in obs["relic_nodes"]:
        state_map[3, i, j] = 1

    # unit state
    for team_idx in range(2):
        for unit_idx in range(EnvParams.max_units):
            unit_energy = obs["units"]["energy"][team_idx][unit_idx][0]
            pos = obs["units"]["position"][team_idx][unit_idx]
            mask = obs["units_mask"][team_idx][
                unit_idx
            ]  # Trueの場合見えない(TODO: 自チームのunitは状態管理していればわかるのでは？)
            if not mask:
                if team_idx == target_team_idx:
                    state_map[4, pos[0], pos[1]] = 1
                    state_map[5, pos[0], pos[1]] = unit_energy / EnvParams.max_unit_energy
                else:
                    state_map[6, pos[0], pos[1]] = -1
                    state_map[7, pos[0], pos[1]] = -unit_energy / EnvParams.max_unit_energy

    # 自チームのvisionしか得られないはず?
    state_map[8] = np.array(obs["vision_power_map"][target_team_idx])  # (24, 24)

    # game state
    state_map[9] = obs["match_steps"] / EnvParams.max_steps_in_match  # そのマッチの進行度
    state_map[10] = obs["steps"] // EnvParams.max_steps_in_match  # 何試合目か
    state_map[11] = (obs["team_points"][target_team_idx] - obs["team_points"][opponent_team_idx]) / 100
    state_map[12] = (
        obs["team_wins"][target_team_idx] - obs["team_wins"][opponent_team_idx]
    ) / EnvParams.match_count_per_episode
    return state_map


def extract_action(step_info: dict[str, Any], target_team_idx: int) -> tuple[np.ndarray, np.ndarray]:
    action_map = np.zeros((EnvParams.map_width, EnvParams.map_height), dtype=np.float32)
    obs = step_info[0]["info"]["replay"]["observations"][0]

    # unit state
    actions = step_info[target_team_idx]["action"]
    num_units = len(obs["units_mask"][0])
    for unit_idx in range(num_units):
        pos = obs["units"]["position"][target_team_idx][unit_idx]
        # mask = obs['units_mask'][target_team_idx][unit_idx]  # Trueの場合見えない(TODO: 自チームのunitは状態管理していればわかるのでは？)
        if actions:
            # 0 = center, 1 = up, 2 = right, 3 = down, 4 = left, 5 = sap
            # sapの場合座標も学習してみたい
            action_map[pos[0], pos[1]] = actions[unit_idx][0]

    return action_map


def get_valid_policy_map(obs: dict[str, Any], team_idx: int, action_space_size: int = 6) -> np.ndarray:
    validate_policy_map = np.zeros((action_space_size, 48, 48), dtype=np.float32)
    for unit_idx in range(EnvParams.max_units):
        pos = tuple(obs["units"]["position"][team_idx][unit_idx])
        energy = obs["units"]["energy"][team_idx][unit_idx][0]
        mask = obs["units_mask"][team_idx][unit_idx]  # Trueの場合見えない

        validate_policy_map[:5, pos[0], pos[1]] = 1  # 移動行動は一旦全て有効化
        if not mask:
            for dir in range(5):
                if not can_move(pos, energy, dir):
                    validate_policy_map[dir, pos[0], pos[1]] = 0
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


def can_move(pos: tuple[int, int], energy: int, dir: int) -> bool:
    """
    以下の条件のどれかに該当する場合、Falseを返す。全ての条件を満たす場合、Trueを返す。
    1.グリッド範囲外
    2.保持しているエネルギーより移動コストの方が大きい
    3.自チームのunitと衝突する(一旦skip)
    """
    next_pos = calc_next_pos(pos, dir)
    if not in_map(next_pos):
        return False
    if energy < EnvParams.unit_move_cost:
        return False
    return True
