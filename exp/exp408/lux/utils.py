from enum import IntEnum, auto
from typing import Any

import flax
import numpy as np
import torch

from .params import EnvParams, env_params_ranges


class State(IntEnum):
    TILE_TYPE = 0  # 0スタート
    NEXT_TILE_TYPE = auto()
    ENERGY = auto()
    NEBULA_ENERGY_REDUCTION = auto()
    SENSOR_MASK = auto()
    # VISION_POWER_MAP = auto()
    RELICS = auto()
    POINTS = auto()  # relic nodes周辺のポイントを獲得できるノード
    ENTROPY = auto()
    OWN_ACTING_UNIT_POSITION = auto()  # 次に行動するユニットの位置
    OWN_NOT_ACTED_UNIT_POSITION = auto()  # まだ行動していないユニットの位置
    OWN_ACTED_UNIT_NEXT_POSITION = auto()  # すでに行動済みのユニットの次の位置
    SAP_SCHEDULED_POSITION = auto()  # sapを予定している位置 大きさはは回数
    OWN_UNIT_COUNT = auto()
    OWN_UNIT_ENERGY = auto()
    # OWN_UNIT_MASK = auto()
    OPP_UNIT_COUNT = auto()
    OPP_UNIT_ENERGY = auto()
    # OPP_UNIT_MASK = auto()
    VISIT_COUNT = auto()
    SAP_AVAILABLE_AREA = auto()  # sapを使用できるエリア


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
    NEBULA_TILE_VISION_REDUCTION_MEAN = auto()  # 推定値
    NEBULA_TILE_VISION_REDUCTION_SIGMA = auto()  # 推定精度
    ENERGY_NODE_DRIFT_SPEED_MEAN = auto()  # 推定値
    ENERGY_NODE_DRIFT_SPEED_SIGMA = auto()  # 推定精度
    # ENERGY_VOID_FACTOR_MEAN = auto()  # 推定値 推定可能だが未実装
    # ENERGY_VOID_FACTOR_SIGMA = auto()  # 推定精度
    NEBULA_TILE_ENERGY_REDUCTION_MEAN = auto()  # 推定値 二次元特徴量とは別に渡しているが必要かは不明
    NEBULA_TILE_ENERGY_REDUCTION_SIGMA = auto()  # 推定精度
    UNIT_SAP_DROPOFF_FACTOR_MEAN = auto()  # 推定値
    UNIT_SAP_DROPOFF_FACTOR_SIGMA = auto()  # 推定精度


class HiddenState(IntEnum):
    # OWN_UNIT_COUNT = 0
    OPP_UNIT_COUNT = 0
    POINTS = auto()


# episodeごとに変動する環境パラメータ
class HiddenGlobalState(IntEnum):
    # NEBULA_TILE_VISION_REDUCTION = 0
    # NEBULA_TILE_ENERGY_REDUCTION = 0
    # UNIT_SAP_DROPOFF_FACTOR = auto()
    # UNIT_ENERGY_VOID_FACTOR = auto()
    # NEBULA_TILE_DRIFT_SPEED = auto() . # 推定可能なので不要
    # ENERGY_NODE_DRIFT_SPEED = auto()
    ENERGY_NODE_DRIFT_MAGNITUDE = 0  # 多分推定できるが重要度は低いと想定


class Action(IntEnum):
    CENTER = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4
    SAP = 5


class TileType(IntEnum):
    UNKNOWN = -1
    EMPTY = 0
    NEBULA = 1
    ASTEROID = 2


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


class EnergyNodeGuesser:
    def __init__(self) -> None:
        self._energy_node_candidates = [
            (j, i)
            for i in range(EnvParams.map_height)
            for j in range(EnvParams.map_width)
            if i + j <= EnvParams.map_height - 1
        ]
        # (bool, int)のタプルで、boolはその場所のエネルギーが取得済みかどうか、intはその場所のエネルギー量 二次元リスト
        self._energy_map = np.empty((EnvParams.map_height, EnvParams.map_width), dtype=object)
        for y in range(EnvParams.map_height):
            for x in range(EnvParams.map_width):
                self._energy_map[y, x] = (False, 0)

        self._energy_func = lambda d: np.sin(d * 1.2 + 1) * 4
        self._energy_tile_patterns = self.precalculate_energy_tile_pattern()

        self._drift_speed_prob = [
            1 / len(env_params_ranges["energy_node_drift_speed"])
            for _ in range(len(env_params_ranges["energy_node_drift_speed"]))
        ]
        # fmt: off
        self.ok_drift_steps = [
            {2,102,202,302,402,502},
            {2,52,102,152,202,252,302,352,402,452,502},
            {2,36,69,102,136,169,202,236,269,302,336,369,402,436,469,502},
            {2,27,52,77,102,127,177,202,227,252,277,302,327,352,377,402,427,452,477,502},
            {2,22,42,62,82,102,122,142,162,182,202,222,242,262,282,302,322,342,362,382,402,422,442,462,482,502}
        ]
        # fmt: on
        # ほとんどは上記でカバーされるが、episodeId 66954207でステップ203でdriftが発生するケースがあったので数値誤差を考慮して前後1ステップを追加
        # もし他にも落ちるようであれば毎ターン確認するようにしたほうがいいかも
        self._drift_steps = set()
        for step_set in self.ok_drift_steps:
            for i in step_set:
                self._drift_steps.add(i)
                self._drift_steps.add(i - 1)
                self._drift_steps.add(i + 1)

    def precalculate_energy_tile_pattern(self):
        energy_tile_patterns = np.empty((EnvParams.map_height, EnvParams.map_width), dtype=object)
        for y in range(EnvParams.map_height):
            for x in range(EnvParams.map_width):
                energy_field = np.zeros((6, EnvParams.map_height, EnvParams.map_width), dtype=np.float32)
                for y2 in range(EnvParams.map_height):
                    for x2 in range(EnvParams.map_width):
                        d = np.linalg.norm(np.array([x, y]) - np.array([x2, y2]))
                        opposite = get_opposite(x, y)
                        d_opposite = np.linalg.norm(np.array(opposite) - np.array([x2, y2]))
                        energy_field[0, y2, x2] = self._energy_func(d)
                        energy_field[3, y2, x2] = self._energy_func(d_opposite)
                        # 他は0のまま
                mean = energy_field.mean()
                if mean < 0.25:
                    energy_field += 0.25 - mean
                energy_field = np.round(energy_field.sum(axis=0)).astype(np.int16)
                energy_field = np.clip(energy_field, EnvParams.min_energy_per_tile, EnvParams.max_energy_per_tile)
                energy_tile_patterns[y, x] = energy_field

        return energy_tile_patterns

    def _will_drift(self, obs: dict[str, Any]) -> bool:
        return obs["steps"] in self._drift_steps

    def _drift_energy_node(self, obs: dict[str, Any]) -> None:
        # driftさせる
        next_energy_node_candidates = []
        for dx in range(
            -max(env_params_ranges["energy_node_drift_magnitude"]),
            max(env_params_ranges["energy_node_drift_magnitude"]) + 1,
        ):
            for dy in range(
                -max(env_params_ranges["energy_node_drift_magnitude"]),
                max(env_params_ranges["energy_node_drift_magnitude"]) + 1,
            ):
                for i, (x, y) in enumerate(self._energy_node_candidates):
                    nx, ny = x + dx, y + dy
                    if in_map((nx, ny)) and (nx + ny <= EnvParams.map_height - 1):
                        next_energy_node_candidates.append((nx, ny))
        self._energy_node_candidates = next_energy_node_candidates

        # energy_mapを更新
        for y in range(EnvParams.map_height):
            for x in range(EnvParams.map_width):
                self._energy_map[y, x] = (False, 0)

    def get_energy_map(self) -> np.ndarray:
        return np.array([[value for _, value in row] for row in self._energy_map])

    def is_determistic(self) -> bool:
        return len(self._energy_node_candidates) == 1

    def _update_energy_map(self, obs: dict[str, Any]) -> None:
        sensor_mask = np.array(obs["sensor_mask"]).T
        obs_energy_map = np.array(obs["map_features"]["energy"]).T
        # energy driftが発生しているかどうか
        # 全ての候補有効な候補でenergy driftが発生するなら今回のステップでenergy driftが発生する
        # 過去の履歴からenergy driftが発生しているかどうかを判断することも可能だがdx,dy = (0, 0)のドリフトによって外すことがあるので一旦やめている
        if self._will_drift(obs):
            self._drift_energy_node(obs)

        # energy nodeの情報を更新
        # 全ての候補についてまだ正しいかどうか検証
        # energy_mapが未定でsensor_maskがTrueの場所のみ検証
        points_to_check = [
            (x, y)
            for y in range(EnvParams.map_height)
            for x in range(EnvParams.map_width)
            if sensor_mask[y, x] and not self._energy_map[y, x][0]
        ]
        next_candidates = set()
        for energy_node in self._energy_node_candidates:
            ok = True
            for check_x, check_y in points_to_check:
                if (
                    obs_energy_map[check_y, check_x]
                    != self._energy_tile_patterns[energy_node[1], energy_node[0]][check_y, check_x]
                ):
                    ok = False
                    break
            if ok:
                next_candidates.add(energy_node)

        self._energy_node_candidates = next_candidates
        assert len(self._energy_node_candidates) > 0

        # energy_mapを更新
        result_by_node_candidates = np.zeros(len(self._energy_node_candidates), dtype=np.float32)
        for y in range(EnvParams.map_height):
            for x in range(EnvParams.map_width):
                for candidate_idx, (node_x, node_y) in enumerate(self._energy_node_candidates):
                    result_by_node_candidates[candidate_idx] = self._energy_tile_patterns[node_y, node_x][y, x]
                # 全部が一致している場合はenergy_mapを更新
                if all(result_by_node_candidates == result_by_node_candidates[0]):
                    self._energy_map[y, x] = (True, result_by_node_candidates[0])
                else:
                    # 平均
                    self._energy_map[y, x] = (False, result_by_node_candidates.mean())

    def _bayesian_update(self, priors: list, likelihoods: list) -> list:
        """
        ベイズ更新を行います。
        :param priors: 各候補の事前確率 (shape=(5,), 合計は1)
        :param likelihoods: 各候補の尤度 (shape=(5,))。各要素は 0, 1, 1/n のいずれか。
        :return: 更新後の確率分布 (事後確率)
        """
        # 事前確率と尤度の積を計算
        post = priors * likelihoods
        total = post.sum()
        # 全候補の尤度がゼロの場合は例外とする（またはそのまま事前を返す）
        if total == 0:
            # driftするターンが間違っている場合は稀にここに入る
            # この場合はそのまま事前を返す
            return priors
        return post / total

    def _update_energy_speed(self, obs: dict[str, Any]) -> None:
        # まずenergyの位置が移動しているかどうかを判断する
        current_energy_map = np.array(obs["map_features"]["energy"]).T
        sensor_mask = np.array(obs["sensor_mask"]).T
        drifted = False
        for y in range(EnvParams.map_height):
            for x in range(EnvParams.map_width):
                if self._energy_map[y, x][0] and sensor_mask[y, x]:
                    if current_energy_map[y, x] != self._energy_map[y, x][1]:
                        drifted = True
                        break
            if drifted:
                break

        # 今回の観測が得られた条件下での各候補の確率 = 事前確率 * speedが該当の場合に今回の観測が得られる確率 / normalizing constant
        likelihoods = np.zeros(len(self._drift_speed_prob))
        max_magnitude = max(env_params_ranges["energy_node_drift_magnitude"])
        non_move_prob = 1 / (2 * max_magnitude + 1) ** 2
        for i in range(len(self._drift_speed_prob)):
            if obs["steps"] in self.ok_drift_steps[i]:
                if drifted:
                    # 動くはずで動いている場合
                    likelihoods[i] = 1 - non_move_prob
                else:
                    # 動くはずで動いていない場合
                    likelihoods[i] = non_move_prob
            elif drifted:
                # 動かないはずで動いている場合
                likelihoods[i] = 0
            else:
                # 動かないはずで動いていない場合
                likelihoods[i] = 1

        self._drift_speed_prob = self._bayesian_update(self._drift_speed_prob, likelihoods)

    def update_energy(self, obs: dict[str, Any]) -> None:
        self._update_energy_speed(obs)
        self._update_energy_map(obs)

    # meanとsigmaを返す
    def get_energy_drft_speed_estimate(self) -> tuple[float, float]:
        sum = 0
        sigma = 0
        for i, prob in enumerate(self._drift_speed_prob):
            sum += prob * env_params_ranges["energy_node_drift_speed"][i]
        for i, prob in enumerate(self._drift_speed_prob):
            sigma += prob * (env_params_ranges["energy_node_drift_speed"][i] - sum) ** 2
        sigma = np.sqrt(sigma)
        return (sum, sigma)


class NebulaTileVisionReductionGuesser:
    def __init__(self, target_team_id: int, unit_sensor_range: int) -> None:
        self._nebula_tile_vision_reduction_candidates = env_params_ranges["nebula_tile_vision_reduction"]
        self._target_team_id = target_team_id
        self._unit_sensor_range = unit_sensor_range
        self._prev_tile_type_obs = None

    def compute_sensor_masks(self, obs: dict[str, Any]):
        """Compute the vision power and sensor mask for both teams

        Algorithm:

        generate a integer vision power array over the map.
        For each unit, add unit sensor range value (its kind of like the units sensing power/depth) to each tile the unit's sensor range
        Clamp the vision power array to range [0, unit_sensing_range].

        With 2 vision power maps, take the nebula vision mask * nebula vision power and subtract it from the vision power maps.
        This forms the sensor mask
        """

        max_sensor_range = self._unit_sensor_range
        vision_power_map_padding = max_sensor_range
        vision_power_map = np.zeros(
            shape=(
                EnvParams.map_height + 2 * vision_power_map_padding,
                EnvParams.map_width + 2 * vision_power_map_padding,
            ),
            dtype=np.int16,
        )

        # Update sensor mask based on the sensor range
        def update_vision_power_map(unit_pos, vision_power_map):
            x, y = unit_pos
            start_x = x - max_sensor_range + vision_power_map_padding
            start_y = y - max_sensor_range + vision_power_map_padding
            # 対象領域のサイズは (max_sensor_range * 2 + 1) × (max_sensor_range * 2 + 1)
            slice_size = max_sensor_range * 2 + 1
            # Python のスライスを用いて、対象領域を抽出
            existing_vision_power = vision_power_map[start_x : start_x + slice_size, start_y : start_y + slice_size]

            update = np.zeros_like(existing_vision_power)
            for i in range(max_sensor_range + 1):
                val = np.where(
                    i > max_sensor_range - self._unit_sensor_range - 1,
                    i + 1 - (max_sensor_range - self._unit_sensor_range),
                    0,
                ).astype(np.int16)
                update[i : max_sensor_range * 2 + 1 - i, i : max_sensor_range * 2 + 1 - i] = val
            # vision of position at center of update has an extra 10
            update[max_sensor_range, max_sensor_range] += 10
            new_region = existing_vision_power + update
            vision_power_map[start_x : start_x + slice_size, start_y : start_y + slice_size] = new_region
            return vision_power_map

        own_unit_positions = np.array(obs["units"]["position"][self._target_team_id])

        for unit_id in range(EnvParams.max_units):
            pos = own_unit_positions[unit_id]
            if pos[0] != -1 and pos[1] != -1:
                # 転置しているのでx,yが逆
                pos = (pos[1], pos[0])
                vision_power_map = update_vision_power_map(pos, vision_power_map)

        vision_power_map = vision_power_map[
            vision_power_map_padding:-vision_power_map_padding,
            vision_power_map_padding:-vision_power_map_padding,
        ]
        return vision_power_map

    def _update_nebula_tile_vision_reduction(self, obs: dict[str, Any]) -> None:
        if self._nebula_tile_vision_reduction_candidates == 1:
            return
        if self._prev_tile_type_obs is None:
            self._prev_tile_type_obs = np.array(obs["map_features"]["tile_type"]).T
            return

        vision_power_map = self.compute_sensor_masks(
            obs,
        )

        sensor_mask = np.array(obs["sensor_mask"]).T

        for y in range(EnvParams.map_height):
            for x in range(EnvParams.map_width):
                if len(self._nebula_tile_vision_reduction_candidates) == 1:
                    break
                new_candidates = set()
                if sensor_mask[y, x]:
                    # 前のターンのtile_type_mapに基づいて視界が決まることに注意
                    if self._prev_tile_type_obs[y, x] == TileType.NEBULA:
                        # nebulaの影響があっても見える
                        max_vision_reduction = vision_power_map[y, x] - 1
                        # 候補をしぼる
                        for candidate in self._nebula_tile_vision_reduction_candidates:
                            if max_vision_reduction >= candidate:
                                new_candidates.add(candidate)
                elif vision_power_map[y, x] > 0:
                    # 本来は見えるはずの場所が見えない場合はそこにnebula tileがあるとわかる
                    min_vision_reduction = vision_power_map[y, x]
                    # 候補をしぼる
                    for candidate in self._nebula_tile_vision_reduction_candidates:
                        if min_vision_reduction <= candidate:
                            new_candidates.add(candidate)
                # TODO: 100試合に1回程度(バグっていて？)0になるので直す
                if len(new_candidates) != 0:
                    self._nebula_tile_vision_reduction_candidates = new_candidates

        self._prev_tile_type_obs = np.array(obs["map_features"]["tile_type"]).T

    def get_nebula_tile_vision_reduction_estimate(self) -> tuple[float, float]:
        sum = 0
        for candidate in self._nebula_tile_vision_reduction_candidates:
            sum += candidate
        mean = sum / len(self._nebula_tile_vision_reduction_candidates)
        sigma = 0
        for candidate in self._nebula_tile_vision_reduction_candidates:
            sigma += (candidate - mean) ** 2
        sigma = np.sqrt(sigma / len(self._nebula_tile_vision_reduction_candidates))
        return (mean, sigma)


def get_action_cost(action: Action, unit_energy: int, unit_sap_cost: int, unit_move_cost: int) -> int:
    if action == Action.SAP:
        if unit_energy >= unit_sap_cost:
            action_cost = unit_sap_cost
        else:
            action_cost = 0
    elif action == Action.CENTER:
        action_cost = 0
    elif unit_energy >= unit_move_cost:
        action_cost = unit_move_cost
    else:
        action_cost = 0
    return action_cost


# sap dropoff factor/energy void factorの推定
class EnergyAttackFactorGuesser:
    def __init__(self, target_team_id: int, unit_sap_cost: int, unit_move_cost: int) -> None:
        # TODO: ロジックの確認が終わったら、すべての候補を覚えておいて平均を使う方が安全かもしれない
        self._sap_dropoff_factor = None
        self._unit_energy_void_factor_candidates = env_params_ranges["unit_energy_void_factor"]
        self._target_team_id = target_team_id
        self._unit_sap_cost = unit_sap_cost
        self._unit_move_cost = unit_move_cost
        # fmt off
        self._adj_8_vec = [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]

    # fmt on

    def _update_energy_drop_factor(
        self, obs: dict[str, Any], prev_obs: dict[str, Any], actions: np.ndarray, energy_node_guesser: EnergyNodeGuesser
    ) -> None:
        if self._sap_dropoff_factor is not None:
            return

        unit_positions = np.array(obs["units"]["position"])  # (max_units, 2)
        prev_unit_positions = np.array(prev_obs["units"]["position"])
        unit_energies = np.array(obs["units"]["energy"])  # (max_units, 1)
        prev_unit_energies = prev_obs["units"]["energy"]
        tile_type_map = np.array(prev_obs["map_features"]["tile_type"]).T
        prev_energy_map = np.array(prev_obs["map_features"]["energy"]).T

        unit_positions_set = set()
        prev_unit_positions_set = set()
        for unit_id in range(EnvParams.max_units):
            if unit_positions[self._target_team_id][unit_id][0] != -1:
                unit_positions_set.add(
                    (unit_positions[self._target_team_id][unit_id][0], unit_positions[self._target_team_id][unit_id][1])
                )
            if prev_unit_positions[self._target_team_id][unit_id][0] != -1:
                prev_unit_positions_set.add(
                    (
                        prev_unit_positions[self._target_team_id][unit_id][0],
                        prev_unit_positions[self._target_team_id][unit_id][1],
                    )
                )

        my_sap_count = np.zeros_like(tile_type_map)
        my_adj_sap_count = np.zeros_like(tile_type_map)
        for unit_id, action in enumerate(actions):
            (prev_x, prev_y) = prev_unit_positions[self._target_team_id][unit_id]
            # 有効なshipである (今のターンはいなくてもよい)
            if prev_x == -1 or prev_y == -1:
                continue

            if action[0] == Action.SAP and can_sap(
                prev_unit_energies[self._target_team_id][unit_id], self._unit_sap_cost
            ):
                sap_pos = (prev_x + action[1], prev_y + action[2])
                # env.pyのl.333-334相当
                if not in_map(sap_pos):
                    continue
                my_sap_count[sap_pos[1], sap_pos[0]] += 1
                for dx, dy in self._adj_8_vec:
                    nx = sap_pos[0] + dx
                    ny = sap_pos[1] + dy
                    if in_map((nx, ny)):
                        my_adj_sap_count[ny, nx] += 1

        # energy void の影響がなく、前後で位置が確定している相手ユニットからsap drop off factorを推定
        for unit_id in range(EnvParams.max_units):
            # 位置が確定している
            (x, y) = unit_positions[1 - self._target_team_id][unit_id]
            (prev_x, prev_y) = prev_unit_positions[1 - self._target_team_id][unit_id]
            if x == -1 or y == -1 or prev_x == -1 or prev_y == -1:
                continue
            # respawnしてない
            if (x == 0 and y == 0) or (x == EnvParams.map_width - 1 and y == EnvParams.map_height - 1):
                continue
            # 移動してないと行動コストがわからないので無視
            if (x, y) == (prev_x, prev_y):
                continue
            # 隣接にsapしてないとわからない
            if my_adj_sap_count[y, x] == 0:
                continue
            # energy void の影響がない(移動後の位置で判定)
            # 衝突の可能性がない(移動前の位置で判定)
            adjacent = False
            for dx in [-1, 1]:
                if (x + dx, y) in unit_positions_set:
                    adjacent = True
                if (prev_x + dx, prev_y) in prev_unit_positions_set:
                    adjacent = True
            for dy in [-1, 1]:
                if (x, y + dy) in unit_positions_set:
                    adjacent = True
                if (prev_x, prev_y + dy) in prev_unit_positions_set:
                    adjacent = True

            if adjacent:
                continue

            # nebula energy reductionの影響があるところは一旦無視
            # TODO: あった方が正確かもしれない (energy reductionの精度次第)
            if tile_type_map[y, x] == TileType.NEBULA:
                continue

            if energy_node_guesser._energy_map[y, x][0] is False:
                assert prev_energy_map[y, x] == -1
                continue

            # 今のエネルギー - 前のエネルギー = energy_map (-nebula reduction) - sap(中央)*回数 - sap(8隣接)*回数 - 行動のコスト
            # sap(8隣接)*回数 = energ_map (- nebula reduction) - sap(中央)*回数 - 行動のコスト + 前のエネルギー - 今のエネルギー
            prev_energy_field = energy_node_guesser._energy_map[y, x][1]
            sap_adj_sum = (
                -my_sap_count[y, x] * self._unit_sap_cost
                - self._unit_move_cost
                + prev_unit_energies[1 - self._target_team_id][unit_id]
                - unit_energies[1 - self._target_team_id][unit_id]
            )

            # 0未満のものはfieldからenergyを得られない
            if unit_energies[1 - self._target_team_id][unit_id] >= 0:
                sap_adj_sum += prev_energy_field

            # print(f"{sap_adj_sum=} {obs['steps']=} {x=}, {y=}, {prev_energy_field=}, {my_sap_count[y, x]=}, {self._unit_sap_cost=}, {self._unit_move_cost=}, {prev_unit_energies[1 - self._target_team_id][unit_id]=}, {unit_energies[1 - self._target_team_id][unit_id]=} {my_adj_sap_count[y, x]=}")
            sap_dropoff_factor = sap_adj_sum / (self._unit_sap_cost * my_adj_sap_count[y, x])

            # candidateから一番近いものを選ぶ
            min_diff = 1e9
            for candidate in env_params_ranges["unit_sap_dropoff_factor"]:
                diff = abs(candidate - sap_dropoff_factor)
                if diff < min_diff:
                    min_diff = diff
                    self._sap_dropoff_factor = candidate
            # assert min_diff < 1e-1, f"guess {sap_dropoff_factor=} is too far"
            return

    def _update_energy_void_factor(
        self, obs: dict[str, Any], prev_obs: dict[str, Any], actions: np.ndarray, energy_node_guesser: EnergyNodeGuesser
    ) -> None:
        # 0ステップ目は計算できないのでskip
        if prev_obs is None:
            return
        # TODO

    def update(
        self, obs: dict[str, Any], prev_obs: dict[str, Any], actions: np.ndarray, energy_node_guesser: EnergyNodeGuesser
    ) -> None:
        # 0ステップ目は計算できないのでskip
        if prev_obs is None:
            return

        self._update_energy_drop_factor(obs, prev_obs, actions, energy_node_guesser)
        self._update_energy_void_factor(obs, prev_obs, actions, energy_node_guesser)

    def get_sap_dropoff_factor_estimate(self) -> tuple[float, float]:
        if self._sap_dropoff_factor is None:
            return (
                np.mean(env_params_ranges["unit_sap_dropoff_factor"]),
                np.std(env_params_ranges["unit_sap_dropoff_factor"]),
            )
        else:
            return (self._sap_dropoff_factor, 0)


class EpisodeStore:
    def __init__(
        self,
        target_team_id: int,
        env_cfg: dict | EnvParams,
        validation: bool = False,
        episode_id: int | None = None,
    ) -> None:
        self._init_low_prob = 0.1  # マップ全体に設定されるpoint発生確率
        self._init_high_prob = 0.5  # 可能性があるところに設定されるpoint発生確率
        self._relic_map = np.zeros((EnvParams.map_height, EnvParams.map_width), dtype=np.float32)
        self._point_map = np.ones((EnvParams.map_height, EnvParams.map_width), dtype=np.float32) * self._init_low_prob
        self._tile_type_map = np.ones((EnvParams.map_height, EnvParams.map_width), dtype=np.float32) * TileType.UNKNOWN
        self._vision_power_map = np.zeros((EnvParams.map_height, EnvParams.map_width), dtype=np.int16)
        self._next_tile_type_map = (
            np.ones((EnvParams.map_height, EnvParams.map_width), dtype=np.float32) * TileType.UNKNOWN
        )
        self._nebula_tile_drift_speed_candidates = set(env_params_ranges["nebula_tile_drift_speed"])
        self.speed_to_step = {0.15: 7, 0.1: 10, 0.05: 20, 0.025: 40}
        self.step_to_speed = {v: k for k, v in self.speed_to_step.items()}
        self._nebula_energy_reduction = None

        self.energy_node_guesser = EnergyNodeGuesser()

        self._visit_count = np.zeros(
            (EnvParams.map_height, EnvParams.map_width), dtype=np.float32
        )  # 訪問回数を正規化して記録
        self._target_team_id = target_team_id
        self._relic_nodes = set()
        self._is_popup_relic_in_this_match = False
        self.validation = validation
        self.episode_id = episode_id

        if isinstance(env_cfg, EnvParams):
            env_cfg = flax.serialization.to_state_dict(env_cfg)

        self.unit_move_cost = env_cfg["unit_move_cost"]
        self.unit_sap_cost = env_cfg["unit_sap_cost"]
        self.unit_sap_range = env_cfg["unit_sap_range"]
        self.nebula_tile_vision_reduction_guesser = NebulaTileVisionReductionGuesser(
            target_team_id, env_cfg["unit_sensor_range"]
        )
        self.energy_attack_guesser = EnergyAttackFactorGuesser(target_team_id, self.unit_sap_cost, self.unit_move_cost)
        self.max_sensor_range = env_params_ranges["unit_sensor_range"][-1]
        self.unit_sensor_range = env_cfg["unit_sensor_range"]
        self.reset()

    def reset(self) -> None:
        # matchが切り替わったらリセットする
        self._prev_points = 0
        self._current_points = 0
        self._is_popup_relic_in_this_match = False
        self._visit_count = np.zeros((EnvParams.map_height, EnvParams.map_width), dtype=np.float32)
        self.prev_obs = None

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
    def entropy_map(self) -> np.ndarray:
        eps = 1e-13
        p = self.point_map
        q = 1 - p
        entropy = -(p * np.log2(p + eps) + q * np.log2(q + eps))
        return entropy

    @property
    def visit_count(self) -> np.ndarray:
        return self._visit_count.copy()

    @property
    def tile_type_map(self) -> np.ndarray:
        return self._tile_type_map.copy()

    @property
    def next_tile_type_map(self) -> np.ndarray:
        return self._next_tile_type_map.copy()

    @property
    def nebula_energy_reduction(self) -> np.ndarray:
        """
        nebula tileによるエネルギ減少を表す
        パラメータが未知の場合は平均で埋める
        """
        if self._nebula_energy_reduction is None:
            return (self.tile_type_map == TileType.NEBULA) * np.mean(env_params_ranges["nebula_tile_energy_reduction"])
        else:
            target_map = (
                (self.tile_type_map == TileType.NEBULA) * self._nebula_energy_reduction / EnvParams.init_unit_energy
            )
            return target_map

    @property
    def vision_power_map(self) -> np.ndarray:
        return self._vision_power_map.copy()

    def update(self, obs: dict[str, Any], actions: np.ndarray) -> None:
        self._update_vision_power_map(obs)
        self._update_tile_type_map(obs)
        self._update_nebula_energy_reduction(obs, actions)
        self._update_relic_map(obs)
        self._update_visit_count(obs)
        self._update_points(obs)
        self._update_point_map(obs)
        self.energy_node_guesser.update_energy(obs)
        self.nebula_tile_vision_reduction_guesser._update_nebula_tile_vision_reduction(obs)
        self.energy_attack_guesser.update(obs, self.prev_obs, actions, self.energy_node_guesser)
        self.prev_obs = obs

    def _is_finished_relic_search(self) -> bool:
        return self._relic_map.sum() == EnvParams.max_relic_nodes

    def _filter_tile_speed_candidates(self, new_tile_type_map: np.ndarray, steps: int) -> None:
        """
        記録済みの前のステップ時点でのマップと新しいマップを比較してマップの移動速度を特定する
        """

        def get_valid_mask(new_tile_type_map: np.ndarray, old_tile_type_map: np.ndarray) -> np.ndarray:
            """
            新しいマップと前回のマップの、両方が UNKNOWN でないセルのマスクを作成
            """
            return (new_tile_type_map != TileType.UNKNOWN) & (old_tile_type_map != TileType.UNKNOWN)

        def get_mujun_mask(
            new_tile_type_map: np.ndarray, old_tile_type_map: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            """
            2つのマップがUNKNOWNでないセルのみを比較して矛盾している箇所を抽出
            """
            # 新しいマップと前回のマップの、両方が UNKNOWN でないセルのマスクを作成
            valid_mask = get_valid_mask(new_tile_type_map, old_tile_type_map)
            # その中で、値が矛盾している箇所を抽出
            diff_mask = valid_mask & (new_tile_type_map != old_tile_type_map)
            return diff_mask, valid_mask

        def get_roll_mujun_count(new_tile_type_map: np.ndarray, old_tile_type_map: np.ndarray, sign: int) -> int:
            rolled_map = np.roll(old_tile_type_map, shift=(1 * sign, -1 * sign), axis=(0, 1))
            mujun_mask, _ = get_mujun_mask(new_tile_type_map, rolled_map)
            return mujun_mask.sum()

        mujun_mask, valid_mask = get_mujun_mask(new_tile_type_map, self._tile_type_map)
        valid_count = valid_mask.sum()
        mujun_count = mujun_mask.sum()

        # 内部マップを正方向 (1, -1) および負方向 (-1, 1) に roll した２種類のマップを作成
        pos_mujun_count = get_roll_mujun_count(new_tile_type_map, self._tile_type_map, sign=1)
        neg_mujun_count = get_roll_mujun_count(new_tile_type_map, self._tile_type_map, sign=-1)

        # 矛盾がない場合でも移動している可能性がある(移動したが移動先のセルも同じになる場合)
        # TODO: すでに除外してる候補のstepの場合この処理は不要なのでスキップしてもいいかも。逆に誤って除外してる場合に追加するてもある？ミスっててもいい感じになるようにしたい
        if mujun_count == 0:
            # 移動させた場合に矛盾が生じるなら移動していないと考える
            if pos_mujun_count > 0 and neg_mujun_count > 0:
                # 移動していないなら候補を絞れる(steps=10で移動していないなら10は除外できる)
                candidates = {
                    cand
                    for cand in self._nebula_tile_drift_speed_candidates
                    if (steps - 1) % self.speed_to_step[abs(cand)] != 0
                }
                self._nebula_tile_drift_speed_candidates = candidates
                return
            # 片方に矛盾がない場合、移動している可能性もある(観測セル数が少ない場合こういうことが起こる)
            else:
                # この場合は新しい観測で上書きするのが安全な気がする
                # self._tile_type_map = new_tile_type_map
                return
        # 矛盾が発生する場合、移動しているはずなのでステップ数がどの倍数かに基づいて候補を絞り込む
        else:
            candidates = [
                cand
                for cand in self._nebula_tile_drift_speed_candidates
                if (steps - 1) % self.speed_to_step[abs(cand)] == 0
            ]

            if len(candidates) == 0:
                if self.validation:
                    raise ValueError(
                        f"矛盾が発生しているが条件を満たす候補が無い: {self.episode_id=} {steps=} {self._nebula_tile_drift_speed_candidates=}"
                    )
                else:
                    # 矛盾が発生しているが条件を満たす候補が無い場合はスキップ(たまにある)
                    return

            if self.validation:
                # どちらかは一致しているはず
                assert pos_mujun_count == 0 or neg_mujun_count == 0
                # 両方一致はないはず(ここがミスる場合は緩和した方がいい)
                assert pos_mujun_count + neg_mujun_count > 0

            # 0の場合は正方向を選択
            sign = 1 if pos_mujun_count == 0 else -1

            # 有効な候補の中から、chosen_sign に合致する drift_speed のみを残す
            new_candidates = {cand for cand in candidates if np.sign(cand) == sign}
            # 候補リストを更新
            self._nebula_tile_drift_speed_candidates = new_candidates
            # 移動させる
            self._tile_type_map = np.roll(self._tile_type_map, shift=(1 * sign, -1 * sign), axis=(0, 1))

    # https://github.com/okumura2997/lux-ai-season-3/blob/69b37bf069c377986004ea083a6e399b88811c7a/src/agent.py#L510
    def _update_vision_power_map(self, obs: dict[str, Any]) -> None:
        """
        visionは現在のステップのunit位置と前のステップのタイル位置に基づいて決まる
        とりあえずここではunit位置に基づくvisionを計算する
        """
        vision_power_map_padding = self.max_sensor_range
        padded_h = EnvParams.map_height + 2 * vision_power_map_padding
        padded_w = EnvParams.map_width + 2 * vision_power_map_padding
        vision_power_map = np.zeros((padded_h, padded_w), dtype=np.int16)

        unit_positions = np.array(obs["units"]["position"][self._target_team_id])
        for x, y in unit_positions:
            if x == -1 and y == -1:
                continue
            padded_x = x + vision_power_map_padding
            padded_y = y + vision_power_map_padding

            start_x = padded_x - self.max_sensor_range
            start_y = padded_y - self.max_sensor_range
            slice_size = self.max_sensor_range * 2 + 1

            existing = vision_power_map[start_x : start_x + slice_size, start_y : start_y + slice_size].copy()
            update = np.zeros_like(existing, dtype=np.int16)

            for i in range(self.max_sensor_range + 1):
                if i > (self.max_sensor_range - self.unit_sensor_range - 1):
                    val = i + 1 - (self.max_sensor_range - self.unit_sensor_range)
                else:
                    val = 0
                update[i : slice_size - i, i : slice_size - i] = val

            update[self.max_sensor_range, self.max_sensor_range] = 10

            vision_power_map[start_x : start_x + slice_size, start_y : start_y + slice_size] = existing + update

        updated_vision_power_map = vision_power_map[
            vision_power_map_padding:-vision_power_map_padding, vision_power_map_padding:-vision_power_map_padding
        ]
        self._vision_power_map = updated_vision_power_map.T

    def _update_tile_type_map(self, obs: dict[str, Any]) -> None:
        """
        visionは前のtile情報をもとにreductionされる
        """
        # 新しいタイプマップ（内部規則に合わせ転置済み）
        new_tile_type_map = np.array(obs["map_features"]["tile_type"]).T
        new_tile_type_map = mirroring(new_tile_type_map, null_value=TileType.UNKNOWN)
        sensor_mask = np.array(obs["sensor_mask"]).T
        # vision>0にも関わらず観測できないセルがあれば前のステップにおけるそのはnebulaであると考える
        self._tile_type_map = np.where(
            (self._vision_power_map > 0) & (~sensor_mask),
            TileType.NEBULA,
            self._tile_type_map,
        )

        # speed候補を絞り込む(移動の可能性のあるstepでのみ行う)
        if (obs["steps"] - 1) % 7 == 0 or (obs["steps"] - 1) % 10 == 0:
            if len(self._nebula_tile_drift_speed_candidates) > 1:
                self._filter_tile_speed_candidates(new_tile_type_map, obs["steps"])
            elif len(self._nebula_tile_drift_speed_candidates) == 1:
                # すでに drift_speed が確定している場合、移動が発生するタイミングでmapをrollする
                # これだと確定していない場合に移動が発生する可能性もあるがそれが無視されている
                speed = list(self._nebula_tile_drift_speed_candidates)[0]
                # 切り替わるタイミングであればマップを更新
                if (obs["steps"] - 1) % self.speed_to_step[abs(speed)] == 0:
                    sign = int(np.sign(speed))
                    self._tile_type_map = np.roll(self._tile_type_map, shift=(1 * sign, -1 * sign), axis=(0, 1))

        # 観測値を上書き(未知の場合はそのままでそれ以外は観測値で上書き)
        self._tile_type_map = np.where(new_tile_type_map == TileType.UNKNOWN, self._tile_type_map, new_tile_type_map)
        if len(self._nebula_tile_drift_speed_candidates) == 1:
            speed = list(self._nebula_tile_drift_speed_candidates)[0]
            next_step = obs["steps"]
            # 次のステップで移動する場合はマップを更新
            if next_step % self.speed_to_step[abs(speed)] == 0:
                sign = int(np.sign(speed))
                self._next_tile_type_map = np.roll(self._tile_type_map, shift=(1 * sign, -1 * sign), axis=(0, 1))
            else:
                self._next_tile_type_map = self._tile_type_map.copy()
        else:
            # 絞れていない場合はそのまま
            self._next_tile_type_map = self._tile_type_map.copy()

        self._tile_type_map = mirroring(self._tile_type_map, null_value=TileType.UNKNOWN)
        self._next_tile_type_map = mirroring(self._next_tile_type_map, null_value=TileType.UNKNOWN)

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
        # x, y = 17, 5
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

    def _update_nebula_energy_reduction(self, obs: dict[str, Any], actions: np.ndarray) -> None:
        """
        nebulaセルにいるunitのエネルギー値を見ることでnebulaセルのエネルギー減少を推定する
        """
        # 一度確定させればあとは計算不要
        if self._nebula_energy_reduction is not None:
            return
        # 0ステップ目は計算できないのでskip
        if self.prev_obs is None:
            return

        unit_positions = np.array(obs["units"]["position"][self._target_team_id])  # (max_units, 2)
        prev_unit_positions = np.array(self.prev_obs["units"]["position"][self._target_team_id])  # (max_units, 2)
        unit_energies = np.array(obs["units"]["energy"][self._target_team_id])  # (max_units, 1)
        map_energies = np.array(obs["map_features"]["energy"]).T
        prev_unit_energies = self.prev_obs["units"]["energy"][self._target_team_id]
        prev_tile_type_map = np.array(self.prev_obs["map_features"]["tile_type"]).T
        # TODO: energy voidとかsapされることを考慮してない. ほとんど場合には相手と会う前に決まるので問題ない
        for unit_id, ((x, y), (prev_x, prev_y), unit_energy) in enumerate(
            zip(unit_positions, prev_unit_positions, unit_energies)
        ):
            if x == -1 and y == -1:
                continue
            # このターンspawnしたunitにもactionを送れるが無効なので取り除く
            if prev_x == -1 and prev_y == -1:
                continue
            if abs(x - prev_x) + abs(y - prev_y) > 1:
                continue
            if prev_tile_type_map[y, x] != TileType.NEBULA:
                continue
            # sap等の相手からの干渉なしでは0未満にならないようになっている. 負の場合は相手から干渉されている.
            # どちらのケースもnebula_energy_reductionを計算することができないのでスキップ
            if unit_energy <= 0:
                continue

            map_energy = map_energies[y, x]
            prev_unit_energy = prev_unit_energies[unit_id]

            # 前ステップからの行動によってユニットのエネルギーが減少するのでそれを考慮
            action = actions[unit_id][0].item()
            action_cost = get_action_cost(action, unit_energy, self.unit_sap_cost, self.unit_move_cost)

            # 現在のエネルギ = 前stepのエネルギ - 移動コスト + マップのエネルギ - nebulaによるエネルギ減少
            # unit_energy = prev_unit_energy - action_cost + map_energy - nebula_energy_reduction
            nebula_energy_reduction = (prev_unit_energy - action_cost + map_energy) - unit_energy
            # print(f"{obs['steps']=} {nebula_energy_reduction=} {prev_unit_energy=} {unit_energy=} {action_cost=} {map_energy=} {unit_id=}")
            if nebula_energy_reduction in env_params_ranges["nebula_tile_energy_reduction"]:
                self._nebula_energy_reduction = nebula_energy_reduction
                return


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
                pass
                # state_map[HiddenState.OWN_UNIT_COUNT, y, x] += 1 / EnvParams.max_units
            else:
                state_map[HiddenState.OPP_UNIT_COUNT, y, x] += 1 / EnvParams.max_units

    state_map[HiddenState.POINTS] = get_gt_point_map(gt_obs)

    # state_map[HiddenState.ENERGY] = np.array(gt_obs["map_features"]["energy"]).T / 10  # (24, 24)
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
            else:
                state_map[State.OPP_UNIT_COUNT, y, x] += 1
                state_map[State.OPP_UNIT_ENERGY, y, x] += unit_energy / EnvParams.init_unit_energy

    return state_map


def extract_state(
    obs: dict[str, Any],
    target_team_id: int,
    episode_store: EpisodeStore,
    detemined_actions: dict[int, Any],
    acting_unig_id: int | None,
) -> np.ndarray:
    state_space_size: int = len(State)
    # enemy_team_id = 1 - target_team_id
    state_map = np.zeros((state_space_size, EnvParams.map_width, EnvParams.map_height), dtype=np.float32)

    # state
    # map state
    # state_map[State.TILE_TYPE] = np.array(obs["map_features"]["tile_type"]).T
    state_map[State.TILE_TYPE] = episode_store.tile_type_map
    state_map[State.NEXT_TILE_TYPE] = episode_store.next_tile_type_map
    # energy nodesの位置は未知(tileのenergyはvisionで観測可能) energy系は正規化の分母をinit_unit_energyにする
    state_map[State.ENERGY] = episode_store.energy_node_guesser.get_energy_map() / EnvParams.init_unit_energy
    state_map[State.NEBULA_ENERGY_REDUCTION] = episode_store.nebula_energy_reduction
    state_map[State.SENSOR_MASK] = np.array(obs["sensor_mask"]).T
    # state_map[State.VISION_POWER_MAP] = episode_store.vision_power_map

    state_map[State.RELICS] = episode_store.relic_map
    state_map[State.POINTS] = episode_store.point_map
    state_map[State.ENTROPY] = episode_store.entropy_map
    state_map[State.VISIT_COUNT] = episode_store.visit_count

    acted_unit = set(detemined_actions.keys())
    if acting_unig_id is not None:
        acting_x, acting_y = obs["units"]["position"][target_team_id][acting_unig_id]
        state_map[State.OWN_ACTING_UNIT_POSITION, acting_y, acting_x] += 1
    for unit_id, action in detemined_actions.items():
        x, y = obs["units"]["position"][target_team_id][unit_id]
        unit_energy = obs["units"]["energy"][target_team_id][unit_id]
        if (
            action[0] != Action.SAP
            and action[0] != Action.CENTER
            and can_move((x, y), unit_energy, action[0], episode_store.tile_type_map, episode_store.unit_move_cost)
        ):
            nx, ny = calc_next_pos((x, y), action[0])
            state_map[State.OWN_ACTED_UNIT_NEXT_POSITION, ny, nx] += 1
        else:
            state_map[State.OWN_ACTED_UNIT_NEXT_POSITION, y, x] += 1
        if action[0] == Action.SAP and can_sap(unit_energy, episode_store.unit_sap_cost):
            sap_x = x + action[1]
            sap_y = y + action[2]
            if in_map((sap_x, sap_y)):
                state_map[State.SAP_SCHEDULED_POSITION, sap_y, sap_x] += 1

    # unit state
    for team_id in range(2):
        # 敵チームの情報はvision内にいない限り見れない
        unit_energies = np.array(obs["units"]["energy"][team_id])  # (max_units, 1)
        unit_positions = np.array(obs["units"]["position"][team_id])  # (max_units, 2)
        unit_masks = np.array(obs["units_mask"][team_id])  # (max_units, )
        if team_id != target_team_id:
            # sensor_maskが1(見える範囲)の場合は0にする。それ以外は0.5
            state_map[State.OPP_UNIT_COUNT] = -1
            state_map[State.OPP_UNIT_COUNT] *= 1 - state_map[State.SENSOR_MASK]
            # アステロイドのところは存在しないので0
            state_map[State.OPP_UNIT_COUNT] *= 1 - (state_map[State.TILE_TYPE] == TileType.ASTEROID)

            state_map[State.OPP_UNIT_ENERGY] = -1
            state_map[State.OPP_UNIT_ENERGY] *= 1 - state_map[State.SENSOR_MASK]
            # アステロイドのところは存在しないので0
            state_map[State.OPP_UNIT_ENERGY] *= 1 - (state_map[State.TILE_TYPE] == TileType.ASTEROID)

        # available_unit_ids = np.where(unit_masks)[0]
        for unit_id in range(EnvParams.max_units):
            unit_energy = unit_energies[unit_id]
            x, y = unit_positions[unit_id]
            unit_mask = unit_masks[unit_id] * 1
            if x == -1 and y == -1:
                continue
            # 味方同士は重複可能なのでincrementする（敵との重複はないため打ち消し合うことはないはず）
            if team_id == target_team_id:
                if unit_id not in acted_unit and unit_id != acting_unig_id:
                    state_map[State.OWN_NOT_ACTED_UNIT_POSITION, y, x] += 1

                # 重複はそんなに発生しないだろうということで正規化はしない
                state_map[State.OWN_UNIT_COUNT, y, x] += 1 / EnvParams.max_units
                state_map[State.OWN_UNIT_ENERGY, y, x] += unit_energy / EnvParams.init_unit_energy
                # sapを使用できるエリアを1にする
                for dx in range(-EnvParams.unit_sap_range, EnvParams.unit_sap_range + 1):
                    for dy in range(-EnvParams.unit_sap_range, EnvParams.unit_sap_range + 1):
                        nx, ny = x + dx, y + dy
                        if in_map((nx, ny)) and can_sap(unit_energy, episode_store.unit_sap_cost):
                            state_map[State.SAP_AVAILABLE_AREA, ny, nx] = 1
                # state_map[State.OWN_UNIT_MASK, y, x] = unit_mask
            else:
                state_map[State.OPP_UNIT_COUNT, y, x] += 1 / EnvParams.max_units
                state_map[State.OPP_UNIT_ENERGY, y, x] += unit_energy / EnvParams.init_unit_energy
                # state_map[State.OPP_UNIT_MASK, y, x] = unit_mask
    return state_map


def extract_global_state(
    obs: dict[str, Any], target_team_id: int, env_params: EnvParams, episode_store: EpisodeStore
) -> np.ndarray:
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
    (vision_reduction_mean, vision_reduction_std) = (
        episode_store.nebula_tile_vision_reduction_guesser.get_nebula_tile_vision_reduction_estimate()
    )
    global_states[GlobalState.NEBULA_TILE_VISION_REDUCTION_MEAN] = vision_reduction_mean / max(
        env_params_ranges["nebula_tile_vision_reduction"]
    )
    global_states[GlobalState.NEBULA_TILE_VISION_REDUCTION_SIGMA] = vision_reduction_std / max(
        env_params_ranges["nebula_tile_vision_reduction"]
    )
    (energy_drift_speed_mean, energy_drift_speed_std) = (
        episode_store.energy_node_guesser.get_energy_drft_speed_estimate()
    )
    global_states[GlobalState.ENERGY_NODE_DRIFT_SPEED_MEAN] = energy_drift_speed_mean / max(
        env_params_ranges["energy_node_drift_speed"]
    )
    global_states[GlobalState.ENERGY_NODE_DRIFT_SPEED_SIGMA] = energy_drift_speed_std / max(
        env_params_ranges["energy_node_drift_speed"]
    )
    global_states[GlobalState.NEBULA_TILE_ENERGY_REDUCTION_MEAN] = (
        episode_store._nebula_energy_reduction
        if episode_store._nebula_energy_reduction is not None
        else np.mean(env_params_ranges["nebula_tile_energy_reduction"])
    ) / max(env_params_ranges["nebula_tile_energy_reduction"])
    global_states[GlobalState.NEBULA_TILE_ENERGY_REDUCTION_SIGMA] = (
        0
        if episode_store._nebula_energy_reduction is not None
        else np.std(env_params_ranges["nebula_tile_energy_reduction"])
    ) / max(env_params_ranges["nebula_tile_energy_reduction"])
    (unit_sap_dropoff_factor_mean, unit_sap_dropoff_factor_std) = (
        episode_store.energy_attack_guesser.get_sap_dropoff_factor_estimate()
    )
    global_states[GlobalState.UNIT_SAP_DROPOFF_FACTOR_MEAN] = unit_sap_dropoff_factor_mean / max(
        env_params_ranges["unit_sap_dropoff_factor"]
    )
    global_states[GlobalState.UNIT_SAP_DROPOFF_FACTOR_SIGMA] = unit_sap_dropoff_factor_std / max(
        env_params_ranges["unit_sap_dropoff_factor"]
    )
    return global_states


def extract_hidden_global_state(env_params: dict[str, Any]) -> np.ndarray:
    hidden_global_states = np.zeros((len(HiddenGlobalState),), dtype=np.float32)
    # hidden_global_states[HiddenGlobalState.NEBULA_TILE_VISION_REDUCTION] = env_params.nebula_tile_vision_reduction
    # hidden_global_states[HiddenGlobalState.NEBULA_TILE_ENERGY_REDUCTION] = env_params.nebula_tile_energy_reduction
    # hidden_global_states[HiddenGlobalState.UNIT_SAP_DROPOFF_FACTOR] = env_params.unit_sap_dropoff_factor
    # hidden_global_states[HiddenGlobalState.UNIT_ENERGY_VOID_FACTOR] = env_params.unit_energy_void_factor
    # hidden_global_states[HiddenGlobalState.NEBULA_TILE_DRIFT_SPEED] = env_params.nebula_tile_drift_speed
    # hidden_global_states[HiddenGlobalState.ENERGY_NODE_DRIFT_SPEED] = env_params.energy_node_drift_speed
    hidden_global_states[HiddenGlobalState.ENERGY_NODE_DRIFT_MAGNITUDE] = env_params.energy_node_drift_magnitude

    return hidden_global_states


def extract_action(actions: np.ndarray, obs: dict[str, Any], target_team_id: int) -> np.ndarray:
    action_map = np.zeros((2, EnvParams.map_width, EnvParams.map_height), dtype=np.float32)
    # unit state
    unit_masks = np.array(obs["units_mask"][target_team_id])  # (max_units, )
    unit_positions = np.array(obs["units"]["position"][target_team_id])  # (max_units, 2)

    available_unit_ids = np.where(unit_masks)[0]
    for unit_id in available_unit_ids:
        x, y = unit_positions[unit_id]
        action_map[0, y, x] = actions[unit_id][0]
        # sapしている位置を1にする
        if actions[unit_id][0] == Action.SAP:
            dx, dy = actions[unit_id][1:]
            nx = x + dx
            ny = y + dy
            if in_map((nx, ny)):
                action_map[1, ny, nx] = 1
    return action_map


def extract_action_per_unit(
    unit_id: int, action: tuple[Action, int, int], obs: dict[str, Any], target_team_id: int
) -> np.ndarray:
    # action_map = np.zeros(len(Action)-1 + EnvParams.map_height*EnvParams.map_width, dtype=np.float32)
    x, y = obs["units"]["position"][target_team_id][unit_id]
    if action[0] == Action.SAP:
        dx, dy = action[1:]
        nx = x + dx
        ny = y + dy
        if in_map((nx, ny)) and obs["units"]["energy"][target_team_id][unit_id] >= EnvParams.unit_sap_cost:
            return Action.SAP + ny * EnvParams.map_width + nx
        else:
            return Action.CENTER
    else:
        return action[0]


# def extract_action_per_unit(unit_id : int, action: tuple[Action, int, int], obs: dict[str, Any], target_team_id: int) -> np.ndarray:
#     # action_map = np.zeros(len(Action)-1 + EnvParams.map_height*EnvParams.map_width, dtype=np.float32)
#     action_map = np.zeros((2, EnvParams.map_width, EnvParams.map_height), dtype=np.float32)
#     x, y = obs["units"]["position"][target_team_id][unit_id]
#     if action[0] == Action.SAP:
#         dx, dy = action[1:]
#         nx = x + dx
#         ny = y + dy
#         if in_map((nx, ny)) and obs["units"]["energy"][target_team_id][unit_id] >= EnvParams.unit_sap_cost:
#             action_map[1, ny, nx] = 1
#         else:
#             action_map[0, y, x] = 1
#     else:
#         nx, ny = calc_next_pos((x, y), action[0])
#         action_map[0, ny, nx] = 1
#     return action_map


def get_valid_policy_map(obs: dict[str, Any], team_id: int, episode_store: EpisodeStore, unit_id: int) -> np.ndarray:
    validate_policy_map = np.ones(len(Action) - 1 + EnvParams.map_width * EnvParams.map_height, dtype=np.float32)
    pos = tuple(obs["units"]["position"][team_id][unit_id])
    x, y = pos
    energy = obs["units"]["energy"][team_id][unit_id]

    for dir in [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]:
        if not can_move(pos, energy, dir, episode_store.tile_type_map, episode_store.unit_move_cost):
            validate_policy_map[dir] = 0

    if not can_sap(energy, episode_store.unit_sap_cost):
        for y in range(EnvParams.map_height):
            for x in range(EnvParams.map_width):
                validate_policy_map[Action.SAP + y * EnvParams.map_width + x] = 0
    return validate_policy_map


# def get_valid_policy_map(obs: dict[str, Any], team_id: int, episode_store: EpisodeStore) -> np.ndarray:
#     validate_policy_map = np.zeros((len(Action), EnvParams.map_width, EnvParams.map_height), dtype=np.float32)
#     available_unit_ids = np.where(obs["units_mask"][team_id])[0]
#     for unit_id in available_unit_ids:
#         pos = tuple(obs["units"]["position"][team_id][unit_id])
#         x, y = pos
#         energy = obs["units"]["energy"][team_id][unit_id]

#         validate_policy_map[:, y, x] = 1  # 行動は一旦全て有効化

#         for dir in [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]:
#             if not can_move(pos, energy, dir, episode_store.tile_type_map, episode_store.unit_move_cost):
#                 validate_policy_map[dir, y, x] = 0

#         if not can_sap(energy, episode_store.unit_sap_cost):
#             validate_policy_map[Action.SAP, y, x] = 0
#     return validate_policy_map


def get_valid_sap_map(obs: dict[str, Any], team_id: int, episode_store: EpisodeStore) -> np.ndarray:
    validate_sap_map = np.zeros((EnvParams.map_width, EnvParams.map_height), dtype=np.float32)
    # tileがASTEROID_TILEの場合はsapできない
    validate_sap_map[episode_store.tile_type_map == TileType.ASTEROID] = 0
    # 味方unitにはsapしてもいいのか？
    return validate_sap_map


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
