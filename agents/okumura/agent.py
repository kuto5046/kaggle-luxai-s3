from sys import stderr

import numpy as np
from base import Global, NodeType, ActionType, get_match_step, is_on_diagonal, is_team_sector, get_match_number
from debug import show_map, show_energy_field, show_exploration_map
from fleet import Fleet
from space import Node, Space
from pathfinding import (
    astar,
    create_weights,
    path_to_actions,
    nearby_positions,
    manhattan_distance,
    find_closest_target,
    estimate_energy_cost,
)


class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg

        Global.MAX_UNITS = env_cfg["max_units"]
        Global.UNIT_MOVE_COST = env_cfg["unit_move_cost"]
        Global.UNIT_SAP_COST = env_cfg["unit_sap_cost"]
        Global.UNIT_SAP_RANGE = env_cfg["unit_sap_range"]
        Global.UNIT_SENSOR_RANGE = env_cfg["unit_sensor_range"]

        self.space = Space()
        self.fleet = Fleet(self.team_id)
        self.opp_fleet = Fleet(self.opp_team_id)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        match_step = get_match_step(step)
        match_number = get_match_number(step)

        if Global.DEBUG:
            print(f"start step={match_step}({step}, {match_number})", file=stderr)

        if match_step == 0:
            # nothing to do here at the beginning of the match
            # just need to clean up some of the garbage that was left after the previous match
            self.fleet.clear()
            self.opp_fleet.clear()
            self.space.clear()
            self.space.move_obstacles(step)

            if match_number < Global.LAST_MATCH_WHEN_RELIC_CAN_APPEAR:
                self.space.clear_exploration_info()
            elif (
                match_number == Global.LAST_MATCH_WHEN_RELIC_CAN_APPEAR
                and (self.space.relic_nodes) == 2
                and all([n.explored_for_relic for n in self.space])
            ):
                pass
            return self.create_actions_array()

        points = int(obs["team_points"][self.team_id])

        # how many points did we score in the last step
        reward = max(0, points - self.fleet.points)

        self.space.update(step, obs, self.team_id, reward)
        self.fleet.update(obs, self.space)
        self.opp_fleet.update(obs, self.space)
        self.find_nebula_node(obs)

        if Global.DEBUG:
            self.show_explored_map()
            self.show_exploration_map()

        self.harvest()
        self.find_relics()
        self.find_rewards()
        self.get_out_of_reward_range()

        self.move_to_enemy_reward_nodes()
        self.energy_charge()
        self.stop()
        self.sap()

        if Global.DEBUG:
            for ship in self.fleet:
                print(ship, ship.target, ship.action, file=stderr)

        return self.create_actions_array()

    def find_relics(self):
        if Global.ALL_RELICS_FOUND:
            for ship in self.fleet:
                if ship.task == "find_relics":
                    ship.task = None
                    ship.target = None
            return

        targets = set()
        for node in self.space:
            if not node.explored_for_relic:
                if is_team_sector(self.fleet.team_id, *node.coordinates):
                    targets.add(node.coordinates)

        def set_task(ship):
            if ship.task and ship.task != "find_relics":
                return False

            if ship.energy < Global.UNIT_MOVE_COST:
                return False

            for i in range(5):
                target, _ = find_closest_target(ship.coordinates, targets)
                if not target:
                    return False

                path = astar(create_weights(self.space), ship.coordinates, target)
                energy = estimate_energy_cost(self.space, path)
                actions = path_to_actions(path)
                if actions and ship.energy >= energy:
                    ship.task = "find_relics"
                    ship.target = self.space.get_node(*target)
                    ship.action = actions[0]

                    for x, y in path:
                        for xy in nearby_positions(x, y, Global.UNIT_SENSOR_RANGE):
                            if xy in targets:
                                targets.remove(xy)
                    return True
                else:
                    targets.remove(target)

            return False

        for ship in self.fleet:
            if set_task(ship):
                continue

            if ship.task == "find_relics":
                ship.task = None
                ship.target = None

    def find_rewards(self):
        if Global.ALL_REWARDS_FOUND:
            for ship in self.fleet:
                if ship.task == "find_rewards":
                    ship.task = None
                    ship.target = None
            return

        unexplored_relics = self.get_unexplored_relics()
        if not unexplored_relics:
            for ship in self.fleet:
                if ship.task == "find_rewards":
                    ship.task = None
                    ship.target = None
            return

        relic_node_to_ship = {}
        if unexplored_relics:
            relic = unexplored_relics[0]
            # find the closest ship to the relic node
            min_distance, closest_ship = float("inf"), None
            for ship in self.fleet:
                if ship.task and ship.task != "find_rewards":
                    continue

                if ship.energy < Global.UNIT_MOVE_COST * 5:
                    continue

                distance = manhattan_distance(ship.coordinates, relic.coordinates)
                if distance < min_distance:
                    min_distance, closest_ship = distance, ship

            if closest_ship:
                relic_node_to_ship[relic] = closest_ship

        def set_task(ship, relic_node):
            targets = []
            for x, y in nearby_positions(*relic_node.coordinates, Global.RELIC_REWARD_RANGE):
                node = self.space.get_node(x, y)
                if not node.explored_for_reward and node.is_walkable:
                    targets.append((x, y))

            remaining_targets = targets.copy()
            while remaining_targets:
                target, _ = find_closest_target(ship.coordinates, remaining_targets)

                if not target:
                    return False

                path = astar(create_weights(self.space), ship.coordinates, target)
                energy = estimate_energy_cost(self.space, path)
                actions = path_to_actions(path)

                if actions and ship.energy >= energy:
                    ship.task = "find_rewards"
                    ship.target = self.space.get_node(*target)
                    ship.action = actions[0]
                    return True
                else:
                    remaining_targets.remove(target)

            return False

        for n, s in sorted(list(relic_node_to_ship.items()), key=lambda _: _[1].unit_id):
            if set_task(s, n):
                other_ships = [ship for ship in self.fleet if ship != s and ship.task == "find_rewards"]
                for ship in other_ships:
                    ship.task = None
                    ship.target = None

            elif s.task == "find_rewards":
                s.task = None
                s.target = None

    def get_out_of_reward_range(self):
        if Global.ALL_REWARDS_FOUND:
            for ship in self.fleet:
                if ship.task == "get_out_of_reward_range":
                    ship.task = None
                    ship.target = None
            return
        finding_rewards = any([ship for ship in self.fleet if ship.task == "find_rewards"])
        if not finding_rewards:
            return

        reward_explored_nodes = [node for node in self.space if node.explored_for_reward and node.is_walkable]
        for ship in self.fleet:
            if ship.task == "get_out_of_reward_range" and ship.node.explored_for_reward:
                ship.task = None
                ship.target = None
                continue
            if ship.task == "find_rewards" or ship.task == "harvest":
                continue
            elif not ship.node.explored_for_reward:
                targets = set()
                for n in reward_explored_nodes:
                    targets.add(n.coordinates)
                target, _ = find_closest_target(ship.coordinates, targets)
                if target:
                    path = astar(create_weights(self.space), ship.coordinates, target)
                    energy = estimate_energy_cost(self.space, path)
                    actions = path_to_actions(path)
                    if actions and ship.energy >= energy:
                        ship.task = "get_out_of_reward_range"
                        ship.target = self.space.get_node(*target)
                        ship.action = actions[0]
                    else:
                        ship.task = None
                        ship.target = None
            else:
                pass

    def harvest(self):
        if len(self.space.reward_nodes) == 0:
            return

        def set_task(ship, target_node):
            if (
                ship.node == target_node
                and ship.energy + ship.node.energy - (Global.NEBULA_ENERGY_REDUCTION if ship.node.is_nebula else 0) > 0
            ):
                ship.task = "harvest"
                ship.target = target_node
                ship.action = ActionType.center
                return True

            path = astar(
                create_weights(self.space),
                start=ship.coordinates,
                goal=target_node.coordinates,
            )
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)

            if not actions or ship.energy < energy:
                return False
            else:
                ship.task = "harvest"
                ship.target = target_node
                ship.action = actions[0]
                return True

        booked_nodes = set()
        for ship in self.fleet:
            if ship.task == "harvest":
                if ship.target is None:
                    ship.task = None
                    continue

                if set_task(ship, ship.target):
                    booked_nodes.add(ship.target)
                else:
                    ship.task = None
                    ship.target = None

        targets = set()
        for n in self.space.reward_nodes:
            if n.is_walkable and n not in booked_nodes:
                targets.add(n.coordinates)
        if not targets:
            return

        for ship in self.fleet:
            if not targets:
                break
            if ship.task == "harvest" or ship.task == "find_rewards":
                continue

            target, _ = find_closest_target(ship.coordinates, targets)

            if target and set_task(ship, self.space.get_node(*target)):
                targets.remove(target)
            else:
                ship.task = None
                ship.target = None

    def energy_charge(self):
        def set_task(ship, target_node):
            if ship.energy > 300:
                return False

            path = astar(
                create_weights(self.space),
                start=ship.coordinates,
                goal=target_node.coordinates,
            )
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)

            if len(actions) == 0:
                ship.task = "energy_charge"
                ship.target = target_node
                ship.action = ActionType.center
                return True

            if not actions or ship.energy < energy:
                return False
            else:
                ship.task = "energy_charge"
                ship.target = target_node
                ship.action = actions[0]
                return True

        booked_nodes = set()
        for ship in self.fleet:
            if ship.task == "energy_charge":
                if ship.target.energy is None:
                    ship.task = None
                    ship.target = None
                    continue
                elif ship.target.energy - (Global.NEBULA_ENERGY_REDUCTION if ship.target.is_nebula else 0) < 5:
                    ship.task = None
                    ship.target = None
                    continue

                if set_task(ship, ship.target):
                    booked_nodes.add(ship.target)
                else:
                    ship.task = None
                    ship.target = None

        targets = []
        for n in self.space:
            condition = (
                n.energy is not None
                and n.is_walkable
                and n not in booked_nodes
                and is_team_sector(self.team_id, *n.coordinates)
                and n.energy - (Global.NEBULA_ENERGY_REDUCTION if n.is_nebula else 0) > 5
            )
            if condition:
                targets.append(n)

        if not targets:
            return

        targets.sort(key=lambda node: node.energy, reverse=True)

        for ship in self.fleet:
            if not targets:
                break
            if ship.task:
                continue

            assigned = False
            for i, node in enumerate(targets):
                if set_task(ship, node):
                    targets.pop(i)
                    assigned = True
                    break

            if not assigned:
                ship.task = None
                ship.target = None

    def move_to_enemy_reward_nodes(self):
        if len(self.space.reward_nodes) == 0:
            return

        def set_task(ship, target_node):
            if (
                ship.node == target_node
                and ship.energy + ship.node.energy - (Global.NEBULA_ENERGY_REDUCTION if ship.node.is_nebula else 0) > 0
            ):
                ship.task = "move_to_enemy_reward_nodes"
                ship.target = target_node
                ship.action = ActionType.center
                return True

            path = astar(
                create_weights(self.space),
                start=ship.coordinates,
                goal=target_node.coordinates,
            )
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)

            # energy_thr = Global.UNIT_MOVE_COST * 5 + Global.UNIT_SAP_COST
            energy_thr = 0
            if not actions or (ship.energy - energy) < energy_thr:
                return False
            else:
                ship.task = "move_to_enemy_reward_nodes"
                ship.target = target_node
                ship.action = actions[0]
                return True

        booked_nodes = set()
        for ship in self.fleet:
            if ship.task == "move_to_enemy_reward_nodes":
                if ship.target is None:
                    ship.task = None
                    continue

                if set_task(ship, ship.target):
                    booked_nodes.add(ship.target)
                else:
                    ship.task = None
                    ship.target = None

        targets = set()
        for n in self.space.reward_nodes:
            if (
                n not in booked_nodes
                and is_team_sector(self.opp_team_id, *n.coordinates)
                and not is_on_diagonal(*n.coordinates)
            ):
                targets.add(n.coordinates)
        if not targets:
            return

        for ship in self.fleet:
            if not targets:
                break
            if ship.task and ship.task != "move_to_enemy_reward_nodes":
                continue

            target, _ = find_closest_target(ship.coordinates, targets)

            if target and set_task(ship, self.space.get_node(*target)):
                pass
            else:
                ship.task = None
                ship.target = None

    def sap(self):
        enemy_ships = [ship for ship in self.opp_fleet if ship.energy > 0 and ship.node.reward]
        if len(enemy_ships) == 0:
            return

        for ship in self.fleet:
            if ship.task in ["find_relics", "find_rewards", "get_out_of_reward_range"]:
                continue
            if ship.energy < Global.UNIT_SAP_COST:
                continue

            for enemy_ship in enemy_ships:
                if enemy_ship.energy < 0:
                    continue
                dx = enemy_ship.coordinates[0] - ship.coordinates[0]
                dy = enemy_ship.coordinates[1] - ship.coordinates[1]

                is_in_sap_range = max(abs(dx), abs(dy)) <= Global.UNIT_SAP_RANGE
                if is_in_sap_range and ship.energy < enemy_ship.energy:
                    ship.action = ActionType.sap
                    ship.sap_dx = dx
                    ship.sap_dy = dy
                    enemy_ship.energy -= Global.UNIT_SAP_COST
                    break
                else:
                    continue

    def stop(self):
        for ship in self.fleet:
            if ship.energy > 300:
                continue
            if ship.task == "harvest" or ship.task == "get_out_of_reward_range":
                continue
            energy_gain = ship.node.energy if ship.node.energy is not None else Global.HIDDEN_NODE_ENERGY
            energy_gain -= Global.NEBULA_ENERGY_REDUCTION if ship.node.is_nebula else 0
            if np.random.rand() < energy_gain / 10 and energy_gain > 6:
                ship.action = ActionType.center

    def get_unexplored_relics(self) -> list[Node]:
        unexplored_relic_nodes = []
        for relic_node in self.space.relic_nodes:
            if not is_team_sector(self.team_id, *relic_node.coordinates):
                continue

            explored = True
            for x, y in nearby_positions(*relic_node.coordinates, Global.RELIC_REWARD_RANGE):
                node = self.space.get_node(x, y)
                if not node.explored_for_reward and node.is_walkable:
                    explored = False
                    break

            if explored:
                continue
            else:
                unexplored_relic_nodes.append(relic_node)

        return unexplored_relic_nodes

    def _compute_vision_power_map(self):
        max_sensor_range = Global.MAX_SENSOR_RANGE
        unit_sensor_range = Global.UNIT_SENSOR_RANGE
        vision_power_map_padding = max_sensor_range
        map_height = Global.SPACE_SIZE
        map_width = Global.SPACE_SIZE
        padded_h = map_height + 2 * vision_power_map_padding
        padded_w = map_width + 2 * vision_power_map_padding
        vision_power_map = np.zeros((padded_h, padded_w), dtype=np.int16)

        for ship in self.fleet:
            x, y = ship.coordinates
            padded_x = x + vision_power_map_padding
            padded_y = y + vision_power_map_padding

            start_x = padded_x - max_sensor_range
            start_y = padded_y - max_sensor_range
            slice_size = max_sensor_range * 2 + 1

            existing = vision_power_map[start_x : start_x + slice_size, start_y : start_y + slice_size].copy()
            update = np.zeros_like(existing, dtype=np.int16)

            for i in range(max_sensor_range + 1):
                if i > (max_sensor_range - unit_sensor_range - 1):
                    val = i + 1 - (max_sensor_range - unit_sensor_range)
                else:
                    val = 0
                update[i : slice_size - i, i : slice_size - i] = val

            update[max_sensor_range, max_sensor_range] = 10

            vision_power_map[start_x : start_x + slice_size, start_y : start_y + slice_size] = existing + update

        return vision_power_map[
            vision_power_map_padding:-vision_power_map_padding, vision_power_map_padding:-vision_power_map_padding
        ]

    def find_nebula_node(self, obs):
        vision_power_map = self._compute_vision_power_map()
        sensor_mask = obs["sensor_mask"]
        for node in self.space:
            x, y = node.coordinates
            is_visible = sensor_mask[x, y]
            if vision_power_map[x, y] > 0 and not is_visible:
                node.type = NodeType.nebula

    def create_actions_array(self):
        ships = self.fleet.ships
        actions = np.zeros((len(ships), 3), dtype=int)
        for i, ship in enumerate(ships):
            if ship.action is not None:
                if ship.action == ActionType.sap:
                    actions[i] = ship.action, ship.sap_dx, ship.sap_dy
                else:
                    actions[i] = ship.action, 0, 0

        return actions

    def show_visible_energy_field(self):
        print("Visible energy field:", file=stderr)
        show_energy_field(self.space)

    def show_explored_energy_field(self):
        print("Explored energy field:", file=stderr)
        show_energy_field(self.space, only_visible=False)

    def show_visible_map(self):
        print("Visible map:", file=stderr)
        show_map(self.space, self.fleet, self.opp_fleet)

    def show_explored_map(self):
        print("Explored map:", file=stderr)
        show_map(self.space, self.fleet, only_visible=False)

    def show_exploration_map(self):
        print("Exploration map:", file=stderr)
        show_exploration_map(self.space)
