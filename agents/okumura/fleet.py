from base import Global, ActionType
from space import Node, Space


class Ship:
    def __init__(self, unit_id: int):
        self.unit_id = unit_id
        self.prev_energy = 0
        self.energy = 0
        self.node: Node | None = None

        self.task: str | None = None
        self.target: Node | None = None
        self.action: ActionType | None = None
        self.sap_dx, self.sap_dy = 0, 0

    def __repr__(self):
        return f"Ship({self.unit_id}, node={self.node.coordinates}, energy={self.energy}), task={self.task}"

    @property
    def coordinates(self):
        return self.node.coordinates if self.node else None

    def clean(self):
        self.energy = 0
        self.node = None
        self.task = None
        self.target = None
        self.action = None


class Fleet:
    def __init__(self, team_id):
        self.team_id: int = team_id
        self.points: int = 0  # how many points have we scored in this match so far
        self.ships = [Ship(unit_id) for unit_id in range(Global.MAX_UNITS)]

    def __repr__(self):
        return f"Fleet({self.team_id})"

    def __iter__(self):
        for ship in self.ships:
            if ship.node is not None:
                yield ship

    def clear(self):
        self.points = 0
        for ship in self.ships:
            ship.clean()

    def update(self, obs, space: Space):
        self.points = int(obs["team_points"][self.team_id])

        for ship, active, position, energy in zip(
            self.ships,
            obs["units_mask"][self.team_id],
            obs["units"]["position"][self.team_id],
            obs["units"]["energy"][self.team_id],
        ):
            if active:
                if energy == 0:
                    ship.task = None
                    ship.energy = 0
                    ship.target = None
                    ship.action = None
                    continue

                ship.node = space.get_node(*position)

                # find nebula energy reduction
                if Global.NEBULA_ENERGY_REDUCTION_FOUND:
                    pass
                elif ship.node.is_nebula:
                    energy_diff = energy - ship.energy

                    if ship.action == ActionType.sap:
                        action_cost = Global.UNIT_SAP_COST
                    elif ship.action == ActionType.center:
                        action_cost = 0
                    else:
                        action_cost = Global.UNIT_MOVE_COST

                    nebula_energy_reduction = -(energy_diff + action_cost - ship.node.energy)
                    if nebula_energy_reduction in [0, 1, 2, 3, 5, 25]:
                        Global.NEBULA_ENERGY_REDUCTION = nebula_energy_reduction
                        Global.NEBULA_ENERGY_REDUCTION_FOUND = True

                ship.energy = int(energy)
                ship.action = None
            else:
                ship.clean()
