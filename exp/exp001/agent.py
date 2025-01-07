from typing import Any
from pathlib import Path

import numpy as np
import torch
from lux.utils import Action, extract_state, get_valid_policy_map
from lux.models import LuxUNetModel
from lux.params import EnvParams
from scipy.special import softmax


class Config:
    seed: int = 2025
    in_channels: int = 13
    out_channels: int = 6
    checkpoint_path: Path = Path(__file__).parent / "output/best_model.ckpt"


class ILAgent:
    def __init__(self, env_cfg: EnvParams, checkpoint_path: Path, in_channels: int, out_channels: int) -> None:
        self.torch_model = LuxUNetModel(in_channels=in_channels, out_channels=out_channels)
        ckpt = torch.load(checkpoint_path, weights_only=True)
        self.torch_model.load_state_dict(ckpt["state_dict"], strict=False)
        self.torch_model.eval()
        self.player = None
        self.env_cfg = env_cfg

    def predict(self, obs: dict[str, Any], team_id: int):
        state = extract_state(obs, team_id)
        with torch.no_grad():
            output = self.torch_model(state)
            policy_map = output["policy"].squeeze().numpy()

        legal_action_map = get_valid_policy_map(state, team_id)
        action_mask_map = np.ones_like(policy_map) * 1e32
        action_mask_map[legal_action_map > 0] = 0
        policy_map = softmax(policy_map - action_mask_map, axis=0) * (action_mask_map == 0) * 1
        return policy_map


cfg = Config()
imitation_model = ILAgent(EnvParams, cfg.checkpoint_path, cfg.in_channels, cfg.out_channels)


class Agent:
    def __init__(self, player: str, env_cfg: EnvParams) -> None:
        self.cfg = Config()
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(self.cfg.seed)
        self.env_cfg = env_cfg

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        policy_map = imitation_model.predict(obs, self.team_id)

        unit_mask = np.array(obs["units_mask"][self.team_id])  # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id])  # shape (max_units, 2)

        # ids of units you can control at this timestep
        available_unit_ids = np.where(unit_mask)[0]
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        # unit ids range from 0 to max_units - 1
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            action = policy_map[unit_pos[0], unit_pos[1]].argmax()
            # sapの場合は何もしない
            if action == Action.SAP:
                actions[unit_id] = [0, 0, 0]
            else:
                actions[unit_id] = [action, 0, 0]
        return actions
