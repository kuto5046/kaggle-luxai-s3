from typing import Any
from pathlib import Path
from collections import deque

import numpy as np
import torch
from lightning import seed_everything
from lux.utils import State, Action, HiddenState, EpisodeStore, extract_state, get_valid_policy_map
from lux.models import LuxUNetModel
from lux.params import EnvParams
from scipy.special import softmax


class Config:
    seed: int = 2025
    # 確率的な行動を取るかどうか
    stochastic: bool = False  # Falseにするとargmaxで行動を選択する
    n_stack: int = 4

    checkpoint_path: Path = Path(__file__).parent / "output/best_model.ckpt"


class ILAgent:
    def __init__(self, env_cfg: EnvParams, checkpoint_path: Path, n_stack: int) -> None:
        self.model = LuxUNetModel(
            state_space_size=len(State), action_space_size=len(Action), hidden_state_space_size=len(HiddenState)
        )
        ckpt = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.player = None
        self.env_cfg = env_cfg
        # n_stack分のstateを保持するqueue
        self.stack_states = deque(maxlen=n_stack)
        for i in range(n_stack):
            self.stack_states.append(np.zeros((len(State), 24, 24)))

    def predict(self, obs: dict[str, Any], team_id: int, episode_store: EpisodeStore):
        state = extract_state(obs, team_id, episode_store)
        self.stack_states.append(state)
        stacked_state = np.stack(list(self.stack_states), axis=0)
        stacked_state = torch.from_numpy(stacked_state).unsqueeze(0).float()

        with torch.no_grad():
            output = self.model(stacked_state)
            policy_map = output["policy"].squeeze().numpy()

        legal_action_map = get_valid_policy_map(obs, team_id, self.env_cfg)
        action_mask_map = np.ones_like(policy_map) * 1e32
        action_mask_map[legal_action_map > 0] = 0  # legal actionは0でそれ以外は1e32
        # 無効な行動は負の大きな値になるためsoftmax後は0になる。その上で再度無効な行動を0にする
        policy_map = softmax(policy_map - action_mask_map, axis=0) * (action_mask_map == 0) * 1
        return policy_map


cfg = Config()
seed_everything(cfg.seed, workers=True)
imitation_model = ILAgent(EnvParams, cfg.checkpoint_path, cfg.n_stack)


class Agent:
    def __init__(self, player: str, env_cfg: EnvParams) -> None:
        self.cfg = Config()
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(self.cfg.seed)
        self.env_cfg = env_cfg
        self.episode_store = EpisodeStore(self.team_id, env_cfg)
        self.prev_actions = {}

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # マッチごとにリセットされる要素をリセット
        if obs["match_steps"] == 0:
            self.episode_store.reset()
        self.episode_store.update(obs, self.prev_actions)
        policy_map = imitation_model.predict(obs, self.team_id, self.episode_store)

        unit_mask = np.array(obs["units_mask"][self.team_id])  # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id])  # shape (max_units, 2)
        available_unit_ids = np.where(unit_mask)[0]

        opp_unit_positions = np.array([pos for pos in obs["units"]["position"][self.opp_team_id] if pos[0] != -1])
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        # unit ids range from 0 to max_units - 1
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            x, y = unit_pos
            policy = policy_map[:, y, x]
            # print(policy, file=sys.stderr)

            if cfg.stochastic:
                action = np.random.choice(range(6), p=policy)
            else:
                action = policy.argmax()

            if action == Action.SAP:
                # params.unit_sap_rangeの範囲内にいる敵ユニットをランダムに選択
                opp_unit_ids = get_nearby_enemy_unit_ids(unit_pos, opp_unit_positions, self.env_cfg["unit_sap_range"])
                # 敵のユニットがいる場合はランダムにサンプリングしてSAPする
                if len(opp_unit_ids) > 0:
                    opp_unit_id = np.random.choice(opp_unit_ids)
                    opp_unit_pos = opp_unit_positions[opp_unit_id]
                    relative_pos = calc_relative_pos(unit_pos, opp_unit_pos)
                    # print(f"{unit_pos=}, {opp_unit_pos=}", file=sys.stderr)
                    actions[unit_id] = [Action.SAP, relative_pos[0], relative_pos[1]]
                else:
                    actions[unit_id] = [Action.CENTER, 0, 0]

            else:
                actions[unit_id] = [action, 0, 0]
        self.prev_actions = actions.copy()
        return actions


# 相対位置を計算
def calc_relative_pos(base_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    return target_pos - base_pos


# マスの半径kマス以内に該当するかどうか
def is_within_k_tiles(base_pos: np.ndarray, target_pos: np.ndarray, k: int) -> bool:
    return np.abs(base_pos[0] - target_pos[0]) <= k and np.abs(base_pos[1] - target_pos[1]) <= k


# 自身の周囲kタイル以内にいる敵ユニットを抽出
def get_nearby_enemy_unit_ids(unit_pos: np.ndarray, opp_unit_positions: np.ndarray, k: int) -> list[int]:
    return [unit_id for unit_id, pos in enumerate(opp_unit_positions) if is_within_k_tiles(unit_pos, pos, k)]
