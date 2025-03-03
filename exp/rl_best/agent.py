from pathlib import Path

import torch
from lightning import seed_everything
from lux.utils import (
    State,
    Action,
    GlobalState,
    EpisodeStore,
)
from lux.models import LuxUNetModel
from lux.params import EnvParams
from lux.imitation_agent import ILAgent, load_model, policy_to_action


class Config:
    seed: int = 2025
    # 確率的な行動を取るかどうか
    stochastic: bool = True  # Falseにするとargmaxで行動を選択する
    res: bool = True
    n_stack: int = 4
    # 同じマスに複数のユニットが移動する場合のペナルティ、0=重複を許可(greedy)、1=重複を禁止
    overlap_penalty: float = 2.0

    tta: bool = False

    checkpoint_path: Path = Path(__file__).parent / "output/best_model.ckpt"


cfg = Config()
seed_everything(cfg.seed, workers=True)
# 提出時にエラーとなるためグローバルでモデルを定義
model = LuxUNetModel(
    state_space_size=len(State),
    global_state_space_size=len(GlobalState),
    action_space_size=len(Action),
    n_stack=cfg.n_stack,
    res=cfg.res,
)
model = load_model(model, cfg.checkpoint_path)
imitation_model = ILAgent(model, cfg)


class Agent:
    def __init__(self, player: str, env_cfg: dict) -> None:
        torch.set_num_threads(1)
        self.cfg = Config()
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        # np.random.seed(self.cfg.seed)
        self.env_cfg = EnvParams(**env_cfg)
        self.episode_store = EpisodeStore(self.team_id, self.env_cfg)
        self.prev_actions = None

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # マッチごとにリセットされる要素をリセット
        if obs["match_steps"] == 0:
            self.episode_store.reset()
        else:
            self.episode_store.update(obs, self.prev_actions)
        policy_map, _point_map, sap_map = imitation_model.predict(obs, self.team_id, self.episode_store)
        actions = policy_to_action(
            policy_map, sap_map, obs, self.team_id, self.env_cfg, self.cfg.stochastic, self.cfg.overlap_penalty
        )
        self.prev_actions = actions
        return actions
