import os
from typing import Any
from dataclasses import dataclass

import jax
import flax
import numpy as np
import torch
import gymnasium as gym
import jax.numpy as jnp
import flax.serialization
from lux.utils import State, Action, HiddenState, EpisodeStore, extract_state
from lux.models import LuxUNetModel
from lux.params import EnvParams
from luxai_s3.env import LuxAIS3Env
from luxai_s3.utils import to_numpy
from luxai_s3.params import env_params_ranges

import ray
from ray.tune.registry import register_env
from ray.rllib.core.columns import Columns
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.annotations import override
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.models.torch.torch_distributions import TorchCategorical, TorchDistribution
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule


@dataclass
class Config:
    exp_name: str = "exp014"
    model_name: str = "lux_unet"
    env_name: str = "lux-s3-v0"
    n_stack: int = 1
    pretrained_path: str | None = None
    debug: bool = True


def env_creator(config: dict[str, Any]) -> MultiAgentEnv:
    """環境作成関数"""
    return RLLibLuxEnv(config)


class RLLibLuxEnv(MultiAgentEnv):
    """
    MultiAgentEnvは古いapi形式であるためMultiAgentEnvCompatibilityを継承することが推奨される
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.env = LuxAIS3Env()

        self.state = None
        # アクション・観測空間の設定
        self.action_spaces = self._create_action_space()
        self.observation_spaces = self._create_obs_space()

        self.agents = self.possible_agents = ["player_0", "player_1"]
        self._agent_ids = set(self.agents)

        # reset時に更新
        self.rng_key = jax.random.PRNGKey(0)
        self.env_params = self._set_params()
        self.episode_store1 = EpisodeStore(target_team_id=0, env_cfg=self.env_params)
        self.episode_store2 = EpisodeStore(target_team_id=1, env_cfg=self.env_params)

    def _set_params(self) -> EnvParams:
        randomized_game_params = {}
        for k, v in env_params_ranges.items():
            self.rng_key, subkey = jax.random.split(self.rng_key)
            randomized_game_params[k] = jax.random.choice(subkey, jnp.array(v)).item()
        return EnvParams(**randomized_game_params)

    def _create_action_space(self):
        num_actions = len(Action)
        num_units = EnvParams.max_units
        low = np.zeros(num_units)
        high = np.ones(num_units) * num_actions
        return {
            "player_0": gym.spaces.Box(low=low, high=high, shape=(num_units,), dtype=np.int32),
            "player_1": gym.spaces.Box(low=low, high=high, shape=(num_units,), dtype=np.int32),
        }

    def _create_obs_space(self) -> gym.spaces.Dict:
        observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(len(State), EnvParams.map_height, EnvParams.map_width), dtype=np.float32
        )
        return {"player_0": observation_space, "player_1": observation_space}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self.rng_key = jax.random.PRNGKey(seed)
        self.rng_key, reset_key = jax.random.split(self.rng_key)

        params = self._set_params()

        self.env_params = params
        obs, self.state = self.env.reset(reset_key, params=self.env_params)
        obs = to_numpy(flax.serialization.to_state_dict(obs))
        infos = {k: {} for k in obs.keys()}

        self.episode_store1 = EpisodeStore(target_team_id=0, env_cfg=self.env_params)
        self.episode_store2 = EpisodeStore(target_team_id=1, env_cfg=self.env_params)
        state = {
            "player_0": extract_state(obs["player_0"], 0, self.episode_store1),
            "player_1": extract_state(obs["player_1"], 1, self.episode_store2),
        }
        return state, infos

    def step(self, action_dict: dict[str, Any]) -> tuple:
        self.rng_key, step_key = jax.random.split(self.rng_key)
        # actionを(16,) -> (16, 3)に変換。ただしsap時も0になっている
        action = {agent_id: np.zeros((EnvParams.max_units, 3), dtype=np.int32) for agent_id in self.agents}
        action["player_0"][:, 0] = action_dict["player_0"]
        action["player_1"][:, 0] = action_dict["player_1"]

        obs, self.state, reward, _terminated, _truncated, _ = self.env.step(
            step_key, self.state, action, self.env_params
        )
        obs = to_numpy(flax.serialization.to_state_dict(obs))
        state = {
            "player_0": extract_state(obs["player_0"], 0, self.episode_store1),
            "player_1": extract_state(obs["player_1"], 1, self.episode_store2),
        }

        reward = to_numpy(reward)
        terminated = {agent_id: done.item() for agent_id, done in _terminated.items()}

        truncated = {agent_id: done.item() for agent_id, done in _truncated.items()}
        # "__all__" (required) is used to indicate env termination.
        terminated["__all__"] = np.all(list(truncated.values()))  # luxaiはtruncatedがTrueになる
        info = {agent_id: {} for agent_id in self.agents}

        return state, reward, terminated, truncated, info


class LuxUnetTorchRLModule(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def setup(self):
        self.model = LuxUNetModel(
            state_space_size=len(State),
            action_space_size=len(Action),
            hidden_state_space_size=len(HiddenState),
            n_stack=self.model_config["n_stack"],
            bilinear=True,
        )

        # TODO: weight読み込み
        if self.model_config["pretrained_path"]:
            pass

        self._values = None

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        batch_size = batch[Columns.OBS].shape[0]
        outputs = {
            "policy": torch.zeros((batch_size, EnvParams.max_units, len(Action))),
            "value": torch.rand((batch_size,)),
        }
        policy_logits = outputs["policy"]
        return {
            Columns.ACTION_DIST_INPUTS: policy_logits,
        }

    @override(ValueFunctionAPI)
    def compute_values(self, batch: dict[str, Any]) -> torch.Tensor:
        batch_size = batch[Columns.OBS].shape[0]
        outputs = {
            "policy": torch.zeros((batch_size, EnvParams.max_units, len(Action))),
            "value": torch.rand((batch_size,)),
        }
        self._values = outputs["value"].tanh()
        return self._values

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self) -> type[TorchDistribution]:
        return TorchCategorical


def create_rl_config(cfg: Config) -> AlgorithmConfig:
    tmp_env = env_creator({})
    observation_space = tmp_env.get_observation_space("player_0")
    action_space = tmp_env.get_action_space("player_0")
    rl_module_spec = RLModuleSpec(
        module_class=LuxUnetTorchRLModule,
        observation_space=observation_space,
        action_space=action_space,
        # モデル内部でself.model_config["key"]でアクセスできる
        model_config={
            "n_stack": cfg.n_stack,
            "pretrained_path": cfg.pretrained_path,
        },
    )

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        # 環境設定
        .environment(env=cfg.env_name, env_config={})
        # ゲームをしてデータを生成するrunnerの数. cpuの数と合わせる
        .env_runners(
            num_env_runners=os.cpu_count() if not cfg.debug else 1,
        )
        # モデルを学習するlearnerの数。gpuの数と合わせる
        .learners(num_learners=1)
        # 学習パラメータ設定
        .training(
            gamma=0.99,
            lr=1e-4,
            minibatch_size=1024,
            train_batch_size=1024,
        )
        # https://docs.ray.io/en/latest/rllib/rllib-rlmodule.html#construction-through-rlmodulespecs
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                # All agents (0 and 1) use the same (single) RLModule.
                rl_module_specs={
                    "player_0": rl_module_spec,
                    "player_1": rl_module_spec,
                }
            )
        )
        # マルチエージェント設定
        .multi_agent(
            policies=["player_0", "player_1"], policy_mapping_fn=lambda agent_id, *args, **kwargs: f"{agent_id}"
        )
        .framework(
            framework="torch",
            eager_tracing=True,
        )
    )
    return config


def main() -> None:
    cfg = Config()
    # debug mode
    ray.init(ignore_reinit_error=True, runtime_env={"env_vars": {"RAY_DEBUG": "1"}})
    # 環境の登録
    register_env(name=cfg.env_name, env_creator=env_creator)

    config = create_rl_config(cfg)
    trainer = config.build_algo(env=cfg.env_name)

    num_iterations = 1
    for i in range(num_iterations):
        result = trainer.train()
        print(f"Iteration {i} result:", result)


if __name__ == "__main__":
    main()
