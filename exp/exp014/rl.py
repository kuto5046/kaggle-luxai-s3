import os
from typing import Any
from collections import deque
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
from ray.rllib.utils.typing import ModuleID, TensorType
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_utils import explained_variance
from ray.rllib.algorithms.ppo.ppo import (
    LEARNER_RESULTS_KL_KEY,
    LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY,
    LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY,
)
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.core.learner.learner import ENTROPY_KEY, VF_LOSS_KEY, POLICY_LOSS_KEY
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.models.torch.torch_distributions import TorchCategorical, TorchDistribution
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner


@dataclass
class Config:
    exp_name: str = "exp014"
    model_name: str = "lux_unet"
    env_name: str = "lux-s3-v0"
    n_stack: int = 1
    pretrained_path: str | None = None
    debug: bool = True


def env_creator(config: dict[str, Any]) -> MultiAgentEnv:
    return RLLibLuxEnv(config)


class RLLibLuxEnv(MultiAgentEnv):
    """
    MultiAgentEnvは古いapi形式であるためMultiAgentEnvCompatibilityを継承することが推奨される
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.env = LuxAIS3Env()
        self.n_stack = config["n_stack"]
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

        self.agent0_states = deque(maxlen=self.n_stack)
        self.agent1_states = deque(maxlen=self.n_stack)

    def _set_params(self) -> EnvParams:
        randomized_game_params = {}
        for k, v in env_params_ranges.items():
            self.rng_key, subkey = jax.random.split(self.rng_key)
            randomized_game_params[k] = jax.random.choice(subkey, jnp.array(v)).item()
        return EnvParams(**randomized_game_params)

    def _create_action_space(self):
        num_actions = len(Action)
        num_units = EnvParams.max_units
        # Boxは連続値の行動用なので離散アクションはMultiDiscreteを使う
        return {
            "player_0": gym.spaces.MultiDiscrete([num_actions] * num_units),
            "player_1": gym.spaces.MultiDiscrete([num_actions] * num_units),
        }

    def _create_obs_space(self) -> gym.spaces.Dict:
        observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(self.n_stack, len(State), EnvParams.map_height, EnvParams.map_width),
            dtype=np.float32,
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
        state = self._create_state(obs)
        return state, infos

    def _create_state(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        agent0_state = extract_state(obs["player_0"], 0, self.episode_store1)
        agent1_state = extract_state(obs["player_1"], 1, self.episode_store2)
        self.agent0_states.append(agent0_state)
        self.agent1_states.append(agent1_state)
        return {
            "player_0": np.stack(list(self.agent0_states), axis=0),
            "player_1": np.stack(list(self.agent1_states), axis=0),
        }

    def step(self, action_dict: dict[str, Any]) -> tuple:
        self.rng_key, step_key = jax.random.split(self.rng_key)
        # actionを(16,) -> (16, 3)に変換。ただしsap時も0になっている
        action = {agent_id: np.zeros((EnvParams.max_units, 3), dtype=np.int32) for agent_id in self.agents}
        print(action_dict)
        action["player_0"][:, 0] = action_dict["player_0"]
        action["player_1"][:, 0] = action_dict["player_1"]

        obs, self.state, reward, _terminated, _truncated, _ = self.env.step(
            step_key, self.state, action, self.env_params
        )
        obs = to_numpy(flax.serialization.to_state_dict(obs))
        state = self._create_state(obs)

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
        outputs = self.model(batch[Columns.OBS])
        policy_logits = outputs["own_policy"]
        # この時点では(batch, action, height, width)なので、(batch, action)に変換
        # unitの位置を取得
        unit_positions = batch[Columns.OBS][State.UNIT_POSITIONS]
        return {
            Columns.ACTION_DIST_INPUTS: policy_logits,
        }

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        outputs = self.model(batch[Columns.OBS])
        policy_logits = outputs["own_policy"]
        self._values = outputs["value"].tanh()
        return {
            Columns.ACTION_DIST_INPUTS: policy_logits,
        }

    @override(ValueFunctionAPI)
    def compute_values(self, batch: dict[str, Any], embeddings: Any | None = None) -> torch.Tensor:
        return self._values

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self) -> type[TorchDistribution]:
        return TorchCategorical


class CustomPPOTorchLearner(PPOTorchLearner):
    """Implements torch-specific PPO loss logic on top of PPOLearner.

    This class implements the ppo loss under `self.compute_loss_for_module()`.
    """

    @override(PPOTorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig,
        batch: dict[str, Any],
        fwd_out: dict[str, TensorType],
    ) -> TensorType:
        module = self.module[module_id].unwrapped()

        # Possibly apply masking to some sub loss terms and to the total loss term
        # at the end. Masking could be used for RNN-based model (zero padded `batch`)
        # and for PPO's batched value function (and bootstrap value) computations,
        # for which we add an (artificial) timestep to each episode to
        # simplify the actual computation.
        if Columns.LOSS_MASK in batch:
            mask = batch[Columns.LOSS_MASK]
            num_valid = torch.sum(mask)

            def possibly_masked_mean(data_):
                return torch.sum(data_[mask]) / num_valid

        else:
            possibly_masked_mean = torch.mean

        action_dist_class_train = module.get_train_action_dist_cls()
        action_dist_class_exploration = module.get_exploration_action_dist_cls()

        curr_action_dist = action_dist_class_train.from_logits(fwd_out[Columns.ACTION_DIST_INPUTS])
        # TODO (sven): We should ideally do this in the LearnerConnector (separation of
        #  concerns: Only do things on the EnvRunners that are required for computing
        #  actions, do NOT do anything on the EnvRunners that's only required for a
        #   training update).
        prev_action_dist = action_dist_class_exploration.from_logits(batch[Columns.ACTION_DIST_INPUTS])

        logp_ratio = torch.exp(curr_action_dist.logp(batch[Columns.ACTIONS]) - batch[Columns.ACTION_LOGP])

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if config.use_kl_loss:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = possibly_masked_mean(action_kl)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = possibly_masked_mean(curr_entropy)

        # MEMO: advantagesをunit数分に拡張する
        batch_size = batch[Postprocessing.ADVANTAGES].shape[0]
        advantages = batch[Postprocessing.ADVANTAGES].view(batch_size, 1).repeat(1, EnvParams.max_units)

        surrogate_loss = torch.min(
            advantages * logp_ratio,
            advantages * torch.clamp(logp_ratio, 1 - config.clip_param, 1 + config.clip_param),
        )

        # Compute a value function loss.
        if config.use_critic:
            value_fn_out = module.compute_values(batch, embeddings=fwd_out.get(Columns.EMBEDDINGS))

            vf_loss = torch.pow(value_fn_out - batch[Postprocessing.VALUE_TARGETS], 2.0)
            vf_loss_clipped = torch.clamp(vf_loss, 0, config.vf_clip_param)
            vf_loss_clipped = vf_loss_clipped.view(batch_size, 1).repeat(1, EnvParams.max_units)
            mean_vf_loss = possibly_masked_mean(vf_loss_clipped)
            mean_vf_unclipped_loss = possibly_masked_mean(vf_loss)
        # Ignore the value function -> Set all to 0.0.
        else:
            z = torch.tensor(0.0, device=surrogate_loss.device)
            value_fn_out = mean_vf_unclipped_loss = vf_loss_clipped = mean_vf_loss = z

        total_loss = possibly_masked_mean(
            -surrogate_loss
            + config.vf_loss_coeff * vf_loss_clipped
            - (self.entropy_coeff_schedulers_per_module[module_id].get_current_value() * curr_entropy)
        )

        # Add mean_kl_loss (already processed through `possibly_masked_mean`),
        # if necessary.
        if config.use_kl_loss:
            total_loss += self.curr_kl_coeffs_per_module[module_id] * mean_kl_loss

        # Log important loss stats.
        self.metrics.log_dict(
            {
                POLICY_LOSS_KEY: -possibly_masked_mean(surrogate_loss),
                VF_LOSS_KEY: mean_vf_loss,
                LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY: mean_vf_unclipped_loss,
                LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY: explained_variance(
                    batch[Postprocessing.VALUE_TARGETS], value_fn_out
                ),
                ENTROPY_KEY: mean_entropy,
                LEARNER_RESULTS_KL_KEY: mean_kl_loss,
            },
            key=module_id,
            window=1,  # <- single items (should not be mean/ema-reduced over time).
        )
        # Return the total loss.
        return total_loss


def create_rl_config(cfg: Config) -> AlgorithmConfig:
    tmp_env = env_creator({"n_stack": cfg.n_stack})
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
        .environment(env=cfg.env_name, env_config={"n_stack": cfg.n_stack})
        # ゲームをしてデータを生成するrunnerの数. cpuの数と合わせる
        .env_runners(
            num_env_runners=os.cpu_count() if not cfg.debug else 1,
        )
        # モデルを学習するlearnerの数。gpuの数と合わせる
        .learners(num_learners=1)
        # 学習パラメータ設定
        .training(
            learner_class=CustomPPOTorchLearner,
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
