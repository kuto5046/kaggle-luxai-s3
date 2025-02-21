import os
import time
from typing import Any, Optional
from pathlib import Path
from collections import deque
from dataclasses import dataclass

import jax
import ray
import flax
import numpy as np
import torch
import gymnasium as gym
import jax.numpy as jnp
import flax.serialization
from lux.utils import (
    State,
    Action,
    GlobalState,
    HiddenState,
    EpisodeStore,
    extract_state,
    extract_global_state,
)
from lux.models import LuxUNetModel
from lux.params import EnvParams
from luxai_s3.env import LuxAIS3Env
from luxai_s3.utils import to_numpy
from luxai_s3.params import env_params_ranges
from ray.tune.registry import register_env
from ray.rllib.core.columns import Columns
from ray.rllib.utils.typing import ModuleID, TensorType, EpisodeType
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_utils import explained_variance
from ray.rllib.algorithms.ppo.ppo import (
    LEARNER_RESULTS_KL_KEY,
    LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY,
    LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY,
)
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.learner.learner import ENTROPY_KEY, VF_LOSS_KEY, POLICY_LOSS_KEY
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.models.torch.torch_distributions import TorchCategorical, TorchDistribution
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner

import wandb


@dataclass
class Config:
    exp_name: str = Path(__file__).parent.name
    notes: str = "rlをrayで動かす"
    model_name: str = "lux_unet"
    env_name: str = "lux-s3-v0"
    n_stack: int = 1
    pretrained_path: str | None = None
    debug: bool = True
    output_dir: str = Path(f"/home/user/work/exp/{exp_name}")
    # runner
    num_env_runners: int = 1  # actorの数
    num_cpus_per_env_runner: int = 1
    # learner
    gamma: float = 0.99
    lr: float = 1e-4
    minibatch_size: int = 64
    train_batch_size_per_learner: int = 505 * 3
    num_epochs: int = 2

    # def __post_init__(self):
    #     if self.debug:
    #         self.num_env_runners = 1
    #         self.num_cpus_per_env_runner = 1
    #         self.minibatch_size = 256
    #         self.train_batch_size_per_learner = 505
    #         self.num_epochs = 1


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
        self.agent0_global_states = deque(maxlen=self.n_stack)
        self.agent1_global_states = deque(maxlen=self.n_stack)

    def _set_params(self) -> EnvParams:
        randomized_game_params = {}
        for k, v in env_params_ranges.items():
            self.rng_key, subkey = jax.random.split(self.rng_key)
            randomized_game_params[k] = jax.random.choice(subkey, jnp.array(v)).item()
        return EnvParams(**randomized_game_params)

    def _create_action_space(self):
        num_actions = len(Action)
        cell_size = EnvParams.map_height * EnvParams.map_width
        # Boxは連続値の行動用なので離散アクションはMultiDiscreteを使う
        action_space = gym.spaces.MultiDiscrete([num_actions] * cell_size)
        return {
            "player_0": action_space,
            "player_1": action_space,
        }

    def _create_obs_space(self) -> gym.spaces.Dict:
        observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(
                    low=-100,
                    high=100,
                    shape=(self.n_stack, len(State), EnvParams.map_height, EnvParams.map_width),
                    dtype=np.float32,
                ),
                "global_state": gym.spaces.Box(
                    low=-100,
                    high=100,
                    shape=(self.n_stack, len(GlobalState)),
                    dtype=np.float32,
                ),
            }
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
        steps = obs["player_0"]["match_steps"]
        if steps == 0:
            self.episode_store1.reset()
            self.episode_store2.reset()
        else:
            self.episode_store1.update(obs["player_0"])
            self.episode_store2.update(obs["player_1"])

        agent0_state = extract_state(obs["player_0"], 0, self.episode_store1)
        agent1_state = extract_state(obs["player_1"], 1, self.episode_store2)
        agent0_global_state = extract_global_state(obs["player_0"], 0, self.env_params)
        agent1_global_state = extract_global_state(obs["player_1"], 1, self.env_params)

        self.agent0_states.append(agent0_state)
        self.agent1_states.append(agent1_state)
        self.agent0_global_states.append(agent0_global_state)
        self.agent1_global_states.append(agent1_global_state)
        return {
            "player_0": {
                "state": np.stack(list(self.agent0_states), axis=0),
                "global_state": np.stack(list(self.agent0_global_states), axis=0),
            },
            "player_1": {
                "state": np.stack(list(self.agent1_states), axis=0),
                "global_state": np.stack(list(self.agent1_global_states), axis=0),
            },
        }

    def _create_action(self, action_dict: dict[str, Any]) -> dict[str, np.ndarray]:
        actions = {agent_id: np.zeros((EnvParams.max_units, 3), dtype=np.int32) for agent_id in self.agents}

        for agent_idx, (agent_id, action_1dmap) in enumerate(action_dict.items()):
            action_2dmap = action_1dmap.reshape(EnvParams.map_height, EnvParams.map_width)
            for unit_id in range(EnvParams.max_units):
                x, y = self.state.units.position[agent_idx][unit_id]
                unit_action = action_2dmap[y, x]
                actions[agent_id][unit_id, 0] = unit_action
                if unit_action == Action.SAP:
                    # TODO: sapの場合ap方策も適用する
                    pass
        return actions

    def step(self, action_dict: dict[str, Any]) -> tuple:
        self.rng_key, step_key = jax.random.split(self.rng_key)
        actions = self._create_action(action_dict)
        obs, self.state, _reward, _terminated, _truncated, _ = self.env.step(
            step_key, self.state, actions, self.env_params
        )
        obs = to_numpy(flax.serialization.to_state_dict(obs))
        state = self._create_state(obs)

        _reward = to_numpy(_reward)
        reward = {agent_id: int(r.item()) for agent_id, r in _reward.items()}
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
            global_state_space_size=len(GlobalState),
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
        batch_size = batch[Columns.OBS]["state"].shape[0]
        outputs = self.model(batch[Columns.OBS])
        policy_logits = outputs["policy"]
        num_actions = policy_logits.shape[1]
        # この時点では(batch, action, height, width)なので(batch, height*width, action)に変換
        policy_logits = policy_logits.reshape(batch_size, num_actions, -1).transpose(2, 1)
        return {
            Columns.ACTION_DIST_INPUTS: policy_logits,
        }

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        batch_size = batch[Columns.OBS]["state"].shape[0]
        outputs = self.model(batch[Columns.OBS])
        policy_logits = outputs["policy"]
        num_actions = policy_logits.shape[1]
        policy_logits = policy_logits.reshape(batch_size, num_actions, -1).transpose(2, 1)

        return {
            Columns.ACTION_DIST_INPUTS: policy_logits,
        }

    @override(ValueFunctionAPI)
    def compute_values(self, batch: dict[str, Any], embeddings: Any | None = None) -> torch.Tensor:
        # outputs = self.model(batch[Columns.OBS])
        batch_size = batch[Columns.OBS]["state"].shape[0]
        self._values = torch.zeros(batch_size, 1)
        return self._values

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self) -> type[TorchDistribution]:
        return TorchCategorical


def train_metric_value(results: dict[str, Any], key: str) -> float:
    # self play前提でplayer_0とplayer_1の値の平均を返す
    return (results["player_0"][key] + results["player_1"][key]) / 2


class WandbLoggerCallback(RLlibCallback):
    def __init__(self):
        self.episode_count = 0
        self.start_time = time.time()

    @override(RLlibCallback)
    def on_train_result(
        self,
        *,
        algorithm: "Algorithm",
        metrics_logger: MetricsLogger | None = None,
        result: dict,
        **kwargs,
    ) -> None:
        """Called at the end of Algorithm.train().

        Args:
            algorithm: Current Algorithm instance.
            metrics_logger: The MetricsLogger object inside the Algorithm. Can be
                used to log custom metrics after traing results are available.
            result: Dict of results returned from Algorithm.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """

        learner_metrics = [
            # loss
            "total_loss",
            "vf_loss",
            "vf_loss_unclipped",
            "policy_loss",
            "mean_kl_loss",
            # other
            "entropy",
            "vf_explained_var",
            "default_optimizer_learning_rate",
        ]
        for key in learner_metrics:
            wandb.log(
                {
                    f"train/{key}": train_metric_value(result["learners"], key),
                }
            )

    def on_evaluate_end(
        self,
        *,
        algorithm: "Algorithm",
        metrics_logger: MetricsLogger | None = None,
        evaluation_metrics: dict,
        **kwargs,
    ) -> None:
        """Runs when the evaluation is done.

        Runs at the end of Algorithm.evaluate().

        Args:
            algorithm: Reference to the algorithm instance.
            metrics_logger: The MetricsLogger object inside the `Algorithm`. Can be
                used to log custom metrics after the most recent evaluation round.
            evaluation_metrics: Results dict to be returned from algorithm.evaluate().
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        print(f"############ on_evaluate_end {evaluation_metrics.keys()=}")
        wandb.log(
            {
                # player_0を自身として評価している
                "evaluate/agent_episode_returns_mean": evaluation_metrics["env_runners"]["agent_episode_returns_mean"][
                    "player_0"
                ],
                "evaluate/episode_duration_sec_mean": evaluation_metrics["env_runners"]["episode_duration_sec_mean"],
            }
        )

    @override(RLlibCallback)
    def on_episode_end(
        self,
        *,
        episode: EpisodeType,
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: MetricsLogger | None = None,
        **kwargs,
    ) -> None:
        # エピソード情報のログ（例としてrewardやdurationを出力）
        rewards = episode.get_rewards()  # 各エピソードの報酬履歴を取得
        reward = {agent_id: rewards[agent_id][-1] for agent_id in rewards.keys()}
        duration = episode.get_duration_s()
        self.episode_count += 1

        # 経過時間を計測し、全体のepisode収集速度（episode/sec）を計算
        elapsed_time = time.time() - self.start_time
        episode_collection_speed = self.episode_count / elapsed_time

        # wandb へログ出力（キー名は任意に変更可）
        print(
            f"episode {self.episode_count} finished. {reward=} {duration=:0.2f}s {episode_collection_speed=:0.2f}ep/s"
        )
        # wandb.log({"train/episode_collection_speed": episode_collection_speed})


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

        # Calculate log probability ratio for each position and sum across spatial dimensions
        # logp_ratio = torch.exp(curr_action_dist.logp(batch[Columns.ACTIONS]) - batch[Columns.ACTION_LOGP])
        logp_ratio = torch.exp((curr_action_dist.logp(batch[Columns.ACTIONS]) - batch[Columns.ACTION_LOGP]).sum(dim=1))

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if config.use_kl_loss:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = possibly_masked_mean(action_kl)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy().sum(dim=1)  # (batch, height*width) -> (batch)
        mean_entropy = possibly_masked_mean(curr_entropy)

        # MEMO: advantagesをunit数分に拡張する
        # batch_size = batch[Postprocessing.ADVANTAGES].shape[0]
        # advantages = batch[Postprocessing.ADVANTAGES].view(batch_size, 1).repeat(1, EnvParams.max_units)
        advantages = batch[Postprocessing.ADVANTAGES]
        surrogate_loss = torch.min(
            advantages * logp_ratio,
            advantages * torch.clamp(logp_ratio, 1 - config.clip_param, 1 + config.clip_param),
        )

        # Compute a value function loss.
        if config.use_critic:
            value_fn_out = module.compute_values(batch, embeddings=fwd_out.get(Columns.EMBEDDINGS))
            vf_loss = torch.pow(value_fn_out - batch[Postprocessing.VALUE_TARGETS], 2.0)
            vf_loss_clipped = torch.clamp(vf_loss, 0, config.vf_clip_param)
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
            num_env_runners=cfg.num_env_runners,
            # num_envs_per_env_runner=cfg.num_envs_per_env_runner,  # multi agentはenv vectorizationが未対応
            num_cpus_per_env_runner=cfg.num_cpus_per_env_runner,
            sample_timeout_s=60 * 5,
        )
        # モデルを学習するlearnerの数。gpuの数と合わせる
        .learners(
            num_learners=1,
            num_cpus_per_learner=1,
            num_gpus_per_learner=1,
        )
        # 学習パラメータ設定
        .training(
            learner_class=CustomPPOTorchLearner,
            # PPOの設定
            use_critic=True,
            use_gae=True,
            use_kl_loss=True,
            # 一般的な学習の設定
            gamma=cfg.gamma,
            lr=cfg.lr,
            minibatch_size=cfg.minibatch_size,
            train_batch_size_per_learner=cfg.train_batch_size_per_learner,  # 3試合データが集まったら学習する
            num_epochs=cfg.num_epochs,
        )
        # .python_environment(
        #     extra_python_environs_for_worker={
        #         "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
        #         "OMP_NUM_THREADS": "1",
        #     }
        # )
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
        .callbacks(WandbLoggerCallback)
        .evaluation(
            evaluation_interval=1,
            evaluation_duration=10,
            evaluation_duration_unit="episodes",
        )
    )
    return config


def setup_wandb(cfg: Config):
    wandb.init(
        project="kaggle-luxai-s3",
        entity="kuto5046",
        group=cfg.exp_name,
        notes=cfg.notes,
        mode="disabled" if cfg.debug else "online",
    )


def main() -> None:
    cfg = Config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    setup_wandb(cfg)
    # debug mode
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "env_vars": {
                "RAY_DEBUG": "1",
            }
        },
    )
    # 環境の登録
    register_env(name=cfg.env_name, env_creator=env_creator)

    config = create_rl_config(cfg)
    trainer = config.build_algo(env=cfg.env_name)
    result = trainer.train()

    checkpoint_dir = trainer.save_to_path(cfg.output_dir)
    print(f"save to {checkpoint_dir}")
    # プログラムを終了
    os._exit(0)


if __name__ == "__main__":
    main()
