from typing import Dict, Optional, Union, Any
import numpy as np
import gymnasium as gym
import copy
import jax
import os
import jax.numpy as jnp
from luxai_s3.env import LuxAIS3Env
# from luxai_s3.wrappers import LuxAIS3GymEnv
from luxai_s3.params import EnvParams, env_params_ranges
from luxai_s3.utils import to_numpy
import flax
import flax.serialization
import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_distributions import TorchDistribution, TorchMultiCategorical
from ray.rllib.utils.annotations import override
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
import torch
from torch import nn
from ray.tune.registry import register_env
from lux.utils import Action, State, HiddenState, extract_state, TileType
from lux.models import LuxUNetModel
from dataclasses import dataclass
import dataclasses
from pathlib import Path
from jax.tree_util import tree_map


@dataclass
class Config:
    exp_name: str = "exp014"
    model_name: str = "lux_unet"
    env_name: str = "lux-s3-v0"
    n_stack: int = 1
    pretrained_path: str | None = None # Path("/home/user/work/exp/exp013/best_model.ckpt")
    debug: bool = True


def env_creator(env_config: dict[str, Any]) -> MultiAgentEnv:
    """環境作成関数"""
    return RLLibLuxEnv(env_config)


class RLLibLuxEnv(MultiAgentEnv):
    """RLlibで使用するためのLuxAI S3環境のラッパー"""

    def __init__(self, config: dict = None):
        """
        Args:
            config (dict): 環境設定。以下のキーをサポート:
                - auto_reset (bool): エピソード終了時に自動リセットするかどうか
                - fixed_env_params (EnvParams): 固定環境パラメータ
                - seed (int): 乱数シード
        """
        super().__init__()

        config = config or {}
        self.auto_reset = config.get("auto_reset", False)
        self.fixed_env_params = config.get("fixed_env_params", EnvParams())

        # 基本環境の初期化
        self.env = LuxAIS3Env(
            auto_reset=self.auto_reset,
            fixed_env_params=self.fixed_env_params
        )

        # シードの設定
        # self.seed(config.get("seed", None))
    
        # 現在の状態の保持
        self.state = None

        # アクション・観測空間の設定
        self.action_spaces = self._create_action_space()
        self.observation_spaces = self._create_obs_space()

        # エージェントIDのリスト
        self.agents = self.possible_agents = ["player_0", "player_1"]
        self._agent_ids = set(self.agents)

        # reset時に更新
        self.rng_key = jax.random.key(0)
        self.env_params = self._set_params()

    def _create_action_space(self):
        num_actions = len(Action)
        num_units = EnvParams.max_units
        unit_sap_range = env_params_ranges["unit_sap_range"][-1] # 最大値で設定しておく
        low = np.zeros((num_units, 3))
        low[:, 1:] = -unit_sap_range
        high = np.ones((num_units, 3)) * num_actions
        high[:, 1:] = unit_sap_range
        return {
            "player_0": gym.spaces.MultiDiscrete([num_actions] * num_units),
            "player_1": gym.spaces.MultiDiscrete([num_actions] * num_units),
        }

    def _create_obs_space(self) -> gym.spaces.Dict:
        """
        EnvObsクラスに対応する観測空間をDictで定義し、返す例。
        """
        observation_space = gym.spaces.Dict({
            "units": gym.spaces.Dict({
                "position": gym.spaces.Box(
                    low=-1,
                    high=EnvParams.map_height,
                    shape=(EnvParams.num_teams, EnvParams.max_units, 2),  # N max units, 2 for x, y
                    dtype=np.int32
                ),
                "energy": gym.spaces.Box(
                    low=-1,
                    high=EnvParams.max_unit_energy,
                    shape=(EnvParams.num_teams, EnvParams.max_units),  # N max units
                    dtype=np.int32
                )
            }),
            "units_mask": gym.spaces.Box(
                low=0,
                high=1,
                shape=(EnvParams.num_teams, EnvParams.max_units),  # T teams, N max units
                dtype=np.bool_
            ),
            "sensor_mask": gym.spaces.Box(
                low=0,
                high=1,
                shape=(EnvParams.map_width, EnvParams.map_height),
                dtype=np.bool_
            ),
            "map_features": gym.spaces.Dict({
                "energy": gym.spaces.Box(
                    low=-1,
                    high=EnvParams.max_energy_per_tile,
                    shape=(EnvParams.map_width, EnvParams.map_height),
                    dtype=np.int32
                ),
                "tile_type": gym.spaces.Box(
                    low=-1,
                    high=2,
                    shape=(EnvParams.map_width, EnvParams.map_height),
                    dtype=np.int32
                )
            }),
            "relic_nodes": gym.spaces.Box(
                low=-1,
                high=EnvParams.map_height,
                shape=(EnvParams.max_relic_nodes, 2),  # N max relic nodes, 2 features for position (x, y)
                dtype=np.int32
            ),
            "relic_nodes_mask": gym.spaces.Box(
                low=0,
                high=1,
                shape=(EnvParams.max_relic_nodes,),  # N max relic nodes
                dtype=np.bool_
            ),
            "team_points": gym.spaces.Box(
                low=0,
                high=EnvParams.max_units * EnvParams.max_steps_in_match,
                shape=(EnvParams.num_teams,),  # T teams
                dtype=np.int32
            ),
            "team_wins": gym.spaces.Box(
                low=0,
                high=EnvParams.match_count_per_episode,
                shape=(EnvParams.num_teams,),  # T teams
                dtype=np.int32
            ),
            "steps": gym.spaces.Discrete(EnvParams.max_steps_in_match * EnvParams.match_count_per_episode),
            "match_steps": gym.spaces.Discrete(EnvParams.max_steps_in_match)
        })
        return {"player_0": observation_space, "player_1": observation_space}


    def _set_params(self) -> EnvParams:

        # generate random game parameters
        # TODO (stao): check why this keeps recompiling when marking structs as static args
        randomized_game_params = dict()
        for k, v in env_params_ranges.items():
            self.rng_key, subkey = jax.random.split(self.rng_key)
            randomized_game_params[k] = jax.random.choice(
                subkey, jax.numpy.array(v)
            ).item()
        params = EnvParams(**randomized_game_params)
        return params


    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        if seed is not None:
            self.rng_key = jax.random.key(seed)
        self.rng_key, reset_key = jax.random.split(self.rng_key)
        
        params = self._set_params()
        if options is not None and "params" in options:
            params = options["params"]

        self.env_params = params
        obs, self.state = self.env.reset(reset_key, params=self.env_params)
        obs = to_numpy(flax.serialization.to_state_dict(obs))
        return obs, {}

    def step(self, action_dict: dict[str, np.ndarray]):
        """環境のステップ実行

        Returns:
            obs (dict):   エージェントIDをキーとする観測辞書
            rewards (dict): エージェントIDをキーとするリワード辞書
            dones (dict): エージェントIDをキーとするdoneフラグ辞書 + "__all__"
            info (dict):  エージェントIDをキーとする情報辞書
        """
        # アクションの検証
        assert set(action_dict.keys()) == self._agent_ids, \
            f"Received actions for agents {action_dict.keys()}, expected {self._agent_ids}"

        # NumPy配列をJAX配列に変換
        action = tree_map(jnp.array, action_dict)

        # もともとの step_env() の戻り値は (obs, state, reward, terminated, truncated, info) の6要素
        self.rng_key, subkey = jax.random.split(self.rng_key)
        obs, self.state, rewards, terminated, truncated, info = self.env.step(
            key=subkey,
            state=self.state,
            action=action,
            params=self.env_params,
        )

        # JAX配列をNumPy配列に変換
        obs = tree_map(np.array, obs)
        rewards = tree_map(np.array, rewards)
        terminated = tree_map(np.array, terminated)
        truncated = tree_map(np.array, truncated)
        info = tree_map(lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x, info)

        # dones 辞書を作成（RLlibのフォーマット）
        # - 各agentのdone: terminated[agent] or truncated[agent] で計算
        # - "__all__": すべてのエージェントがdoneならTrue
        dones = {
            agent_id: bool(terminated[agent_id] or truncated[agent_id])
            for agent_id in self.agents
        }
        dones["__all__"] = all(dones.values())

        return obs, rewards, dones, {}, {}

    # def render(self):
    #     """環境の描画"""
    #     if self.state is not None:
    #         self.env.render(self.state, self.env.default_params)


    # @property
    # def get_agent_ids(self):
    #     """現在アクティブなエージェントIDのセットを返す"""
    #     return self._agent_ids.copy()


class LuxUnetTorchRLModule(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def setup(self):
        # Feel free to access the following useful properties in this class:
        # - `self.model_config`: The config dict for this RLModule class,
        self.model = LuxUNetModel(
            state_space_size=len(State),
            action_space_size=len(Action),
            hidden_state_space_size=len(HiddenState),
            n_stack=self.model_config["n_stack"],
            bilinear=True,
        )

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        # Compute the basic 1D feature tensor (inputs to policy- and value-heads).
        # state = extract_state(batch["obs"], target_team_id)
        # state = batch["obs"]
        # outputs = self.model(state)
        outputs = {
            # "policy": torch.zeros((1, len(Action), 24, 24)),
            "policy": torch.zeros((1, EnvParams.max_units, len(Action))),
            "value": torch.zeros((1, ))
        }
        policy_logits = outputs["policy"]
        self._values = outputs["value"]
        # Return features and logits as ACTION_DIST_INPUTS (categorical distribution).
        return {
            Columns.ACTION_DIST_INPUTS: policy_logits,
        }


    # We implement this RLModule as a ValueFunctionAPI RLModule, so it can be used
    # by value-based methods like PPO or IMPALA.
    @override(ValueFunctionAPI)
    def compute_values(self, batch: dict[str, Any]) -> torch.Tensor:
        return self._values

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self) -> type[TorchDistribution]:
        return TorchMultiCategorical

def create_rl_config(cfg: Config) -> AlgorithmConfig:
    # モデル設定
    tmp_env = env_creator({})
    rl_module_spec = RLModuleSpec(
        module_class=LuxUnetTorchRLModule,
        observation_space=tmp_env.get_observation_space("player_0"),
        action_space=tmp_env.get_action_space("player_0"),
        # モデル内部でself.model_config["key"]でアクセスできる
        model_config={
            "n_stack": cfg.n_stack,
            "pretrained_path": cfg.pretrained_path,
        },
    )
    config = (
        PPOConfig()
        # 環境設定
        .environment(
            env=cfg.env_name,
            env_config={"auto_reset": True}
        )
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
            train_batch_size=1024,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "player_0": copy.deepcopy(rl_module_spec),
                    "player_1": copy.deepcopy(rl_module_spec),
                }
            )
        )
        # マルチエージェント設定
        .multi_agent(
            policies=["player_0", "player_1"],
            policy_mapping_fn=lambda agent_id, *args, **kwargs: f"{agent_id}"
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
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "env_vars": {"RAY_DEBUG": "1"}
        }
    )
    # 環境の登録
    register_env(name=cfg.env_name, env_creator=env_creator)

    # カスタムモデルの登録
    ModelCatalog.register_custom_model(model_name=cfg.model_name, model_class=LuxUnetTorchRLModule)

    # 学習の設定
    config = create_rl_config(cfg)
    trainer = config.build_algo()

    num_iterations = 1
    for i in range(num_iterations):
        result = trainer.train()

if __name__ == "__main__":
    main()

