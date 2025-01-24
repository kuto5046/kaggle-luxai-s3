from typing import Dict, Optional, Union, Any
import numpy as np
import gymnasium as gym
import jax
import jax.numpy as jnp
from luxai_s3.env import LuxAIS3Env
from luxai_s3.state import EnvParams
import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
import torch
from torch import nn
from ray.tune.registry import register_env
from lux.utils import Action, State, HiddenState
from lux.models import LuxUNetModel
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    exp_name: str = "exp014"
    model_name: str = "lux_unet"
    env_name: str = "lux-s3-v0"
    n_stack: int = 1
    pretrained_path: str | None = None # Path("/home/user/work/exp/exp013/best_model.ckpt")


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
        self.seed(config.get("seed", None))

        # 現在の状態の保持
        self.state = None

        # アクション・観測空間の設定
        self._action_space = self.env.action_space()
        self._observation_space = self._create_obs_space()

        # エージェントIDのリスト
        self.agents = ["player_0", "player_1"]
        self._agent_ids = set(self.agents)

    def _create_obs_space(self) -> gym.spaces.Dict:
        """観測空間の作成"""
        return gym.spaces.Dict({
            "units": gym.spaces.Dict({
                "position": gym.spaces.Box(low=-1, high=np.inf, shape=(self.fixed_env_params.max_units, 2), dtype=np.int16),
                "energy": gym.spaces.Box(low=-1, high=np.inf, shape=(self.fixed_env_params.max_units,), dtype=np.int16),
            }),
            "units_mask": gym.spaces.Box(low=0, high=1, shape=(self.fixed_env_params.max_units,), dtype=bool),
            "sensor_mask": gym.spaces.Box(low=0, high=1, shape=(self.fixed_env_params.map_width, self.fixed_env_params.map_height), dtype=bool),
            "map_features": gym.spaces.Dict({
                "energy": gym.spaces.Box(low=-1, high=np.inf, shape=(self.fixed_env_params.map_width, self.fixed_env_params.map_height), dtype=np.int16),
                "tile_type": gym.spaces.Box(low=-1, high=np.inf, shape=(self.fixed_env_params.map_width, self.fixed_env_params.map_height), dtype=np.int16),
            }),
            "team_points": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.fixed_env_params.num_teams,), dtype=np.int32),
            "team_wins": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.fixed_env_params.num_teams,), dtype=np.int32),
            "steps": gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=np.int32),
            "match_steps": gym.spaces.Box(low=-1, high=np.inf, shape=(), dtype=np.int32),
            "relic_nodes": gym.spaces.Box(low=-1, high=np.inf, shape=(self.fixed_env_params.max_relic_nodes, 2), dtype=np.int16),
            "relic_nodes_mask": gym.spaces.Box(low=0, high=1, shape=(self.fixed_env_params.max_relic_nodes,), dtype=bool),
        })

    @property
    def action_space(self) -> gym.spaces.Dict:
        """アクション空間を取得"""
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Dict:
        """観測空間を取得"""
        return self._observation_space

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """環境のリセット

        Args:
            seed: 乱数シード
            options: 追加オプション(未使用)

        Returns:
            tuple: (observations dict, info dict)
        """
        if seed is not None:
            self.seed(seed)

        self.key, subkey = jax.random.split(self.key)
        obs, self.state = self.env.reset(subkey)

        # JAX配列をNumPy配列に変換
        obs = jax.tree_map(np.array, obs)

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
        action = jax.tree_map(jnp.array, action_dict)

        # もともとの step_env() の戻り値は (obs, state, reward, terminated, truncated, info) の6要素
        self.key, subkey = jax.random.split(self.key)
        obs, self.state, rewards, terminated, truncated, info = self.env.step(
            subkey,
            self.state,
            action
        )

        # JAX配列をNumPy配列に変換
        obs = jax.tree_map(np.array, obs)
        rewards = jax.tree_map(np.array, rewards)
        terminated = jax.tree_map(np.array, terminated)
        truncated = jax.tree_map(np.array, truncated)
        info = jax.tree_map(lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x, info)

        # dones 辞書を作成（RLlibのフォーマット）
        # - 各agentのdone: terminated[agent] or truncated[agent] で計算
        # - "__all__": すべてのエージェントがdoneならTrue
        dones = {
            agent_id: bool(terminated[agent_id] or truncated[agent_id])
            for agent_id in self.agents
        }
        dones["__all__"] = all(dones.values())

        # RLlibが想定する返り値 (obs, rewards, dones, info)
        return obs, rewards, dones, info

    def seed(self, seed: int | None = None):
        """乱数シードの設定"""
        if seed is not None:
            self.key = jax.random.PRNGKey(seed)
        else:
            self.key = jax.random.PRNGKey(0)

    def render(self):
        """環境の描画"""
        if self.state is not None:
            self.env.render(self.state, self.env.default_params)

    def close(self):
        """環境のクリーンアップ"""
        pass

    @property
    def get_agent_ids(self):
        """現在アクティブなエージェントIDのセットを返す"""
        return self._agent_ids.copy()


class LuxUnetTorchRLModule(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def setup(self):
        # Feel free to access the following useful properties in this class:
        # - `self.model_config`: The config dict for this RLModule class,
        self.model = LuxUNetModel(
            state_space_size=len(State),
            action_space_size=len(Action),
            hidden_state_size=len(HiddenState),
            n_stack=self.model_config["n_stack"],
            bilinear=True,
        )

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        # Compute the basic 1D feature tensor (inputs to policy- and value-heads).
        outputs = self.model(batch)
        policy_logits = outputs["policy"]
        self._values = outputs["value"]
        # Return features and logits as ACTION_DIST_INPUTS (categorical distribution).
        return {
            Columns.ACTION_DIST_INPUTS: policy_logits,
        }

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        return self._forward(batch, **kwargs)


    # We implement this RLModule as a ValueFunctionAPI RLModule, so it can be used
    # by value-based methods like PPO or IMPALA.
    @override(ValueFunctionAPI)
    def compute_values(self, batch: dict[str, Any]) -> torch.Tensor:
        return self._values


def create_rl_config(cfg: Config) -> AlgorithmConfig:
    config = (
        PPOConfig()
        # 環境設定
        .environment(
            env=cfg.env_name,
            env_config={"auto_reset": True}
        )
        # ゲームをしてデータを生成するrunnerの数. cpuの数と合わせる
        .env_runners(
            num_env_runners=2,
        )
        # モデルを学習するlearnerの数。gpuの数と合わせる
        .learners(num_learners=1)
        # 学習パラメータ設定
        .training(
            gamma=0.99,
            lr=1e-4,
            train_batch_size=1024,
        )
        # モデル設定
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=LuxUnetTorchRLModule,
                # モデル内部でself.model_config["key"]でアクセスできる
                model_config={
                    "n_stack": cfg.n_stack,
                    "pretrained_path": cfg.pretrained_path,
                },
            ),
        )
        # マルチエージェント設定
        .multi_agent(
            policies=["player_0_policy", "player_1_policy"],
            policy_mapping_fn=lambda agent_id, *args, **kwargs: f"{agent_id}_policy"
        )
    )
    return config




def main() -> None:
    cfg = Config()
    ray.init(ignore_reinit_error=True)
    # 環境の登録
    register_env(cfg.env_name, env_creator)

    # カスタムモデルの登録
    ModelCatalog.register_custom_model(cfg.model_name, LuxUnetTorchRLModule)

    # 学習の設定
    config = create_rl_config(cfg)
    trainer = config.build()

    num_iterations = 1000
    for i in range(num_iterations):
        result = trainer.train()

        # ログ出力
        print(f"Iteration {i}")
        print(f"Episode reward mean: {result['episode_reward_mean']}")
        print(f"Episode length mean: {result['episode_len_mean']}")

        # モデルの保存
        if i % 100 == 0:
            checkpoint_dir = trainer.save()
            print(f"Checkpoint saved at {checkpoint_dir}")


if __name__ == "__main__":
    main()

