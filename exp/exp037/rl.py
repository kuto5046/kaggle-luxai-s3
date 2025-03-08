import os
import random
import logging
from enum import IntEnum, auto
from time import time
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
from torch import nn
from lux.utils import (
    State,
    Action,
    GlobalState,
    EpisodeStore,
    extract_state,
    extract_global_state,
    get_valid_policy_map,
)
from lux.models import LuxUNetModel, LuxConvLSTMModel, LuxValueConvModel, LuxUNetModelInferenceWrapper
from lux.params import EnvParams
from luxai_s3.env import LuxAIS3Env
from scipy.signal import convolve2d
from luxai_s3.utils import to_numpy
from luxai_s3.params import env_params_ranges
from ray.tune.registry import register_env
from lux.imitation_agent import action_map_to_action
from ray.rllib.core.columns import Columns
from ray.rllib.utils.typing import ModuleID, TensorType, EpisodeType
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.utils.annotations import override
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.learner.learner import ENTROPY_KEY
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.algorithms.impala.impala import IMPALAConfig
from ray.rllib.connectors.module_to_env import (
    TensorToNumpy,
    ModuleToAgentUnmapping,
    ListifyDataForVectorEnv,
    UnBatchToIndividualItems,
    RemoveSingleTsTimeRankFromBatch,
)
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleSpec
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.algorithms.impala.impala_learner import IMPALALearner
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.models.torch.torch_distributions import (
    TorchCategorical,
    TorchDistribution,
)
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.algorithms.impala.torch.vtrace_torch_v2 import (
    vtrace_torch,
    make_time_major,
)

import wandb

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # multi-gpuでうまく動作できないためGPU0のみ利用可能に制限している

CPU_COUNT = os.cpu_count()

# policy名
OWN_POLICY = "p0"
BEST_POLICY = "best"
LB_BEST_POLICY = "lb_best"
SELF_PLAY_POLICY = "self-play"


class Model(IntEnum):
    UNet = auto()
    ConvLSTM = auto()


@dataclass
class Config:
    # common
    exp_name: str = Path(__file__).parent.name
    debug: bool = False
    notes: str = "unet cacheとflowの高速化を実施"
    env_name: str = "lux-s3-v0"
    root_dir: Path = Path("/home/kyohei.uto/kaggle-luxai-s3")
    # root_dir: Path = Path("/home/user/work")
    exp_dir: Path = root_dir / f"exp/{exp_name}"
    output_dir: Path = root_dir / f"output/{exp_name}"
    # pretrained model
    unet_n_stack: int = 4
    lstm_n_stack: int = 8
    num_layers: int = 3
    hidden_dim: int = 64
    kernel_size: int = 5
    num_repeats: int = 3
    freeze: bool = True
    overlap_penalty: float = 2.0
    stochastic: bool = True
    best_pretrained_path: Path | None = root_dir / "exp/rl_best/output/best_model.ckpt"
    lb_best_pretrained_path: Path | None = root_dir / "exp/lb_best/output/best_model.ckpt"

    num_cpus_per_learner: int = 1
    num_gpus_per_learner: int = 1
    num_cpus_per_env_runner: int = 1
    num_gpus_per_env_runner: int = 0

    # 以下の3つのrunnerにcpuとgpuを割り振る。cpuの合計値がcpu数を超えないように注意
    # 学習用
    # 　IMPALAの場合gpuが1つならlocal workerとして動かすためlearners=0が推奨される。
    # multi-gpuの場合はgpu数=learner数が本来は良いのだがうまく動作しない
    # そこで0を指定しlocal learnerとして動かし直接コードで学習時にcudaを指定するようにしている
    num_learners: int = 0
    # 評価用
    evaluation_num_env_runners: int = 25
    # データ収集用
    num_env_runners: int = 70

    # 学習設定
    training_minutes: int = 60 * 24  # 1日
    # workerからLearnerに送られるバッチのキューの最大サイズ. env_runner数と同じくらいが良いのではと思っている
    learner_queue_size: int = 100
    # 学習時に同じ時系列として扱いたいstep数を設定してやる。報酬が含まれるように1マッチ分の長さにする
    # batch_mode="truncate_episodes"の場合はmin(rollout_fragment_length, 101)stepごとにデータが送信される
    rollout_fragment_length: int | str | None = 101

    # 評価
    evaluation_interval: int = 30  # 何回trainをしたら評価を実施するか　１回が30secくらいなので50回で1500sec=25分くらい
    evaluation_duration: int = 50  # 1回の評価で何エピソード分評価するか
    # learner
    gamma: float = 0.9995
    lr: float = 1e-5
    train_batch_size_per_learner: int = 256
    num_epochs: int = 1  # 1回の学習のepoch数。新しいデータがどんどん追加されてくるためepoch数は1にしている
    replay_proportion: float = 0.0  # リプレイバッファの割合
    # loss
    vtrace_clip_rho_threshold: float = 1.0  # 価値関数のlossの係数
    vtrace_clip_pg_rho_threshold: float = 1.0  # ポリシー勾配のlossの係数
    vf_loss_coeff: float = 1.0  # 価値関数のlossの係数
    entropy_coeff: float = 1e-5  # エントロピーのlossの係数(大きくすると探索が活発になる)
    sap_loss_coeff: float = 1e-3  # sapのlossの係数
    # reward
    point_weight: float = 0  # マッチの報酬を超えないようにすべきなので適用する場合1e-3程度

    def __post_init__(self):
        if self.debug:
            self.num_env_runners = 1
            self.num_cpus_per_env_runner = 1
            self.evaluation_num_env_runners = 1
            self.evaluation_interval = 100
            self.evaluation_duration = 1
            self.training_minutes = 10
            self.train_batch_size_per_learner = 128
            self.learner_queue_size = 1
            self.num_epochs = 1


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
        self.overlap_penalty = config["overlap_penalty"]
        self.stochastic = config["stochastic"]
        self.point_weight = config["point_weight"]
        # アクション・観測空間の設定
        self.action_spaces = self._create_action_space()
        self.observation_spaces = self._create_obs_space()

        self.agents = self.possible_agents = ["player_0", "player_1"]
        self._agent_ids = set(self.agents)

        # reset時に更新
        self.state = None
        self.obs = None
        # reset時に更新
        self.reset(seed=0)

    def _set_params(self) -> EnvParams:
        randomized_game_params = {}
        for k, v in env_params_ranges.items():
            self.rng_key, subkey = jax.random.split(self.rng_key)
            randomized_game_params[k] = jax.random.choice(subkey, jnp.array(v)).item()
        return EnvParams(**randomized_game_params)

    def _create_action_space(self):
        """
        (24*24)の形状
        本来のpolicyは(num_actions, height, width)の形状だが行動空間は実際に取る行動を扱うため(height, width)の形状で扱う(rllibの仕様上)
        加えて2次元マップ(height, width)ではなく1次元マップ(height * width)のMultiDiscreteを使用(rllibの仕様上)

        sapアクションは0から1の連続値を持つ(h*w)の行動空間を追加
        """
        num_actions = len(Action)
        # 離散的な行動空間
        action_space = gym.spaces.MultiDiscrete([num_actions] * EnvParams.map_width * EnvParams.map_height)
        # sapアクション用の連続値行動空間
        sap_action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(EnvParams.map_width * EnvParams.map_height,), dtype=np.float32
        )
        # 複合的な行動空間
        combined_action_space = gym.spaces.Dict({"action": action_space, "sap": sap_action_space})
        return {
            "player_0": combined_action_space,
            "player_1": combined_action_space,
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
                "legal_action_mask": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(len(Action), EnvParams.map_height, EnvParams.map_width),
                    dtype=np.float32,
                ),
                "player_id": gym.spaces.Discrete(2),
                "opp_unit_map": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(EnvParams.map_height, EnvParams.map_width),
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
        self.obs, self.state = self.env.reset(reset_key, params=self.env_params)
        self.obs = to_numpy(flax.serialization.to_state_dict(self.obs))
        infos = {player_id: {} for player_id in self.obs.keys()}
        self.prev_actions = {
            "player_0": np.zeros((EnvParams.max_units, 3), dtype=np.int32),
            "player_1": np.zeros((EnvParams.max_units, 3), dtype=np.int32),
        }
        # 報酬計算用の前回の累積報酬を保存
        self.prev_raw_reward = {
            "player_0": 0,
            "player_1": 0,
        }

        self.episode_store1 = EpisodeStore(target_team_id=0, env_cfg=self.env_params)
        self.episode_store2 = EpisodeStore(target_team_id=1, env_cfg=self.env_params)
        # スタックの長さ分のバッファを用意
        self.agent0_states = deque(maxlen=self.n_stack)
        self.agent1_states = deque(maxlen=self.n_stack)
        self.agent0_global_states = deque(maxlen=self.n_stack)
        self.agent1_global_states = deque(maxlen=self.n_stack)
        # バッファを初期化
        for _ in range(self.n_stack):
            self.agent0_states.append(
                np.zeros((len(State), EnvParams.map_height, EnvParams.map_width), dtype=np.float32)
            )
            self.agent1_states.append(
                np.zeros((len(State), EnvParams.map_height, EnvParams.map_width), dtype=np.float32)
            )
            self.agent0_global_states.append(np.zeros((len(GlobalState),), dtype=np.float32))
            self.agent1_global_states.append(np.zeros((len(GlobalState),), dtype=np.float32))

        state = self._create_state(self.obs)

        return state, infos

    def _create_state(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        steps = obs["player_0"]["match_steps"]
        if steps == 0:
            self.episode_store1.reset()
            self.episode_store2.reset()
        else:
            self.episode_store1.update(obs["player_0"], self.prev_actions["player_0"])
            self.episode_store2.update(obs["player_1"], self.prev_actions["player_1"])

        agent0_state = extract_state(obs["player_0"], 0, self.episode_store1)
        agent1_state = extract_state(obs["player_1"], 1, self.episode_store2)
        # 自陣が(0, 0)になるようにstateを反転(state, height, width)
        agent1_state = np.flip(agent1_state, [1, 2])

        agent0_global_state = extract_global_state(obs["player_0"], 0, self.env_params, self.episode_store1)
        agent1_global_state = extract_global_state(obs["player_1"], 1, self.env_params, self.episode_store2)
        agent0_legal_action_mask = get_valid_policy_map(obs["player_0"], 0, self.episode_store1)
        agent1_legal_action_mask = get_valid_policy_map(obs["player_1"], 1, self.episode_store2)

        # 敵の観測データを使うことで完全な敵ユニット位置を推定できる
        player0_opp_unit_count = agent1_state[State.OWN_UNIT_COUNT] * EnvParams.max_units
        player1_opp_unit_count = agent0_state[State.OWN_UNIT_COUNT] * EnvParams.max_units
        # 1の周囲8マスに0.5を割り振るためのカーネル
        kernel = np.array([[0.3, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.3, 0.3]])

        # 畳み込みを実行（mode='same'で元の行列と同じサイズに）
        player0_opp_unit_map = convolve2d(player0_opp_unit_count, kernel, mode="same").astype(np.float32)
        player1_opp_unit_map = convolve2d(player1_opp_unit_count, kernel, mode="same").astype(np.float32)
        # 最大1になるように正規化 clipのほうがいいかもしれない
        # player0_opp_unit_map = np.clip(player0_opp_unit_map, 0, 1)
        # player1_opp_unit_map = np.clip(player1_opp_unit_map, 0, 1)
        if np.max(player0_opp_unit_map) > 1:
            player0_opp_unit_map = player0_opp_unit_map / np.max(player0_opp_unit_map)
        if np.max(player1_opp_unit_map) > 1:
            player1_opp_unit_map = player1_opp_unit_map / np.max(player1_opp_unit_map)

        self.agent0_states.append(agent0_state)
        self.agent1_states.append(agent1_state)
        self.agent0_global_states.append(agent0_global_state)
        self.agent1_global_states.append(agent1_global_state)
        return {
            "player_0": {
                "state": np.stack(list(self.agent0_states), axis=0),
                "global_state": np.stack(list(self.agent0_global_states), axis=0),
                "legal_action_mask": agent0_legal_action_mask,
                "player_id": 0,
                "opp_unit_map": player0_opp_unit_map,
            },
            "player_1": {
                "state": np.stack(list(self.agent1_states), axis=0),
                "global_state": np.stack(list(self.agent1_global_states), axis=0),
                "legal_action_mask": agent1_legal_action_mask,
                "player_id": 1,
                "opp_unit_map": player1_opp_unit_map,
            },
        }

    def _create_action(self, action_dict: dict[str, Any]) -> dict[str, np.ndarray]:
        actions = {agent_id: np.zeros((EnvParams.max_units, 3), dtype=np.int32) for agent_id in self.agents}

        # 1次元マップの行動空間で渡ってくるので2次元マップに変換
        action_map1 = action_dict["player_0"]["action"].reshape(EnvParams.map_height, EnvParams.map_width)
        action_map2 = action_dict["player_1"]["action"].reshape(EnvParams.map_height, EnvParams.map_width)

        # sapアクションも2次元マップに変換
        sap_map1 = action_dict["player_0"]["sap"].reshape(EnvParams.map_height, EnvParams.map_width)
        sap_map2 = action_dict["player_1"]["sap"].reshape(EnvParams.map_height, EnvParams.map_width)

        actions["player_0"] = action_map_to_action(
            action_map1,
            sap_map1,
            self.obs["player_0"],
            0,
            self.env_params,
            self.stochastic,
            self.overlap_penalty,
            self.episode_store1,
        )
        actions["player_1"] = action_map_to_action(
            action_map2,
            sap_map2,
            self.obs["player_1"],
            1,
            self.env_params,
            self.stochastic,
            self.overlap_penalty,
            self.episode_store2,
        )
        return actions

    def step(self, action_dict: dict[str, Any]) -> tuple:
        self.rng_key, step_key = jax.random.split(self.rng_key)
        actions = self._create_action(action_dict)
        self.prev_actions = actions
        self.obs, self.state, _reward, _terminated, _truncated, _ = self.env.step(
            step_key, self.state, actions, self.env_params
        )
        self.obs = to_numpy(flax.serialization.to_state_dict(self.obs))
        state = self._create_state(self.obs)
        player0_point = self.episode_store1.point
        player1_point = self.episode_store2.point

        terminated = {agent_id: done.item() for agent_id, done in _terminated.items()}
        truncated = {agent_id: done.item() for agent_id, done in _truncated.items()}
        # "__all__" (required) is used to indicate env termination.
        terminated["__all__"] = np.all(list(truncated.values()))  # luxaiはtruncatedがTrueになる
        info = {agent_id: {} for agent_id in self.agents}
        reward = self.reward_fn(_reward, player0_point, player1_point, self.point_weight)
        return state, reward, terminated, truncated, info

    def reward_fn(
        self, raw_reward: jnp.ndarray, player0_point: int, player1_point: int, point_weight: float = 0
    ) -> dict[str, int]:
        """
        raw_rewardは累積値なので、前回との差分を取って現在のステップでの報酬を計算する
        マッチごとに勝利したら1、敗北したら-1、引き分けは0
        """
        _reward = to_numpy(raw_reward)
        current_rewards = {agent_id: int(r.item()) for agent_id, r in _reward.items()}

        # 差分を計算して現在のステップでの報酬を取得
        step_rewards = {
            agent_id: current_rewards[agent_id] - self.prev_raw_reward[agent_id] for agent_id in current_rewards.keys()
        }

        # 0か+1の報酬しか発生しないので、プラスが発生したら反対のチームに負の報酬を与える
        for agent_id, reward in step_rewards.items():
            if reward > 0:
                opp_agent_id = self.agents[1 - self.agents.index(agent_id)]
                step_rewards[opp_agent_id] = -reward

        # 即時報酬としてstepの獲得ポイントを加算
        step_rewards["player_0"] += (player0_point - player1_point) * point_weight
        step_rewards["player_1"] += (player1_point - player0_point) * point_weight
        # 現在の累積報酬を保存
        self.prev_raw_reward = current_rewards

        return step_rewards


def freeze(model: nn.Module, model_name: Model):
    if model_name == Model.UNet:
        # 全てFalseにする
        for param in model.parameters():
            param.requires_grad = False

        # UNet後のpolicyネットワークのパラメータをTrueにする
        for param in model.sap_net1.parameters():
            param.requires_grad = True
        for param in model.sap_net2.parameters():
            param.requires_grad = True
        for param in model.sap_net3.parameters():
            param.requires_grad = True
        for param in model.policy_net1_from_sap.parameters():
            param.requires_grad = True
        for param in model.policy_net2.parameters():
            param.requires_grad = True
        for param in model.policy_net3.parameters():
            param.requires_grad = True
        for param in model.policy_net4.parameters():
            param.requires_grad = True

    elif model_name == Model.ConvLSTM:
        for param in model.inc.parameters():
            param.requires_grad = False
        for param in model.drc.parameters():
            param.requires_grad = False
        for param in model.policy_net.parameters():
            param.requires_grad = False
    else:
        raise ValueError(f"Invalid model name: {model_name}")


class LuxUnetTorchRLModule(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def setup(self):
        torch.set_num_threads(1)
        model_name = self.model_config["model_name"]
        self.n_stack = self.model_config["n_stack"]
        if model_name == Model.UNet:
            base_policy_model = LuxUNetModel(
                state_space_size=len(State),
                global_state_space_size=len(GlobalState),
                action_space_size=len(Action),
                n_stack=self.n_stack,
                res=True,
            )
        elif model_name == Model.ConvLSTM:
            assert False, "ConvLSTMはcache未対応"
            base_policy_model = LuxConvLSTMModel(
                state_space_size=len(State),
                global_state_space_size=len(GlobalState),
                action_space_size=len(Action),
                num_layers=self.model_config["num_layers"],
                hidden_dim=self.model_config["hidden_dim"],
                kernel_size=self.model_config["kernel_size"],
                num_repeats=self.model_config["num_repeats"],
            )
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        # 現在のデバイスを取得
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model_config["pretrained_path"]:
            # デバイスを明示的に指定してモデルをロード
            ckpt = torch.load(self.model_config["pretrained_path"], weights_only=False, map_location=self.device)
            state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
            base_policy_model.load_state_dict(state_dict)
            print(f"Loaded model from {self.model_config['pretrained_path']} on {self.device}")

        if self.model_config["freeze"]:
            freeze(base_policy_model, model_name)

        # モデルを明示的に同じデバイスに配置
        self.policy_model = LuxUNetModelInferenceWrapper(base_policy_model, self.n_stack)
        self.value_model = LuxValueConvModel(
            state_space_size=len(State),
            global_state_space_size=len(GlobalState),
            n_stack=self.n_stack,
        )
        self.value_model.to(self.device)
        self._values = None

    @override(TorchRLModule)
    def _forward(self, batch, is_train=False, **kwargs):
        # (batch, stack, ch, height, width)であり,stackはモデルによって異なる。大きめのstackで渡ってくるためモデルに合わせて変形する
        batch[Columns.OBS]["state"] = batch[Columns.OBS]["state"][:, -self.n_stack :, :, :, :].clone()
        batch[Columns.OBS]["global_state"] = batch[Columns.OBS]["global_state"][:, -self.n_stack :].clone()

        batch_size = batch[Columns.OBS]["state"].shape[0]
        outputs = self.policy_model(batch[Columns.OBS], is_train)
        policy_logits = outputs["policy"]
        sap_logits = outputs["sap"]
        opp_unit_map = batch[Columns.OBS]["opp_unit_map"]  # 反転処理は元々していないためここでも反転はしない
        sap_available_area = batch[Columns.OBS]["state"][:, -1:, State.SAP_AVAILABLE_AREA]
        player_id = batch[Columns.OBS]["player_id"]

        # player_id1のポリシーを反転して自陣を復元する(自陣固定の後処理)
        player_1_mask = (player_id == 1).view(-1, 1, 1, 1)  # バッチ次元に合わせてブロードキャスト可能な形に変換
        flipped_policy_logits = torch.flip(policy_logits, [2, 3]).clone()
        flipped_policy_logits[:, Action.DOWN, :, :], flipped_policy_logits[:, Action.UP, :, :] = (
            flipped_policy_logits[:, Action.UP, :, :].clone(),
            flipped_policy_logits[:, Action.DOWN, :, :].clone(),
        )
        flipped_policy_logits[:, Action.RIGHT, :, :], flipped_policy_logits[:, Action.LEFT, :, :] = (
            flipped_policy_logits[:, Action.LEFT, :, :].clone(),
            flipped_policy_logits[:, Action.RIGHT, :, :].clone(),
        )
        policy_logits = torch.where(player_1_mask, flipped_policy_logits, policy_logits)
        # sapも復元
        flipped_sap_logits = torch.flip(sap_logits, [2, 3]).clone()
        flipped_sap_available_area = torch.flip(sap_available_area, [2, 3]).clone()
        sap_logits = torch.where(player_1_mask, flipped_sap_logits, sap_logits)
        sap_available_area = torch.where(player_1_mask, flipped_sap_available_area, sap_available_area)
        sap_logits = sap_logits - (1 - sap_available_area) * 1e32
        sap_logits = sap_logits.reshape(batch_size, -1)  # (batch, height * width)
        opp_unit_map = opp_unit_map.reshape(batch_size, -1)  # (batch, height * width)
        # batch方向に1つ手前にずらすことで次のstepの敵ユニット位置をtargetとする (sap_targets[0, :] == opp_unit_map[1, :]という関係)
        # rollout_fragment_lengthが101なので連続してる想定だが101stepは連続している。
        # rolloutの境界ではtargetがズレるのでloss計算から除外する処理を後段で行う
        sap_targets = torch.roll(opp_unit_map, shifts=-1, dims=0)

        num_actions = policy_logits.shape[1]
        # stateは(batch, stack, ch, height, width)なので最新のunit位置を以下のように取得(batch, height, width)
        unit_mask = batch[Columns.OBS]["state"][:, -1, State.OWN_UNIT_COUNT] > 0
        action_mask = batch[Columns.OBS]["legal_action_mask"]
        # 無効な行動(action_mask=0)は負の大きな値になるためsoftmax後は0になる。
        masked_policy_logits = policy_logits - 1e32 * (1 - action_mask)
        # この時点では(batch, action, height, width)なので(batch, height, width, action)に変換
        masked_policy_logits = masked_policy_logits.reshape(batch_size, num_actions, -1).transpose(2, 1)
        unit_mask = unit_mask.reshape(batch_size, -1)
        return {
            Columns.ACTION_DIST_INPUTS: masked_policy_logits,
            # unit位置のみpolicyを学習する
            "unit_mask": unit_mask,
            # 行動に利用されるsapの確率
            "sap": torch.sigmoid(sap_logits),
            # 以下はsapの学習に利用する.
            "sap_logits": sap_logits,
            "sap_targets": sap_targets,
            "sap_available_area": sap_available_area.reshape(batch_size, -1),
        }

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        return self._forward(batch, is_train=True, **kwargs)

    @override(TorchRLModule)
    def _forward_inference(self, batch, **kwargs):
        # 各試合の1step目の場合cacheをreset
        if batch[Columns.OBS]["global_state"][GlobalState.MATCH_STEPS] == 0:
            self.policy_model.reset()
        return self._forward(batch, is_train=False, **kwargs)

    @override(ValueFunctionAPI)
    def compute_values(self, batch: dict[str, Any], embeddings: Any | None = None) -> torch.Tensor:
        self.value_model.to("cuda")
        for key in batch[Columns.OBS]:
            if isinstance(batch[Columns.OBS][key], torch.Tensor):
                batch[Columns.OBS][key] = batch[Columns.OBS][key].to("cuda")

        outputs = self.value_model(batch[Columns.OBS])
        self._values = outputs["value"].squeeze(dim=1)
        return self._values

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self) -> type[TorchDistribution]:
        return TorchCategorical


@ray.remote
class EpisodeStatsCollector:
    def __init__(self):
        self.runner_episode_end_times = deque(maxlen=100)  # 直近100エピソードの終了時間を保存して速度を計測する
        self.eval_episode_end_times = deque(maxlen=100)  # 直近100エピソードの終了時間を保存して速度を計測する
        self.runner_total_episodes = 0
        self.eval_total_episodes = 0
        self.last_log_time = time()
        # 評価用の変数
        self.evaluation_wins = []
        self.current_evaluation_id = 0
        self.is_evaluation_active = False
        self.best_win_rate = 0

    def add_episode(self, in_evaluation: bool):
        if in_evaluation:
            self.eval_episode_end_times.append(time())
            self.eval_total_episodes += 1
        else:
            self.runner_episode_end_times.append(time())
            self.runner_total_episodes += 1

    def start_evaluation(self):
        """評価開始時に呼び出す"""
        self.current_evaluation_id += 1
        self.evaluation_wins = []
        self.eval_total_episodes = 0
        return self.current_evaluation_id

    def record_evaluation_result(self, is_win):
        """評価エピソードの結果を記録"""
        self.evaluation_wins.append(is_win)

    def get_evaluation_stats(self):
        """現在の評価統計を取得"""
        wins = sum(self.evaluation_wins) if self.evaluation_wins else 0
        total = len(self.evaluation_wins)
        win_rate = wins / total if total > 0 else 0
        return {
            "wins": wins,
            "total": total,
            "win_rate": win_rate,
        }

    def get_best_win_rate(self):
        return self.best_win_rate

    def update_best_win_rate(self, win_rate: float):
        self.best_win_rate = win_rate

    def get_speed_stats(self, in_evaluation: bool):
        if in_evaluation:
            episode_end_times = self.eval_episode_end_times
            total_episodes = self.eval_total_episodes
        else:
            episode_end_times = self.runner_episode_end_times
            total_episodes = self.runner_total_episodes

        if len(episode_end_times) < 2:
            return {
                "episode_per_minute": 0.0,
                "total_episodes": total_episodes,
            }

        # queueに溜まっているepisode終了時間の差分を計算
        window_duration = episode_end_times[-1] - episode_end_times[0]
        if window_duration == 0:
            return {
                "episode_per_minute": 0.0,
                "total_episodes": total_episodes,
            }

        # 1秒間に何エピソード終了したか
        episode_per_sec = (len(episode_end_times) - 1) / window_duration
        episode_per_minute = episode_per_sec * 60
        return {
            "episode_per_minute": episode_per_minute,
            "total_episodes": total_episodes,
        }


def setup_logger(output_dir: Path):
    """ロガーの設定"""
    # ログディレクトリの作成
    log_dir = output_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    # ロガーの設定
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 既存のハンドラをクリア（重複を避けるため）
    if logger.handlers:
        logger.handlers.clear()

    # ファイルハンドラの設定
    log_file = log_dir / "result.log"
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(logging.INFO)

    # コンソールハンドラの設定
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # フォーマッタの設定
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # ハンドラの追加
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class WandbLoggerCallback(RLlibCallback):
    def __init__(self):
        self.output_dir = Config.output_dir
        # グローバルな統計コレクターを作成（一度だけ）
        if not hasattr(WandbLoggerCallback, "_stats_collector"):
            WandbLoggerCallback._stats_collector = EpisodeStatsCollector.remote()

        # 各ワーカープロセス用にロガーを初期化
        self.logger = setup_logger(self.output_dir)

    # 学習状況をwandbに流す用
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

        # 学習状況をwandbに流す用
        time_metrics = [
            "time_this_iter_s",  # 1回の学習時間
            # 1回の学習イテレーションで
            "mean_num_episode_lists_received",
        ]
        for key in time_metrics:
            wandb.log({f"train/{key}": result[key]})

        # 学習データのサンプリング時間
        if result.get("env_runners"):
            if result["env_runners"].get("time_between_sampling"):
                wandb.log(
                    {
                        "train/env_runner_time_between_sampling": result["env_runners"]["time_between_sampling"],
                    }
                )

        # learnersがない場合はskip(並列で実行しているため最初はないはず)
        if result.get("learners"):
            # learner_metrics = result["learners"][OWN_POLICY].keys()
            learner_metrics = [
                # "num_non_trainable_parameters",  # 一定
                "gradients_default_optimizer_global_norm",
                "diff_num_grad_updates_vs_sampler_policy",
                # "module_train_batch_size_mean",  # 一定
                # "pi_loss",  # mean_pi_lossと同じ
                "num_module_steps_trained_lifetime",  # これが学習したstep数
                # "weights_seq_no",  # 一定
                "total_loss",
                # "default_optimizer_learning_rate",  # 一定
                "mean_pi_loss",
                "num_module_steps_trained",
                "mean_vf_loss",
                # "num_trainable_parameters",  # 一定
                # "curr_entropy_coeff",  # 一定
                # "vf_loss",  # mean_vf_lossと同じ
                "entropy",
                "sap_loss",
            ]
            for key in learner_metrics:
                wandb.log(
                    {
                        f"train/{key}": result["learners"][OWN_POLICY][key],
                    }
                )

            # 学習した総エピソード数
            trained_episodes_lifetime = result["learners"][OWN_POLICY]["num_module_steps_trained_lifetime"] // 505
            # 1分あたりの学習エピソード数
            trained_episodes_per_minute = (trained_episodes_lifetime / result["time_total_s"]) * 60

            wandb.log(
                {
                    "train/trained_episode_lifetime": trained_episodes_lifetime,
                    "train/trained_episodes_per_minute": trained_episodes_per_minute,
                }
            )

    # 学習したモデルの性能評価をwandbに流す用
    def on_evaluate_start(
        self,
        *,
        algorithm: "Algorithm",
        metrics_logger: MetricsLogger | None = None,
        **kwargs,
    ) -> None:
        """Called at the beginning of Algorithm.evaluate()."""
        # 中央の評価トラッカーに評価開始を通知
        self._current_evaluation_id = ray.get(self._stats_collector.start_evaluation.remote())
        self.logger.info(f"Evaluation {self._current_evaluation_id} started")

    def on_evaluate_end(
        self,
        *,
        algorithm: "Algorithm",
        metrics_logger: MetricsLogger | None = None,
        evaluation_metrics: dict,
        **kwargs,
    ) -> None:
        """Runs when the evaluation is done."""

        if not evaluation_metrics.get("env_runners"):
            return

        # 対戦相手のポリシー名
        policy_names = list(evaluation_metrics["env_runners"]["module_episode_returns_mean"].keys())
        self.logger.info(f"evaluation policy names: {policy_names}")

        mean_rewards = evaluation_metrics["env_runners"]["module_episode_returns_mean"][OWN_POLICY]
        evaluation_minutes = evaluation_metrics["env_runners"]["env_to_module_sum_episodes_length_in"] / 60

        # 中央の評価トラッカーから評価結果を取得
        eval_stats = ray.get(self._stats_collector.get_evaluation_stats.remote())
        current_win_rate = eval_stats["win_rate"]
        # wandbに記録
        wandb.log(
            {
                "evaluate/mean_rewards": mean_rewards,
                "evaluate/evaluation_minutes": evaluation_minutes,
                "evaluate/win_rate": current_win_rate,
                "evaluate/num_episodes": eval_stats["total"],
            }
        )

        self.logger.info(f"Evaluation {self._current_evaluation_id} completed episodes={eval_stats['total']}")

        # 評価結果をリセット
        save_model(algorithm, self.output_dir, suffix=f"model_eval_{self._current_evaluation_id}")
        save_model(algorithm, self.output_dir, suffix="latest_model")
        best_win_rate = ray.get(self._stats_collector.get_best_win_rate.remote())
        if best_win_rate < current_win_rate:
            save_model(algorithm, self.output_dir, suffix="best_model")
            self.logger.info(f"Best win rate updated. {best_win_rate=:.4f} -> {current_win_rate=:.4f}")
            # ベスト勝率を更新
            ray.get(self._stats_collector.update_best_win_rate.remote(current_win_rate))

    @override(RLlibCallback)
    def on_episode_end(
        self,
        *,
        episode: EpisodeType,
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: MetricsLogger | None = None,
        **kwargs,
    ) -> None:
        # rolloutのepisodeがちゃんと集計されているか怪しい
        in_evaluation = env_runner.config.in_evaluation
        # エピソード完了を記録
        ray.get(self._stats_collector.add_episode.remote(in_evaluation))

        episode_rewards = episode.get_rewards()
        episode_total_reward = {k: sum(v) for k, v in episode_rewards.items()}
        is_win = (episode_total_reward["player_0"] > episode_total_reward["player_1"]) * 1
        # duration = episode.get_duration()
        episode_duration_s = episode.get_duration_s()
        if in_evaluation:
            ray.get(self._stats_collector.record_evaluation_result.remote(is_win))
            stats = ray.get(self._stats_collector.get_speed_stats.remote(in_evaluation))
            self.logger.info(
                f"Evaluation Episode {stats['total_episodes']} finished. {is_win=} {episode_total_reward=} Episode duration: {episode_duration_s:.2f} sec"
            )
        else:
            stats = ray.get(self._stats_collector.get_speed_stats.remote(in_evaluation))
            self.logger.info(
                f"Env Runner Episode {stats['total_episodes']} finished. {is_win=} {episode_total_reward=} Truncated Episode duration: {episode_duration_s:.2f} sec"
            )


class CustomIMPALATorchLearner(IMPALALearner, TorchLearner):
    """Implements the IMPALA loss function in torch."""

    def apply_device(self, batch: dict, fwd_out: dict, device: str):
        # すべての入力テンソルを同じデバイスに移動
        for key, value in batch.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        batch[key][sub_key] = sub_value.to(device)
            elif isinstance(value, torch.Tensor):
                batch[key] = value.to(device)

        # fwd_outのテンソルも同じデバイスに移動
        for key, value in fwd_out.items():
            if isinstance(value, torch.Tensor):
                fwd_out[key] = value.to(device)

    @override(TorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: IMPALAConfig,
        batch: dict,
        fwd_out: dict[str, TensorType],
    ) -> TensorType:
        module = self.module[module_id].unwrapped()
        # start_time = time()

        # multi-gpuだと異なるgpuのデータが混ざる？のでデバイスを揃える
        self.apply_device(batch, fwd_out, "cuda")
        # TODO (sven): Now that we do the +1ts trick to be less vulnerable about
        #  bootstrap values at the end of rollouts in the new stack, we might make
        #  this a more flexible, configurable parameter for users, e.g.
        #  `v_trace_seq_len` (independent of `rollout_fragment_length`). Separation
        #  of concerns (sampling vs learning).
        rollout_frag_or_episode_len = config.get_rollout_fragment_length()
        recurrent_seq_len = batch.get("seq_lens")

        loss_mask = fwd_out["unit_mask"].float()
        size_loss_mask = torch.sum(loss_mask)

        # Behavior actions logp and target actions logp.
        behaviour_actions_logp = batch[Columns.ACTION_LOGP]
        target_policy_dist = module.get_train_action_dist_cls().from_logits(fwd_out[Columns.ACTION_DIST_INPUTS])
        target_actions_logp = target_policy_dist.logp(batch[Columns.ACTIONS]["action"])
        # loss_maskを適用した上でマップの次元を潰す
        # (batch_size, 24*24)のマップ状態をもつデータをunit位置のmaskを適用した上で(batch_size, 1)に潰す
        behaviour_actions_logp = (behaviour_actions_logp * loss_mask).sum(dim=1)
        target_actions_logp = (target_actions_logp * loss_mask).sum(dim=1)

        # Values and bootstrap values.
        values = module.compute_values(batch, embeddings=fwd_out.get(Columns.EMBEDDINGS))
        values_time_major = make_time_major(
            values,
            trajectory_len=rollout_frag_or_episode_len,
            recurrent_seq_len=recurrent_seq_len,
        )
        assert Columns.VALUES_BOOTSTRAPPED not in batch
        # Use as bootstrap values the vf-preds in the next "batch row", except
        # for the very last row (which doesn't have a next row), for which the
        # bootstrap value does not matter b/c it has a +1ts value at its end
        # anyways. So we chose an arbitrary item (for simplicity of not having to
        # move new data to the device).
        bootstrap_values = torch.cat(
            [
                values_time_major[0][1:],  # 0th ts values from "next row"
                values_time_major[0][0:1],  # <- can use any arbitrary value here
            ],
            dim=0,
        )

        # TODO(Artur): In the old impala code, actions were unsqueezed if they were
        #  multi_discrete. Find out why and if we need to do the same here.
        #  actions = actions if is_multidiscrete else torch.unsqueeze(actions, dim=1)
        target_actions_logp_time_major = make_time_major(
            target_actions_logp,
            trajectory_len=rollout_frag_or_episode_len,
            recurrent_seq_len=recurrent_seq_len,
        )
        behaviour_actions_logp_time_major = make_time_major(
            behaviour_actions_logp,
            trajectory_len=rollout_frag_or_episode_len,
            recurrent_seq_len=recurrent_seq_len,
        )
        rewards_time_major = make_time_major(
            batch[Columns.REWARDS],
            trajectory_len=rollout_frag_or_episode_len,
            recurrent_seq_len=recurrent_seq_len,
        )

        # the discount factor that is used should be gamma except for timesteps where
        # the episode is terminated. In that case, the discount factor should be 0.
        discounts_time_major = (
            1.0
            - make_time_major(
                batch[Columns.TERMINATEDS],
                trajectory_len=rollout_frag_or_episode_len,
                recurrent_seq_len=recurrent_seq_len,
            ).type(dtype=torch.float32)
        ) * config.gamma

        # Note that vtrace will compute the main loop on the CPU for better performance.
        vtrace_adjusted_target_values, pg_advantages = vtrace_torch(
            target_action_log_probs=target_actions_logp_time_major,
            behaviour_action_log_probs=behaviour_actions_logp_time_major,
            discounts=discounts_time_major,
            rewards=rewards_time_major,
            values=values_time_major,
            bootstrap_values=bootstrap_values,
            clip_rho_threshold=config.vtrace_clip_rho_threshold,
            clip_pg_rho_threshold=config.vtrace_clip_pg_rho_threshold,
        )

        # The policy gradients loss.
        pi_loss = -torch.sum(target_actions_logp_time_major * pg_advantages)

        # size_loss_maskで割ることで1step-1ユニットあたりのlossになる
        mean_pi_loss = pi_loss / size_loss_mask

        # The baseline loss.
        delta = values_time_major - vtrace_adjusted_target_values
        vf_loss = 0.5 * torch.sum(torch.pow(delta, 2.0))
        mean_vf_loss = vf_loss / size_loss_mask

        # The entropy loss.
        entropy_loss = -torch.sum(target_policy_dist.entropy() * loss_mask)
        mean_entropy_loss = entropy_loss / size_loss_mask

        # SAPの教師あり学習
        sap_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            fwd_out["sap_logits"], fwd_out["sap_targets"], reduction="none"
        )
        sap_available_area = fwd_out["sap_available_area"]
        # sap範囲外は学習しない
        sap_loss[sap_available_area == 0] = 0
        sap_loss = sap_loss.sum(dim=1)

        # 同じ時系列で扱うべきなのでtime_majorをする
        sap_loss_time_major = make_time_major(
            sap_loss,
            trajectory_len=rollout_frag_or_episode_len,
            recurrent_seq_len=recurrent_seq_len,
        )
        # 時系列方向の最後のステップはtargetがおかしくなるので学習しない
        sap_loss_time_major[-1, :] = 0
        mean_sap_loss = sap_loss_time_major.sum() / sap_available_area.sum() * config.sap_loss_coeff

        # The summed weighted loss.
        total_loss = (
            mean_pi_loss
            + mean_vf_loss * config.vf_loss_coeff
            + (mean_entropy_loss * self.entropy_coeff_schedulers_per_module[module_id].get_current_value())
            + mean_sap_loss
        )

        # Log important loss stats.
        self.metrics.log_dict(
            {
                "pi_loss": pi_loss,
                "mean_pi_loss": mean_pi_loss,
                "vf_loss": vf_loss,
                "mean_vf_loss": mean_vf_loss,
                ENTROPY_KEY: -mean_entropy_loss,
                "sap_loss": mean_sap_loss,
            },
            key=module_id,
            window=1,  # <- single items (should not be mean/ema-reduced over time).
        )
        # Return the total loss.
        # device = fwd_out["unit_mask"].device
        # batch_size = fwd_out["unit_mask"].shape[0]
        # print(f"time: {time() - start_time:.2f} sec {batch_size=} {device=} {module_id=}")
        return total_loss


class CustomGetActions(ConnectorV2):
    @override(ConnectorV2)
    def __call__(
        self,
        *,
        rl_module: RLModule,
        batch: dict[str, Any],
        episodes: list[EpisodeType],
        explore: bool | None = None,
        shared_data: dict | None = None,
        **kwargs,
    ) -> Any:
        is_multi_agent = isinstance(episodes[0], MultiAgentEpisode)

        if is_multi_agent:
            for module_id, module_data in batch.copy().items():
                self._get_actions(module_data, rl_module[module_id], explore)
        else:
            self._get_actions(batch, rl_module, explore)

        return batch

    def _get_actions(self, batch, sa_rl_module, explore):
        # Action have already been sampled -> Early out.
        if Columns.ACTIONS in batch:
            return

        # ACTION_DIST_INPUTS field returned by `forward_exploration|inference()` ->
        # Create a new action distribution object.
        if Columns.ACTION_DIST_INPUTS in batch:
            if explore:
                action_dist_class = sa_rl_module.get_exploration_action_dist_cls()
            else:
                action_dist_class = sa_rl_module.get_inference_action_dist_cls()
            action_dist = action_dist_class.from_logits(
                batch[Columns.ACTION_DIST_INPUTS],
            )
            if not explore:
                action_dist = action_dist.to_deterministic()

            # Sample actions from the distribution.
            actions = action_dist.sample()
            batch[Columns.ACTIONS] = {
                "action": actions,
                "sap": batch["sap"],
            }

            # For convenience and if possible, compute action logp from distribution
            # and add to output.
            if Columns.ACTION_LOGP not in batch:
                batch[Columns.ACTION_LOGP] = action_dist.logp(actions)


def custom_module_to_env_connector(env: MultiAgentEnv) -> list[ConnectorV2]:
    return [
        # GetActions(),
        CustomGetActions(),
        TensorToNumpy(),
        UnBatchToIndividualItems(),
        ModuleToAgentUnmapping(),
        RemoveSingleTsTimeRankFromBatch(),
        # NormalizeAndClipActions(),
        ListifyDataForVectorEnv(),
    ]


def create_rl_config(cfg: Config) -> AlgorithmConfig:
    env_config = {
        # 環境では大きめにstackを作成しておきモデル側で必要なstackを抽出する
        "n_stack": max(cfg.unet_n_stack, cfg.lstm_n_stack),
        "stochastic": cfg.stochastic,
        "overlap_penalty": cfg.overlap_penalty,
        "point_weight": cfg.point_weight,
    }
    tmp_env = env_creator(env_config)
    observation_space = tmp_env.get_observation_space("player_0")
    action_space = tmp_env.get_action_space("player_0")
    best_rl_module_spec = RLModuleSpec(
        module_class=LuxUnetTorchRLModule,
        observation_space=observation_space,
        action_space=action_space,
        # モデル内部でself.model_config["key"]でアクセスできる
        model_config={
            "n_stack": cfg.unet_n_stack,
            "pretrained_path": cfg.best_pretrained_path,
            "freeze": cfg.freeze,
            "model_name": Model.UNet,
        },
    )
    lb_best_rl_module_spec = RLModuleSpec(
        module_class=LuxUnetTorchRLModule,
        observation_space=observation_space,
        action_space=action_space,
        model_config={
            "n_stack": cfg.lstm_n_stack,
            "pretrained_path": cfg.lb_best_pretrained_path,
            "freeze": cfg.freeze,
            "model_name": Model.ConvLSTM,
            "num_layers": cfg.num_layers,
            "hidden_dim": cfg.hidden_dim,
            "kernel_size": cfg.kernel_size,
            "num_repeats": cfg.num_repeats,
        },
    )
    config = (
        IMPALAConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        # 環境設定
        .environment(env=cfg.env_name, env_config=env_config)
        # ゲームをしてデータを生成するrunnerの数. cpuの数と合わせる
        .env_runners(
            num_env_runners=cfg.num_env_runners,
            # num_envs_per_env_runner=cfg.num_envs_per_env_runner,  # multi agentはenv vectorizationが未対応
            num_cpus_per_env_runner=cfg.num_cpus_per_env_runner,
            num_gpus_per_env_runner=cfg.num_gpus_per_env_runner,
            # デフォルト値。101step（truncated=True)のタイミングでデータを収集する.
            batch_mode="truncate_episodes",
            # batch_sizeから自動で適切な値を計算してくれるためこの設定が推奨されている
            # rollout_fragment_length = "auto",
            rollout_fragment_length=cfg.rollout_fragment_length,
            # module -> envの操作をカスタム実装したいためdefaultはoffにしている
            add_default_connectors_to_module_to_env_pipeline=False,
            module_to_env_connector=custom_module_to_env_connector,
            # 環境を作成した後に環境を検証する
            # validate_env_runners_after_construction=True,
        )
        # モデルを学習するlearnerの数。gpuの数と合わせる
        # Can't set both `num_cpus_per_learner` > 1 and  `num_gpus_per_learner` > 0! Either set `num_cpus_per_learner` > 1 (and `num_gpus_per_learner`=0)
        # OR set `num_gpus_per_learner` > 0 (and leave `num_cpus_per_learner` at its default value of 1). This is due to issues with placement group fragmentation.
        # See https://github.com/ray-project/ray/issues/35409 for more details.
        .learners(
            num_learners=cfg.num_learners,
            num_cpus_per_learner=cfg.num_cpus_per_learner,
            num_gpus_per_learner=cfg.num_gpus_per_learner,
        )
        # 学習パラメータ設定
        .training(
            learner_class=CustomIMPALATorchLearner,
            # 一般的な学習の設定
            opt_type="adam",
            gamma=cfg.gamma,
            lr=cfg.lr,
            num_epochs=cfg.num_epochs,
            # learnerの設定
            train_batch_size_per_learner=cfg.train_batch_size_per_learner,
            learner_queue_size=cfg.learner_queue_size,
            replay_proportion=cfg.replay_proportion,
            # loss
            vtrace=True,
            vtrace_clip_rho_threshold=cfg.vtrace_clip_rho_threshold,
            vtrace_clip_pg_rho_threshold=cfg.vtrace_clip_pg_rho_threshold,
            vf_loss_coeff=cfg.vf_loss_coeff,
            entropy_coeff=cfg.entropy_coeff,
        )
        # マルチエージェント設定
        # https://github.com/ray-project/ray/blob/2a85cef1ad8105d8dda01d709da7b0eaeb337caa/rllib/examples/multi_agent/rock_paper_scissors_heuristic_vs_learned.py#L94
        # https://github.com/ray-project/ray/blob/2a85cef1ad8105d8dda01d709da7b0eaeb337caa/rllib/examples/multi_agent/rock_paper_scissors_learned_vs_learned.py#L65
        .multi_agent(
            # RLで扱うagent(policy)の名前
            policies={
                OWN_POLICY,
                SELF_PLAY_POLICY,
                BEST_POLICY,
                # LB_BEST_POLICY,  # cpuで動かすと遅すぎるので現在は使用していない TODO: onnx変換試す
            },
            # 各agentのポリシーを決める関数
            policy_mapping_fn=lambda aid, episode, **kwargs: (
                OWN_POLICY
                if aid == "player_0"
                else random.choice(
                    [
                        SELF_PLAY_POLICY,
                        BEST_POLICY,
                        # LB_BEST_POLICY,
                    ]
                )
            ),
            # 学習は自身のpolicyとself-playのpolicyを学習
            policies_to_train=[OWN_POLICY, SELF_PLAY_POLICY],
        )
        # https://docs.ray.io/en/latest/rllib/rllib-rlmodule.html#construction-through-rlmodulespecs
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                # policy名とモデルの紐づけ
                rl_module_specs={
                    OWN_POLICY: best_rl_module_spec,
                    SELF_PLAY_POLICY: best_rl_module_spec,
                    BEST_POLICY: best_rl_module_spec,
                    # LB_BEST_POLICY: lb_best_rl_module_spec,
                }
            )
        )
        .framework(
            framework="torch",
            eager_tracing=True,
        )
        .callbacks(WandbLoggerCallback)
        .evaluation(
            evaluation_num_env_runners=cfg.evaluation_num_env_runners,
            evaluation_interval=cfg.evaluation_interval,
            evaluation_duration=cfg.evaluation_duration,
            evaluation_duration_unit="episodes",
            evaluation_sample_timeout_s=60 * 20,
            evaluation_force_reset_envs_before_iteration=True,  # 各評価の前に環境をリセット
            evaluation_parallel_to_training=True,  # 評価と学習を並列に実行
            # # 評価用の上書き設定.これにより評価時はlb_bestポリシーと自身の対戦になる
            evaluation_config={
                "multiagent": {
                    "policy_mapping_fn": lambda aid, episode, **kwargs: (
                        OWN_POLICY if aid == "player_0" else BEST_POLICY
                    ),
                }
            },
        )
        # .checkpointing(
        #     export_native_model_files=True,
        # )
    )
    # あまり良くなさそうだが参照しやすいようにここに係数を追加しておく
    config.sap_loss_coeff = cfg.sap_loss_coeff
    return config


def setup_wandb(cfg: Config):
    wandb.init(
        project="kaggle-luxai-s3",
        entity="kuto5046",
        group=cfg.exp_name,
        notes=cfg.notes,
        mode="disabled" if cfg.debug else "online",
    )


def save_model(trainer: Algorithm, output_dir: Path, suffix: str = "model"):
    """
    rllibのapiを使わず直接モデルを保存する
    モデルの名前はrlmoduleで定義した名前を使う
    """
    rl_module = trainer.get_module(OWN_POLICY)

    # wrapしている場合はmodelを取り出す
    if hasattr(rl_module.policy_model, "model"):
        policy_state_dict = rl_module.policy_model.model
    else:
        policy_state_dict = rl_module.policy_model

    value_state_dict = rl_module.value_model

    torch.save(policy_state_dict, output_dir / f"policy_{suffix}.pth")
    torch.save(value_state_dict, output_dir / f"value_{suffix}.pth")


def main() -> None:
    cfg = Config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # ロガーのセットアップ
    logger = setup_logger(cfg.output_dir)
    logger.info(f"{cfg.num_env_runners=} {cfg.num_learners=} {cfg.evaluation_num_env_runners=}")

    setup_wandb(cfg)
    ray.init()

    # 環境の登録
    register_env(name=cfg.env_name, env_creator=env_creator)

    config = create_rl_config(cfg)
    logger.info(f"{config.get_rollout_fragment_length()=}")
    trainer = config.build_algo(env=cfg.env_name)

    train_count = 0
    total_train_start_time = time()
    while True:
        train_start_time = time()
        # データが溜まっていない場合処理は完了するが学習は未実施となる
        result = trainer.train()
        # logger.info(f"train result: {result}")
        train_count += 1
        spend_minutes = (time() - train_start_time) / 60
        total_spend_minutes = (time() - total_train_start_time) / 60
        logger.info(f"Training iteration {train_count} finished. Spent {spend_minutes:.1f} minutes")
        # 指定した時間経ったら学習を終了
        if total_spend_minutes > cfg.training_minutes:
            logger.info(f"Training completed after {spend_minutes:.1f} minutes")
            break

    save_model(trainer, cfg.output_dir, suffix="latest_model")


if __name__ == "__main__":
    main()
