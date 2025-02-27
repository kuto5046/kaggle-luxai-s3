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
    HiddenState,
    EpisodeStore,
    in_map,
    extract_state,
    extract_global_state,
    get_valid_policy_map,
)
from lux.models import Down, DoubleConv, LuxUNetModel
from lux.params import EnvParams
from luxai_s3.env import LuxAIS3Env
from luxai_s3.utils import to_numpy
from luxai_s3.params import env_params_ranges
from ray.tune.registry import register_env
from ray.rllib.core.columns import Columns
from ray.rllib.utils.typing import ModuleID, TensorType, EpisodeType
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.utils.annotations import override
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.learner.learner import ENTROPY_KEY
from ray.rllib.algorithms.impala.impala import IMPALAConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.algorithms.impala.impala_learner import IMPALALearner
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.models.torch.torch_distributions import TorchCategorical, TorchDistribution
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.algorithms.impala.torch.vtrace_torch_v2 import (
    vtrace_torch,
    make_time_major,
)

import wandb


@dataclass
class Config:
    exp_name: str = Path(__file__).parent.name
    notes: str = "rlをrayで動かす"
    model_name: str = "lux_unet"
    env_name: str = "lux-s3-v0"
    n_stack: int = 4
    root_dir: Path = Path(f"/home/user/work/exp/{exp_name}")
    pretrained_path: Path | None = None  # root_dir / "output/best_model.ckpt"
    debug: bool = True
    output_dir: Path = root_dir / "output"

    # 以下の3つのrunnerにcpuとgpuを割り振る。cpuの合計値がcpu数を超えないように注意
    # データ収集用
    num_env_runners: int = 20  # actorの数
    num_cpus_per_env_runner: int = 1
    rollout_fragment_length: int = 1  # 時系列を特に考えない場合1

    # 学習用(GPUの数=learnerと考えて良い)
    num_learners: int = 0  # 0の場合local learnerを使用することを意味する(learnner=1)
    num_cpus_per_learner: int = 1
    num_gpus_per_learner: int = 1

    # 評価用
    evaluation_num_env_runners: int = 2  # 評価用のenv runnerの数
    evaluation_interval: int = 1  # 何回trainをしたら評価を実施するか
    evaluation_duration: int = 4  # 1回の評価で何エピソード分評価するか

    # learner
    training_minutes: int = 1
    learner_queue_size: int = 20  # workerからLearnerに送られるバッチのキューの最大サイズ. [batch_size]*queue_sizeがcpuメモリに乗りbatchごとに学習する
    gamma: float = 0.99
    lr: float = 1e-4
    # batch size
    train_batch_size_per_learner: int = (
        505  # 一応1episodeのサイズにしてるが不要かも。もしくはrollout_fragment_length部分で調整する
    )
    # 1回の学習データ(train_batch_size*queue_size)を何epoch分学習するか
    num_epochs: int = 1

    # loss
    vf_loss_coeff: float = 1.0  # 価値関数のlossの係数
    entropy_coeff: float = 1.0  # エントロピーのlossの係数(大きくすると)

    # def __post_init__(self):
    #     if self.debug:
    #         self.num_env_runners = 1
    #         self.num_cpus_per_env_runner = 1
    #         self.minibatch_size = 256
    #         self.train_batch_size_per_learner = 505
    #         self.num_epochs = 1


# 相対位置を計算
def calc_relative_pos(base_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    return target_pos - base_pos


# マスの半径kマス以内に該当するかどうか
def is_within_k_tiles(base_pos: np.ndarray, target_pos: np.ndarray, k: int) -> bool:
    return np.abs(base_pos[0] - target_pos[0]) <= k and np.abs(base_pos[1] - target_pos[1]) <= k


# 隣接するマスにあるポイントマスを取得
def get_nearby_point_positions(pos: np.ndarray, point_map: np.ndarray, k: int = 1) -> list[np.ndarray]:
    # posを中心にkマス以内のマスを取得
    nearby_positions = []
    up_pos = (pos[0], pos[1] - k)
    if in_map(up_pos) and point_map[up_pos[1], up_pos[0]] == 1:
        nearby_positions.append(up_pos)
    down_pos = (pos[0], pos[1] + k)
    if in_map(down_pos) and point_map[down_pos[1], down_pos[0]] == 1:
        nearby_positions.append(down_pos)
    left_pos = (pos[0] - k, pos[1])
    if in_map(left_pos) and point_map[left_pos[1], left_pos[0]] == 1:
        nearby_positions.append(left_pos)
    right_pos = (pos[0] + k, pos[1])
    if in_map(right_pos) and point_map[right_pos[1], right_pos[0]] == 1:
        nearby_positions.append(right_pos)
    return nearby_positions


# 自身の周囲kタイル以内にいる敵ユニットを抽出
def get_nearby_enemy_unit_ids(
    unit_pos: tuple[int, int], opp_unit_positions: list[tuple[int, int]], k: int
) -> list[int]:
    return [unit_id for unit_id, pos in enumerate(opp_unit_positions) if is_within_k_tiles(unit_pos, pos, k)]


class LuxValueConvModel(nn.Module):
    def __init__(
        self,
        state_space_size: int,
        global_state_space_size: int,
        n_stack: int,
        bilinear: bool = True,
        res: bool = True,
    ) -> None:
        super().__init__()
        self.bilinear = bilinear

        self.inc = DoubleConv(state_space_size, 64, res=res)
        self.down1 = Down(64, 128, res=res)
        self.down2 = Down(128, 256, res=res)
        self.down3 = Down(256, 256, res=res)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.value_net = nn.Sequential(
            nn.Linear((256 + global_state_space_size) * n_stack, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state = batch["state"]
        global_state = batch["global_state"]
        _n, _t, _c, _x, _y = state.shape
        x = state.view(-1, _c, _x, _y)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # sx, syのマップにグローバルステートをブロードキャスト
        sx, sy = x4.shape[2:]
        _n, _t, _c = global_state.shape
        gx = global_state.view(-1, _c, 1, 1)
        gx = gx.repeat(1, 1, sx, sy)

        x4 = torch.cat([x4, gx], dim=1)
        x = self.global_avg_pool(x4).view(_n, -1)
        value_logits = self.value_net(x)

        return {
            "value": value_logits,
        }


def to_action(
    action_map: np.ndarray,
    point_map: np.ndarray,
    obs: dict[str, Any],
    prev_opp_unit_positions: list[tuple[int, int]],
    team_id: int,
    env_params: EnvParams,
) -> np.ndarray:
    """
    sapの地点を決める
    """
    unit_mask = np.array(obs["units_mask"][team_id])  # shape (max_units, )
    unit_positions = np.array(obs["units"]["position"][team_id])  # shape (max_units, 2)
    available_unit_ids = np.where(unit_mask)[0]
    opp_team_id = 1 - team_id

    opp_unit_positions = [tuple(pos) for pos in obs["units"]["position"][opp_team_id] if pos[0] != -1]
    actions = np.zeros((env_params.max_units, 3), dtype=int)
    # unit ids range from 0 to max_units - 1
    for unit_id in available_unit_ids:
        unit_pos = unit_positions[unit_id]
        x, y = unit_pos
        action = action_map[y, x]

        if action == Action.SAP:
            # 範囲内にいる敵ユニットを取得
            nearby_enemy_unit_ids = get_nearby_enemy_unit_ids(unit_pos, opp_unit_positions, env_params.unit_sap_range)
            if len(nearby_enemy_unit_ids) > 0:
                sap_pos = opp_unit_positions[np.random.choice(nearby_enemy_unit_ids)]
                # 敵ユニットが2ステップ以上動いていない場合はsapする
                if point_map[sap_pos[1], sap_pos[0]] == 1 or sap_pos in prev_opp_unit_positions:
                    dx, dy = calc_relative_pos(unit_pos, sap_pos)
                    actions[unit_id] = [Action.SAP, dx, dy]
                else:
                    # 敵ユニットの隣接セルがポイント位置であればそこに移動すると考える。
                    nearby_point_positions = get_nearby_point_positions(sap_pos, point_map)
                    if len(nearby_point_positions) > 0:
                        sap_pos = nearby_point_positions[np.random.choice(len(nearby_point_positions))]
                        dx, dy = calc_relative_pos(unit_pos, sap_pos)
                        actions[unit_id] = [Action.SAP, dx, dy]
        else:
            actions[unit_id] = [action, 0, 0]

    return actions, opp_unit_positions


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
        """
        num_actions = len(Action)
        action_space = gym.spaces.MultiDiscrete([num_actions] * EnvParams.map_width * EnvParams.map_height)
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
                "legal_action_mask": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(len(Action), EnvParams.map_height, EnvParams.map_width),
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
        self.prev_opp_unit_positions = {
            "player_0": [],
            "player_1": [],
        }
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
        agent0_global_state = extract_global_state(obs["player_0"], 0, self.env_params, self.episode_store1)
        agent1_global_state = extract_global_state(obs["player_1"], 1, self.env_params, self.episode_store2)
        agent0_legal_action_mask = get_valid_policy_map(obs["player_0"], 0, self.episode_store1)
        agent1_legal_action_mask = get_valid_policy_map(obs["player_1"], 1, self.episode_store2)

        self.agent0_states.append(agent0_state)
        self.agent1_states.append(agent1_state)
        self.agent0_global_states.append(agent0_global_state)
        self.agent1_global_states.append(agent1_global_state)
        return {
            "player_0": {
                "state": np.stack(list(self.agent0_states), axis=0),
                "global_state": np.stack(list(self.agent0_global_states), axis=0),
                "legal_action_mask": agent0_legal_action_mask,
            },
            "player_1": {
                "state": np.stack(list(self.agent1_states), axis=0),
                "global_state": np.stack(list(self.agent1_global_states), axis=0),
                "legal_action_mask": agent1_legal_action_mask,
            },
        }

    def _create_action(self, action_dict: dict[str, Any]) -> dict[str, np.ndarray]:
        actions = {agent_id: np.zeros((EnvParams.max_units, 3), dtype=np.int32) for agent_id in self.agents}

        # 1次元マップの行動空間で渡ってくるので2次元マップに変換
        action_map1 = action_dict["player_0"].reshape(EnvParams.map_height, EnvParams.map_width)
        action_map2 = action_dict["player_1"].reshape(EnvParams.map_height, EnvParams.map_width)

        point_map1 = self.agent0_states[-1][State.POINTS]
        point_map2 = self.agent1_states[-1][State.POINTS]
        actions["player_0"], self.prev_opp_unit_positions["player_0"] = to_action(
            action_map1, point_map1, self.obs["player_0"], self.prev_opp_unit_positions["player_0"], 0, self.env_params
        )
        actions["player_1"], self.prev_opp_unit_positions["player_1"] = to_action(
            action_map2, point_map2, self.obs["player_1"], self.prev_opp_unit_positions["player_1"], 1, self.env_params
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

        terminated = {agent_id: done.item() for agent_id, done in _terminated.items()}
        truncated = {agent_id: done.item() for agent_id, done in _truncated.items()}
        # "__all__" (required) is used to indicate env termination.
        terminated["__all__"] = np.all(list(truncated.values()))  # luxaiはtruncatedがTrueになる
        info = {agent_id: {} for agent_id in self.agents}
        steps = self.obs["player_0"]["steps"].item()
        reward = self.reward_fn(_reward, steps=steps)
        return state, reward, terminated, truncated, info

    def reward_fn(self, raw_reward: jnp.ndarray, steps: int) -> dict[str, int]:
        """
        raw_rewardは累積値なので、前回との差分を取って現在のステップでの報酬を計算する
        マッチの勝利数をそのまま報酬とする
        他の報酬候補
        - 差分報酬: 3-2の場合1、2-3の場合-1
        - 勝敗報酬: 勝ち1、負け-1 引き分け0
        """
        _reward = to_numpy(raw_reward)
        current_rewards = {agent_id: int(r.item()) for agent_id, r in _reward.items()}

        # 差分を計算して現在のステップでの報酬を取得
        step_rewards = {
            agent_id: current_rewards[agent_id] - self.prev_raw_reward[agent_id] for agent_id in current_rewards.keys()
        }

        # 現在の累積報酬を保存
        self.prev_raw_reward = current_rewards

        return step_rewards


class LuxUnetTorchRLModule(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def setup(self):
        self.policy_model = LuxUNetModel(
            state_space_size=len(State),
            global_state_space_size=len(GlobalState),
            action_space_size=len(Action),
            hidden_state_space_size=len(HiddenState),
            n_stack=self.model_config["n_stack"],
            res=True,
        )

        self.value_model = LuxValueConvModel(
            state_space_size=len(State),
            global_state_space_size=len(GlobalState),
            n_stack=self.model_config["n_stack"],
        )

        if self.model_config["pretrained_path"]:
            ckpt = torch.load(self.model_config["pretrained_path"], weights_only=True, map_location=torch.device("cpu"))
            state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
            self.policy_model.load_state_dict(state_dict)

        self._values = None

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        batch_size = batch[Columns.OBS]["state"].shape[0]
        outputs = self.policy_model(batch[Columns.OBS])
        policy_logits = outputs["policy"]
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
            Columns.LOSS_MASK: unit_mask,
        }

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        return self._forward(batch, **kwargs)

    @override(ValueFunctionAPI)
    def compute_values(self, batch: dict[str, Any], embeddings: Any | None = None) -> torch.Tensor:
        outputs = self.value_model(batch[Columns.OBS])
        self._values = outputs["value"].squeeze(dim=1)
        return self._values

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self) -> type[TorchDistribution]:
        return TorchCategorical


def train_metric_value(results: dict[str, Any], key: str) -> float:
    # 自身はp0として評価している
    return results["p0"][key]


@ray.remote
class EpisodeStatsCollector:
    def __init__(self):
        self.episode_end_times = deque(maxlen=1000)
        self.total_episodes = 0
        self.last_log_time = time()

    def add_episode(self):
        self.episode_end_times.append(time())
        self.total_episodes += 1

    def get_stats(self):
        if len(self.episode_end_times) < 2:
            return 0.0, self.total_episodes

        window_duration = self.episode_end_times[-1] - self.episode_end_times[0]
        if window_duration == 0:
            return 0.0, self.total_episodes

        eps_per_sec = (len(self.episode_end_times) - 1) / window_duration
        eps_per_min = eps_per_sec * 60
        return eps_per_min, self.total_episodes


class WandbLoggerCallback(RLlibCallback):
    def __init__(self):
        self.output_dir = Config.output_dir
        # グローバルな統計コレクターを作成（一度だけ）
        if not hasattr(WandbLoggerCallback, "_stats_collector"):
            WandbLoggerCallback._stats_collector = EpisodeStatsCollector.remote()

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
        print(f"{result['learners'].keys()=}")
        for key in learner_metrics:
            wandb.log(
                {
                    f"train/{key}": train_metric_value(result["learners"], key),
                }
            )

    # 学習したモデルの性能評価をwandbに流す用
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

        evaluation_metrics['env_runners'].keys()=dict_keys([
        'agent_episode_returns_mean', 'timers', 'num_agent_steps_sampled_lifetime', 'episode_len_min', 'num_module_steps_sampled',
        'num_agent_steps_sampled', 'episode_duration_sec_mean', 'episode_return_max', 'episode_len_mean', 'num_env_steps_sampled_lifetime',
        'module_episode_returns_mean', 'num_episodes', 'num_episodes_lifetime', 'agent_steps', 'episode_return_min', 'num_module_steps_sampled_lifetime',
        'num_env_steps_sampled', 'episode_len_max', 'env_to_module_sum_episodes_length_out', 'episode_return_mean', 'env_to_module_sum_episodes_length_in'
        ])
        """
        # 自身はplayer_0として評価している
        # print("#########################")
        # print(f"{evaluation_metrics['env_runners']['agent_episode_returns_mean']=}")
        # print(f"{evaluation_metrics['env_runners']['module_episode_returns_mean']=}")
        # print(f"{evaluation_metrics['env_runners']['episode_return_mean']=}")
        # print("#########################")
        # episode_rewards = evaluation_metrics["env_runners"]["agent_episode_returns_mean"]["player_0"]
        # wins = sum(1 for r in episode_rewards if r > 0)  # 報酬が正の場合は勝利
        # win_rate = wins / len(episode_rewards) if episode_rewards else 0.0

        # wandb.log(
        #     {
        #         # player_0を自身として評価している
        #         "evaluate/agent_episode_returns_mean": evaluation_metrics["env_runners"]["agent_episode_returns_mean"][
        #             "player_0"
        #         ],
        #         "evaluate/episode_duration_sec_mean": evaluation_metrics["env_runners"]["episode_duration_sec_mean"],
        #         "evaluate/win_rate": win_rate,
        #     }
        # )
        checkpoint_dir = algorithm.save_to_path(self.output_dir)
        print(f"save to {checkpoint_dir}")

    # データ収集状況をwandbに流す用
    @override(RLlibCallback)
    def on_episode_end(
        self,
        *,
        episode: EpisodeType,
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: MetricsLogger | None = None,
        **kwargs,
    ) -> None:
        # エピソード完了を記録
        ray.get(self._stats_collector.add_episode.remote())
        # 統計を取得して記録
        eps_per_min, total_episodes = ray.get(self._stats_collector.get_stats.remote())
        # rewards = episode.get_rewards()
        # reward = {agent_id: rewards[agent_id][-1] for agent_id in rewards.keys()}
        # duration = episode.get_duration_s()
        # if total_episodes % 10 == 0:
        print(f"Episode {total_episodes} finished. Collection speed: {eps_per_min:.2f} eps/min")


class CustomIMPALATorchLearner(IMPALALearner, TorchLearner):
    """Implements the IMPALA loss function in torch."""

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

        # 最初にmap情報をbatch方向に展開する処理を入れる
        # これにより通常の実装と同じように計算できる

        # TODO (sven): Now that we do the +1ts trick to be less vulnerable about
        #  bootstrap values at the end of rollouts in the new stack, we might make
        #  this a more flexible, configurable parameter for users, e.g.
        #  `v_trace_seq_len` (independent of `rollout_fragment_length`). Separation
        #  of concerns (sampling vs learning).
        rollout_frag_or_episode_len = config.get_rollout_fragment_length()
        recurrent_seq_len = batch.get("seq_lens")

        loss_mask = fwd_out[Columns.LOSS_MASK].float()
        loss_mask_time_major = make_time_major(
            loss_mask,
            trajectory_len=rollout_frag_or_episode_len,
            recurrent_seq_len=recurrent_seq_len,
        )
        size_loss_mask = torch.sum(loss_mask)

        # Behavior actions logp and target actions logp.
        behaviour_actions_logp = batch[Columns.ACTION_LOGP]
        target_policy_dist = module.get_train_action_dist_cls().from_logits(fwd_out[Columns.ACTION_DIST_INPUTS])
        target_actions_logp = target_policy_dist.logp(batch[Columns.ACTIONS])

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

        # (time_dim, batch_size*24*24) マップ形式のデータ
        # これをユニットがいる位置のみ抽出して1ステップに1つのデータ(time_dim, batch_size)となるようにsumをとる(対数確率)
        target_actions_logp_time_major = (target_actions_logp_time_major * loss_mask_time_major).sum(dim=2)
        behaviour_actions_logp_time_major = (behaviour_actions_logp_time_major * loss_mask_time_major).sum(dim=2)

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
        mean_pi_loss = pi_loss / size_loss_mask

        # The baseline loss.
        delta = values_time_major - vtrace_adjusted_target_values
        vf_loss = 0.5 * torch.sum(torch.pow(delta, 2.0))
        mean_vf_loss = vf_loss / size_loss_mask

        # The entropy loss.
        entropy_loss = -torch.sum(target_policy_dist.entropy() * loss_mask)
        mean_entropy_loss = entropy_loss / size_loss_mask

        # The summed weighted loss.
        total_loss = (
            mean_pi_loss
            + mean_vf_loss * config.vf_loss_coeff
            + (mean_entropy_loss * self.entropy_coeff_schedulers_per_module[module_id].get_current_value())
        )

        # Log important loss stats.
        self.metrics.log_dict(
            {
                "pi_loss": pi_loss,
                "mean_pi_loss": mean_pi_loss,
                "vf_loss": vf_loss,
                "mean_vf_loss": mean_vf_loss,
                ENTROPY_KEY: -mean_entropy_loss,
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
        IMPALAConfig()
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
            # batch_sizeから自動で適切な値を計算してくれるためこの設定が推奨されている
            # rollout_fragment_length = "auto",
            rollout_fragment_length=cfg.rollout_fragment_length,
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
            # loss
            vtrace=True,
            vf_loss_coeff=cfg.vf_loss_coeff,
            entropy_coeff=cfg.entropy_coeff,
        )
        # マルチエージェント設定
        # https://github.com/ray-project/ray/blob/2a85cef1ad8105d8dda01d709da7b0eaeb337caa/rllib/examples/multi_agent/rock_paper_scissors_heuristic_vs_learned.py#L94
        # https://github.com/ray-project/ray/blob/2a85cef1ad8105d8dda01d709da7b0eaeb337caa/rllib/examples/multi_agent/rock_paper_scissors_learned_vs_learned.py#L65
        # TODO: 本当はself-playにして評価のみbest policyと対戦させたいが評価時にpolicyを指定する方法がわからない
        .multi_agent(
            # RLで扱うagent(policy)の名前
            policies={"p0", "best"},
            # 各agentのポリシーを決める関数
            policy_mapping_fn=lambda aid, episode, **kwargs: ("p0" if aid == "player_0" else "best"),
            # 学習はp0だけ学習
            policies_to_train=["p0"],
        )
        # https://docs.ray.io/en/latest/rllib/rllib-rlmodule.html#construction-through-rlmodulespecs
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                # policy名とモデルの紐づけ(self playの場合p0とp1の両方を学習対象にする)
                rl_module_specs={
                    "p0": rl_module_spec,
                    "best": rl_module_spec,
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
            evaluation_sample_timeout_s=60 * 5,
            evaluation_force_reset_envs_before_iteration=True,  # 各評価の前に環境をリセット
            evaluation_parallel_to_training=True,  # 評価と学習を並列に実行
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

    train_start_time = time()
    train_count = 0
    while True:
        result = trainer.train()
        train_count += 1
        spend_minutes = (time() - train_start_time) / 60
        # print(f"{train_count=} is finished. spend {spend_minutes:.1f} minutes")
        print("train finished")
        # print(f"{result.keys()=}")
        # 指定した時間経ったら学習を終了
        # if spend_minutes > cfg.training_minutes:
        break


if __name__ == "__main__":
    main()
