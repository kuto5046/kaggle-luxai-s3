import random
import logging
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
from lux.utils import (
    State,
    Action,
    GlobalState,
    HiddenState,
    EpisodeStore,
    to_np,
    extract_state,
    switch_action,
    calc_relative_pos,
    extract_global_state,
    get_valid_policy_map,
    get_nearby_enemy_unit_ids,
    get_nearby_point_positions,
)
from lux.models import LuxUNetModel, LuxValueConvModel
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

# policy名
OWN_POLICY = "p0"
SELF_PLAY_POLICY = (
    "self-play"  # TODO: 評価時に対戦相手をbestのみに制御することができていないため未使用だが本当は使いたい
)
BEST_POLICY = "best"
LB_BEST_POLICY = "lb_best"  # TODO: モデルや特徴量が異なるため未使用だが本当は使いたい


@dataclass
class Config:
    exp_name: str = Path(__file__).parent.name
    is_gcp: bool = False
    notes: str = "パラメータをlux s2のRL解法に寄せてみる"
    model_name: str = "lux_unet"
    env_name: str = "lux-s3-v0"
    n_stack: int = 4
    root_dir: Path = Path(f"/home/user/work/exp/{exp_name}")
    best_pretrained_path: Path | None = Path("/home/user/work/exp/best/output/best_model.ckpt")
    lb_best_pretrained_path: Path | None = Path("/home/user/work/exp/best/output/best_model.ckpt")
    # lb_best_pretrained_path: Path | None = Path(f"/home/user/work/exp/lb_best/output/best_model.ckpt")
    debug: bool = False
    output_dir: Path = root_dir / "output"

    # 以下の3つのrunnerにcpuとgpuを割り振る。cpuの合計値がcpu数を超えないように注意(現在は24をactor: 21,learner: 1,evaluator:2に割り振る)
    # データ収集用
    num_env_runners: int = 18  # actorの数
    num_cpus_per_env_runner: int = 1
    num_gpus_per_env_runner: int = 0
    rollout_fragment_length: int | str | None = (
        101  # 考慮したいstep数を設定してやる。報酬が含まれるように1マッチ分の長さにする
    )

    # 学習用(GPUの数=learnerと考えて良い)
    num_learners: int = 0  # IMPALAの場合gpuが1つなら0に設定するとlocal learnerとして扱われる、処理が早くなる
    num_cpus_per_learner: int = 1
    num_gpus_per_learner: int = 1

    # 評価用
    evaluation_num_env_runners: int = 5  # 評価用のenv runnerの数
    evaluation_interval: int = 50  # 何回trainをしたら評価を実施するか　１回が30secくらいなので50回で1500sec=25分くらい
    evaluation_duration: int = (
        30  # 1回の評価で何エピソード分評価するか(学習と並列してやるため達成できないこともあるかも)
    )

    # 評価と学習を並列に実行するかどうか
    # 並列に実行すると待機処理が短縮されるようだが、今回の設定だとtrainが30secくらいで終わってしまうため結果として評価がボトルネックになってしまう
    # 1回の学習を長くするか、評価にworkerを多く割り当てて評価時間を短縮するのが良さそう
    evaluation_parallel_to_training: bool = True

    # learner
    training_minutes: int = 60 * 24  # 1日
    learner_queue_size: int = 20  # workerからLearnerに送られるバッチのキューの最大サイズ. [batch_size]*queue_sizeがcpuメモリに乗りbatchごとに学習する
    gamma: float = 0.9995
    lr: float = 1e-5
    # batch size 一応1episodeのサイズにしてるが不要かも。もしくはrollout_fragment_length部分で調整する
    train_batch_size_per_learner: int = 512
    # 1回の学習データ(train_batch_size*queue_size)を何epoch分学習するか
    num_epochs: int = 1
    replay_proportion: float = 0.0  # リプレイバッファの割合
    # loss
    vtrace_clip_rho_threshold: float = 1.0  # 価値関数のlossの係数
    vtrace_clip_pg_rho_threshold: float = 1.0  # ポリシー勾配のlossの係数
    vf_loss_coeff: float = 1.0  # 価値関数のlossの係数
    entropy_coeff: float = 1e-5  # エントロピーのlossの係数(大きくすると探索が活発になる)

    def __post_init__(self):
        if self.is_gcp:
            self.num_env_runners: int = 96 - 4 - 15  # actorの数
            self.num_learners: int = 0
            self.evaluation_num_env_runners: int = 15
            self.learner_queue_size: int = 100
            self.rollout_fragment_length = 505

        if self.debug:
            self.num_env_runners = 1
            self.num_cpus_per_env_runner = 1
            self.evaluation_num_env_runners = 1
            self.evaluation_interval = 1
            self.evaluation_duration = 5
            # self.evaluation_parallel_to_training = False
            self.training_minutes = 10
            self.learner_queue_size = 1
            self.num_epochs = 1


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
        # 自陣が(0, 0)になるようにstateを反転(state, height, width)
        agent1_state = np.flip(agent1_state, [1, 2])

        agent0_global_state = extract_global_state(obs["player_0"], 0, self.env_params, self.episode_store1)
        agent1_global_state = extract_global_state(obs["player_1"], 1, self.env_params, self.episode_store2)
        agent0_legal_action_mask = get_valid_policy_map(obs["player_0"], 0, self.episode_store1)
        agent1_legal_action_mask = get_valid_policy_map(obs["player_1"], 1, self.episode_store2)
        # 自陣が(0, 0)になるようにmask mapを反転(action, height, width)
        # TODO: agent.pyにはないがrlではこれがないとダメなのがよくわかっていない。
        # agent1_legal_action_mask = np.flip(agent1_legal_action_mask, [1, 2])
        # agent1_legal_action_mask[Action.UP], agent1_legal_action_mask[Action.DOWN] = (
        #     agent1_legal_action_mask[Action.DOWN].copy(),
        #     agent1_legal_action_mask[Action.UP].copy(),
        # )
        # agent1_legal_action_mask[Action.LEFT], agent1_legal_action_mask[Action.RIGHT] = (
        #     agent1_legal_action_mask[Action.RIGHT].copy(),
        #     agent1_legal_action_mask[Action.LEFT].copy(),
        # )
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

        # 自陣を(0, 0)に固定していたものを元の位置に戻す
        action_map2 = np.flip(action_map2, [0, 1]).copy()  # x, y軸反転
        action_map2 = switch_action(action_map2, Action.RIGHT, Action.LEFT)
        action_map2 = switch_action(action_map2, Action.UP, Action.DOWN)

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
        steps = self.obs["player_0"]["steps"].item()

        terminated = {agent_id: done.item() for agent_id, done in _terminated.items()}
        truncated = {agent_id: done.item() for agent_id, done in _truncated.items()}
        # "__all__" (required) is used to indicate env termination.
        terminated["__all__"] = np.all(list(truncated.values()))  # luxaiはtruncatedがTrueになる
        info = {agent_id: {} for agent_id in self.agents}
        reward = self.reward_fn(_reward, steps=steps)
        return state, reward, terminated, truncated, info

    def reward_fn(self, raw_reward: jnp.ndarray, steps: int) -> dict[str, int]:
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

        # 現在の累積報酬を保存
        self.prev_raw_reward = current_rewards

        return step_rewards


class LuxUnetTorchRLModule(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def setup(self):
        # torch.set_num_threads(1)
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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(self.model_config["pretrained_path"], weights_only=False, map_location=device)
            state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
            self.policy_model.load_state_dict(state_dict)
            print(f"Loaded model from {self.model_config['pretrained_path']} {device=}")

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
        # MEMO: 自陣固定対応のaction_maskの挙動が怪しいのでひとまず適用しないでおく
        # masked_policy_logits = policy_logits - 1e32 * (1 - action_mask)
        # この時点では(batch, action, height, width)なので(batch, height, width, action)に変換
        policy_logits = policy_logits.reshape(batch_size, num_actions, -1).transpose(2, 1)
        unit_mask = unit_mask.reshape(batch_size, -1)
        return {
            Columns.ACTION_DIST_INPUTS: policy_logits,
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
        # learnersがない場合はskip(並列で実行しているため最初はないはず)
        if "learners" not in result:
            return

        # 1回の学習で学習したデータ数
        time_this_iter_s = result["time_this_iter_s"]
        time_total_s = result["time_total_s"]
        num_training_step_calls_per_iteration = result["num_training_step_calls_per_iteration"]  # 累積値
        # sample_size = result["num_env_steps_sampled_lifetime"]

        # 学習状況をwandbに流す用
        wandb.log(
            {
                "train/training_iteration": result["timers"]["training_iteration"],  # 何回めの学習か
                "train/time_this_iter_s": time_this_iter_s,  # 1回の学習時間
                "train/time_total_s": time_total_s,  # 学習総時間
                "train/num_training_step_calls_per_iteration": num_training_step_calls_per_iteration,  # 1回の学習で何回training_stepが呼ばれたか
            }
        )

        # 学習データのサンプリング時間
        if result.get("env_runners"):
            if result["env_runners"].get("time_between_sampling"):
                wandb.log(
                    {
                        "train/env_runner_time_between_sampling": result["env_runners"]["time_between_sampling"],
                    }
                )

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
        trained_episodes_per_minute = (trained_episodes_lifetime / time_total_s) * 60

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
        best_win_rate = ray.get(self._stats_collector.get_best_win_rate.remote())
        if best_win_rate < current_win_rate:
            save_model(algorithm, self.output_dir, suffix=f"model_eval_{self._current_evaluation_id}")
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

        if in_evaluation:
            ray.get(self._stats_collector.record_evaluation_result.remote(is_win))
            stats = ray.get(self._stats_collector.get_speed_stats.remote(in_evaluation))
            self.logger.info(
                f"Evaluation Episode {stats['total_episodes']} finished. {is_win=} {episode_total_reward=} Collection speed: {stats['episode_per_minute']:.2f} eps/min"
            )
        else:
            stats = ray.get(self._stats_collector.get_speed_stats.remote(in_evaluation))
            self.logger.info(
                f"Episode {stats['total_episodes']} finished. {is_win=} {episode_total_reward=} Collection speed: {stats['episode_per_minute']:.2f} eps/min"
            )


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
        size_loss_mask = torch.sum(loss_mask)

        # Behavior actions logp and target actions logp.
        behaviour_actions_logp = batch[Columns.ACTION_LOGP]
        target_policy_dist = module.get_train_action_dist_cls().from_logits(fwd_out[Columns.ACTION_DIST_INPUTS])
        target_actions_logp = target_policy_dist.logp(batch[Columns.ACTIONS])

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
    best_rl_module_spec = RLModuleSpec(
        module_class=LuxUnetTorchRLModule,
        observation_space=observation_space,
        action_space=action_space,
        # モデル内部でself.model_config["key"]でアクセスできる
        model_config={
            "n_stack": cfg.n_stack,
            "pretrained_path": cfg.best_pretrained_path,
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
            num_gpus_per_env_runner=cfg.num_gpus_per_env_runner,
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
                # SELF_PLAY_POLICY,
                BEST_POLICY,
                # LB_BEST_POLICY,
            },
            # 各agentのポリシーを決める関数
            policy_mapping_fn=lambda aid, episode, **kwargs: (
                OWN_POLICY
                if aid == "player_0"
                else random.choice(
                    [
                        # SELF_PLAY_POLICY,
                        BEST_POLICY,
                        # LB_BEST_POLICY,
                    ]
                )
            ),
            # 学習は自身のpolicyとself-playのpolicyを学習
            policies_to_train=[
                OWN_POLICY,
                # SELF_PLAY_POLICY
            ],
        )
        # https://docs.ray.io/en/latest/rllib/rllib-rlmodule.html#construction-through-rlmodulespecs
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                # policy名とモデルの紐づけ
                rl_module_specs={
                    OWN_POLICY: best_rl_module_spec,
                    # SELF_PLAY_POLICY: best_rl_module_spec,
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
            evaluation_parallel_to_training=cfg.evaluation_parallel_to_training,  # 評価と学習を並列に実行
            # 評価用の上書き設定.これにより評価時はlb_bestポリシーと自身の対戦になる
            # evaluation_config={
            #     "multi_agent": {
            #         "policies": {OWN_POLICY: None, BEST_POLICY: None},
            #         "policy_mapping_fn": lambda aid, episode, **kwargs: (
            #             OWN_POLICY if aid == "player_0" else BEST_POLICY
            #         ),
            #     }
            # },
        )
        # .checkpointing(
        #     export_native_model_files=True,
        # )
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


def save_model(trainer: Algorithm, output_dir: Path, suffix: str = "model"):
    """
    rllibのapiを使わず直接モデルを保存する
    モデルの名前はrlmoduleで定義した名前を使う
    """
    policy_state_dict = trainer.get_module(OWN_POLICY).policy_model
    value_state_dict = trainer.get_module(OWN_POLICY).value_model

    torch.save(policy_state_dict, output_dir / f"policy_{suffix}.pth")
    torch.save(value_state_dict, output_dir / f"value_{suffix}.pth")


def main() -> None:
    cfg = Config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # ロガーのセットアップ
    logger = setup_logger(cfg.output_dir)

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
        train_count += 1
        spend_minutes = (time() - train_start_time) / 60
        total_spend_minutes = (time() - total_train_start_time) / 60
        logger.info(f"Training iteration {train_count} finished. Spent {spend_minutes:.1f} minutes")
        # 指定した時間経ったら学習を終了
        if total_spend_minutes > cfg.training_minutes:
            logger.info(f"Training completed after {spend_minutes:.1f} minutes")
            break

    save_model(trainer, cfg.output_dir, suffix="latest_model")


def debug():
    cfg = Config()
    env = env_creator({"n_stack": cfg.n_stack})
    obs, _ = env.reset()
    policy_model = LuxUNetModel(
        state_space_size=len(State),
        global_state_space_size=len(GlobalState),
        action_space_size=len(Action),
        hidden_state_space_size=len(HiddenState),
        n_stack=cfg.n_stack,
        res=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(cfg.best_pretrained_path, weights_only=False, map_location=device)
    state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
    policy_model.load_state_dict(state_dict)
    policy_model.eval()
    device = "cpu"
    policy_model.to(device)
    for i in range(3):
        state1 = obs["player_0"]
        state2 = obs["player_1"]
        # print(f"{state1['state'][:, State.OWN_UNIT_COUNT, 0, 0]=}" f"{state2['state'][:, State.OWN_UNIT_COUNT, 0, 0]=}")
        # 辞書をtensorに変換しsqueeze(0)してデバイスに載せる
        torch_state1 = {}
        torch_state2 = {}
        for k, v in state1.items():
            torch_state1[k] = torch.from_numpy(v.copy()).unsqueeze(0).to(device)
        for k, v in state2.items():
            torch_state2[k] = torch.from_numpy(v.copy()).unsqueeze(0).to(device)
        outputs1 = policy_model(torch_state1)
        outputs2 = policy_model(torch_state2)
        batch_size = 1
        num_actions = len(Action)
        action1 = to_np(
            outputs1["policy"].reshape(batch_size, num_actions, -1).transpose(2, 1).squeeze(0).argmax(dim=-1)
        )
        action2 = to_np(
            outputs2["policy"].reshape(batch_size, num_actions, -1).transpose(2, 1).squeeze(0).argmax(dim=-1)
        )

        action_dict = {
            "player_0": action1,
            "player_1": action2,
        }
        obs, _, _, _, _ = env.step(action_dict)


if __name__ == "__main__":
    main()
    # debug()
