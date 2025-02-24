import json
from typing import Any
from pathlib import Path
from collections import deque, defaultdict
from dataclasses import dataclass

import numpy as np
import polars as pl
import streamlit as st
import plotly.graph_objects as go
from agent import ILAgent
from lightning import seed_everything
from lux.utils import (
    State,
    Action,
    GlobalState,
    EpisodeStore,
    extract_state,
    extract_gt_state,
    extract_global_state,
)
from lux.params import EnvParams
from plotly.subplots import make_subplots

ACTION_NAMES = [Action(i).name for i in range(len(Action))]

# ページ設定
st.set_page_config(layout="wide")


@dataclass
class Config:
    seed: int = 2025
    n_stack: int = 4
    team_name: str = "kuto & okumura"
    exp_name: str = Path(__file__).parent.name
    checkpoint_path: Path = Path(f"/home/task/kaggle/branch/kibuna/best/exp/{exp_name}/output/best_model.ckpt")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# データ表示
def visualize_state(state, state_enum, active_states: list[str], title="State", n_cols: int = 4):
    st.subheader(title)
    # 全てのチャンネルを可視化する

    # ヒートマップ表示
    num_channels = state.shape[0]

    cols = st.columns(n_cols)
    figure_count = 0
    for i in range(num_channels):
        if state_enum(i).name in active_states:
            with cols[figure_count % n_cols]:
                fig = go.Figure(data=go.Heatmap(z=state[i], colorscale="greens"))
                fig.update_layout(title=f"{state_enum(i).name}", width=300, height=290)
                st.plotly_chart(fig, key=f"{title}_{i}")
            figure_count += 1


def visualize_pred_action(action, title="Predict Action", color="blues"):
    st.subheader(title)
    # ヒートマップ表示 sequentialではないdeiscreteな色を使う
    fig = go.Figure(data=go.Heatmap(z=action, zmax=len(Action) - 1, zmin=0, colorscale=color))
    fig.update_layout(width=400, height=400)
    st.plotly_chart(fig, key=f"{title}")


def visualize_action(action, title="Action"):
    st.subheader(title)
    # ヒートマップ表示 sequentialではないdeiscreteな色を使う
    fig = go.Figure(data=go.Heatmap(z=action, zmax=len(Action) - 1, zmin=0, colorscale="reds"))
    fig.update_layout(width=400, height=400)
    st.plotly_chart(fig, key=f"{title}")


@st.cache_data(show_spinner="episodeのデータを取得中...")
def extract_results(
    json_load: dict[str, Any], target_team_id: int, env_params: EnvParams, episode_id: str, cfg: Config
) -> dict[str, list[np.ndarray]]:
    model = ILAgent(env_params, cfg.checkpoint_path, cfg.n_stack)
    # データを全step取得
    episode_store = EpisodeStore(target_team_id, env_params, False, episode_id)
    steps = json_load["steps"]
    results = defaultdict(list)
    states = deque(maxlen=cfg.n_stack)
    global_states = deque(maxlen=cfg.n_stack)
    prev_actions = np.zeros((env_params.max_units, 3), dtype=np.int32)
    for i in range(cfg.n_stack):
        states.append(np.zeros((len(State), 24, 24), dtype=np.float32))
        global_states.append(np.zeros((len(GlobalState),), dtype=np.float32))

    for step_idx in range(len(steps) - 1):  # 505でdoneとなるため-1
        step_info = steps[step_idx]
        next_step_info = steps[step_idx + 1]
        obs = json.loads(step_info[target_team_id]["observation"]["obs"])
        gt_obs = step_info[0]["info"]["replay"]["observations"][0]
        next_actions = next_step_info[target_team_id]["action"]

        # マッチごとにリセットされる要素をリセット
        if obs["match_steps"] == 0:
            episode_store.reset()
        # リセット時以外はupdateをする
        else:
            prev_actions = np.array(step_info[target_team_id]["action"])
            episode_store.update(obs, prev_actions)

        state = extract_state(obs, target_team_id, episode_store)
        global_state = extract_global_state(obs, target_team_id, env_params, episode_store)
        gt_state = extract_gt_state(gt_obs, target_team_id)
        policy_map, _ = model.predict(obs, target_team_id, episode_store)

        results["obs"].append(obs)
        results["state"].append(state)
        results["gt_state"].append(gt_state)
        results["global_state"].append(global_state)
        results["policy_map"].append(policy_map)
        results["action"].append(next_actions)  # その状態からどう行動したかを知りたいのnext_actions
    return results


def visualize_unit_positions(unit_positions: list[tuple[int, int]], target_team_id: int, title="Unit Positions"):
    unit_map = np.ones((24, 24), dtype=np.int32) * -1
    for unit_id, (x, y) in enumerate(unit_positions):
        unit_map[y, x] = unit_id
    # テキストマトリックスを作成
    text_matrix = [["" for _ in range(24)] for _ in range(24)]
    for unit_id, (x, y) in enumerate(unit_positions):
        text_matrix[y][x] = str(unit_id)

    # 公式visualizerと同じ見た目にするためにフリップ
    if target_team_id == 0:
        unit_map = np.flip(unit_map, axis=0)
        text_matrix = np.flip(text_matrix, axis=0)
    else:
        unit_map = np.flip(unit_map, axis=1)
        text_matrix = np.flip(text_matrix, axis=1)

    fig = go.Figure(
        data=go.Heatmap(
            z=unit_map,
            showscale=False,
            colorscale=[[0, "white"], [1, "rgb(100, 149, 237)"]],
            zmin=0,
        )
    )

    # テキストの追加
    fig.update_traces(text=text_matrix, texttemplate="%{text}", textfont={"size": 20, "color": "black"})

    # グリッド線の追加
    for i in range(25):
        fig.add_shape(type="line", x0=i - 0.5, y0=-0.5, x1=i - 0.5, y1=23.5, line=dict(color="lightgray", width=1))
        fig.add_shape(type="line", x0=-0.5, y0=i - 0.5, x1=23.5, y1=i - 0.5, line=dict(color="lightgray", width=1))

    fig.update_layout(
        title=title,
        width=800,
        height=800,
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
    )
    st.plotly_chart(fig, key="unitの位置")


def visualize_policy(
    policy_map: np.ndarray,
    real_actions: np.ndarray,
    unit_positions: list[tuple[int, int]],
    title="Policy",
    n_cols: int = 8,
):
    # ユニット数に基づいて行数を計算（1行あたり4ユニット）
    n_units = len(unit_positions)
    n_cols = 4
    n_rows = (n_units + n_cols - 1) // n_cols
    # subplotの作成
    fig = go.Figure()
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[
            f"{unit_id=} \n real action={Action(real_actions[unit_id][0]).name}"
            for unit_id, (x, y) in enumerate(unit_positions)
        ],
    )

    for unit_id, (x, y) in enumerate(unit_positions):
        row = unit_id // n_cols + 1
        col = unit_id % n_cols + 1
        policy = policy_map[:, y, x]

        fig.add_trace(go.Bar(x=ACTION_NAMES, y=policy, showlegend=False), row=row, col=col)
        fig.update_yaxes(range=[0, 1], row=row, col=col)
    # レイアウトの調整
    fig.update_layout(
        height=200 * n_rows,
        # width=800,
        showlegend=False,
        margin=dict(t=30, l=30, r=30, b=30),
    )
    st.plotly_chart(fig, key="policy_plots")


def visualize_global_state(global_state: np.ndarray):
    # GlobalStateの各要素と値を辞書に変換
    state_dict = {GlobalState(i).name: global_state[i] for i in range(len(GlobalState))}

    # DataFrameに変換して表示
    df = pl.DataFrame([state_dict])
    st.subheader("Global State")
    st.write(df)


def usage():
    st.subheader("使い方")
    st.write("""
    1. jsonファイルをuploadする。提出後のjsonファイルを想定しています。
    2. 最初に状態やポリシー計算をまとめて実施するため5秒ほどかかります。
    3. 読み込みが完了すると公式visualizerのリンク、環境パラメータ、フィルタ、State, Policyが表示されます。
    4. 環境パラメータは観測できないものも含む真のパラメータです。
    5. フィルタは現在はepisodeのステップ数を選択できます。
    6. Stateは特徴量に使っているState, GlobalStateと比較用の真値を表示しています。数が多いため現在は比較したそうな推定系の特徴量に絞ってます
    7. Policyは左側にユニットの位置、右側にユニットごとのポリシーを表示します。ユニット位置は公式visualizerと同じ見た目にしていますが表示されるx,y座標は可視化の都合上違う値になってるため注意してください。
    8. はっきりとした理由は分かっていませんが、実際に対戦で取った行動(real_action)とpolicyが一致しないケースがあります。
    """)


def main():
    cfg = Config()
    seed_everything(cfg.seed, workers=True)
    st.title(f"Episode Data Visualizer in {cfg.exp_name}")
    usage()
    # jsonファイルをupload
    json_load = st.file_uploader("jsonファイルをupload", type="json")
    if json_load is not None:
        json_load = json.load(json_load)
        episode_id = json_load["info"]["EpisodeId"]
        target_team_id = json_load["info"]["TeamNames"].index(cfg.team_name)
        link = f"https://s3vis.lux-ai.org/#/visualizer?input={episode_id}"
        st.info(f"[{episode_id=}の公式Visualizerリンク]({link})")

        ############################################################
        # 環境パラメータの情報
        ############################################################
        st.subheader("環境パラメータ")
        env_params = EnvParams(**json_load["configuration"]["env_cfg"])
        gt_env_params = EnvParams(**json_load["steps"][0][0]["info"]["replay"]["params"])
        with st.expander("Env Params"):
            st.write(gt_env_params)

        results = extract_results(json_load, target_team_id, env_params, episode_id, cfg)

        ############################################################
        # フィルタ
        ############################################################
        st.subheader("フィルタ")
        step_idx = st.select_slider("ステップを選択", options=range(505))

        ############################################################
        # 状態マップの可視化
        ############################################################
        st.subheader("State")
        with st.expander("状態マップ"):
            active_states = st.multiselect(
                "表示する特徴量を選択",
                options=[State(i).name for i in range(len(State))],
                default=[State.TILE_TYPE.name, State.RELICS.name, State.POINTS.name, State.OPP_UNIT_COUNT.name],
            )
            state = results["state"][step_idx]
            gt_state = results["gt_state"][step_idx]
            global_state = results["global_state"][step_idx]
            obs = results["obs"][step_idx]
            col1, col2 = st.columns(2)
            with col1:
                visualize_state(state, State, active_states, title="State", n_cols=4)
            with col2:
                visualize_state(gt_state, State, active_states, title="GT State", n_cols=4)

            visualize_global_state(global_state)

        ############################################################
        # ポリシーマップの可視化
        ############################################################
        unit_positions = obs["units"]["position"][target_team_id]
        policy_map = results["policy_map"][step_idx]
        real_actions = results["action"][step_idx]
        st.subheader("Policy")
        cols = st.columns([1, 2])
        with cols[0]:
            visualize_unit_positions(unit_positions, target_team_id, title="Unit Positions")
        with cols[1]:
            visualize_policy(policy_map, real_actions, unit_positions, title="Policy", n_cols=8)


if __name__ == "__main__":
    main()
