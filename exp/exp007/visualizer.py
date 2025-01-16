from pathlib import Path

import h5py
import numpy as np
import torch
import polars as pl
import streamlit as st
import plotly.graph_objects as go
from lux.utils import State, Action, HiddenState, to_np
from lux.models import LuxUNetModel

# ページ設定
st.set_page_config(layout="wide")


# データ表示
def visualize_state(state, n_cols: int = 5):
    st.subheader("State")
    # 全てのチャンネルを可視化する

    # ヒートマップ表示
    num_channels = state.shape[0]
    cols = st.columns(n_cols)
    for i in range(num_channels):
        with cols[i % n_cols]:
            fig = go.Figure(data=go.Heatmap(z=state[i], colorscale="greens"))
            fig.update_layout(title=f"{State(i).name}", width=300, height=300)
            st.plotly_chart(fig)


def visualizer_pred_action(action):
    st.subheader("Predict Action")
    # ヒートマップ表示 sequentialではないdeiscreteな色を使う
    fig = go.Figure(data=go.Heatmap(z=action, zmax=len(Action) - 1, zmin=0, colorscale="blues"))
    fig.update_layout(width=400, height=400)
    st.plotly_chart(fig)


def visualize_action(action):
    st.subheader("Action")
    # ヒートマップ表示 sequentialではないdeiscreteな色を使う
    fig = go.Figure(data=go.Heatmap(z=action, zmax=len(Action) - 1, zmin=0, colorscale="reds"))
    fig.update_layout(width=400, height=400)
    st.plotly_chart(fig)


@st.cache_data()
def load_model(exp_name: str) -> LuxUNetModel:
    checkpoint_path = Path(f"/home/user/work/exp/{exp_name}/output/best_model.ckpt")
    model = LuxUNetModel(
        state_space_size=len(State), action_space_size=len(Action), hidden_state_space_size=len(HiddenState)
    )
    ckpt = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    exp_name = Path(__file__).parent.name
    st.title(f"Episode Data Visualizer in {exp_name}")

    # ファイルパス生成
    input_dir = Path(f"output/feature_store/{exp_name}")
    h5_file_path = input_dir / "episodes.h5"
    df_path = input_dir / "train.csv"

    # データ読み込み
    h5_file = h5py.File(h5_file_path, "r")
    df = pl.read_csv(df_path)
    episode_ids = df["EpisodeId"].to_list()
    # エピソード選択
    selected_episode = str(st.selectbox("エピソードを選択", episode_ids))

    model = load_model(exp_name)
    if selected_episode:
        link = f"https://s3vis.lux-ai.org/#/visualizer?input={selected_episode}"
        st.info(f"[Lux AI Visualizer]({link})")

        episode_data = h5_file[selected_episode]
        # ステップ数取得
        steps = sorted([int(step) for step in episode_data["states"].keys()])
        # ステップ選択
        selected_step = str(st.select_slider("ステップを選択", options=steps))
        if selected_step:
            # データ取得
            action = np.array(h5_file[selected_episode]["actions"][selected_step])
            state = np.array(h5_file[selected_episode]["states"][selected_step])
            torch_state = torch.tensor(state).unsqueeze(0).float()
            with torch.no_grad():
                output = model(torch_state)
                pred_action = to_np(output["policy"].argmax(dim=1).cpu().squeeze())

            col1, col2 = st.columns(2)
            with col1:
                visualize_action(action)
            with col2:
                visualizer_pred_action(pred_action)
            visualize_state(state)

    # h5_file.close()


if __name__ == "__main__":
    main()
