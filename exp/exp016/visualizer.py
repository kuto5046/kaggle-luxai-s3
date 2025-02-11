from pathlib import Path

import h5py
import numpy as np
import torch
import polars as pl
import streamlit as st
import plotly.graph_objects as go
from lux.utils import State, Action, GlobalState, HiddenState, HiddenGlobalState, to_np
from lux.models import LuxUNetModel
from lux.params import EnvParams

# ページ設定
st.set_page_config(layout="wide")


# データ表示
def visualize_state(state, state_enum, title="State", n_cols: int = 5):
    st.subheader(title)
    # 全てのチャンネルを可視化する

    # ヒートマップ表示
    num_channels = state.shape[0]
    cols = st.columns(n_cols)
    for i in range(num_channels):
        with cols[i % n_cols]:
            fig = go.Figure(data=go.Heatmap(z=state[i], colorscale="greens"))
            fig.update_layout(title=f"{state_enum(i).name}", width=300, height=300)
            st.plotly_chart(fig, key=f"{title}_{i}")


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


@st.cache_data()
def load_model(exp_name: str, n_stack: int) -> LuxUNetModel | None:
    checkpoint_path = Path(f"/home/user/work/exp/{exp_name}/output/best_model.ckpt")
    if not checkpoint_path.exists():
        return None

    model = LuxUNetModel(
        state_space_size=len(State),
        global_state_space_size=len(GlobalState),
        action_space_size=len(Action),
        hidden_state_space_size=len(HiddenState),
        hidden_global_state_space_size=len(HiddenGlobalState),
        n_stack=n_stack,
        res=True,
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
    episode_id = str(st.selectbox("エピソードを選択", episode_ids))

    n_stack = 4
    model = load_model(exp_name, n_stack=n_stack)
    if episode_id:
        link = f"https://s3vis.lux-ai.org/#/visualizer?input={episode_id}"
        st.info(f"[Lux AI Visualizer]({link})")

        episode_data = h5_file[episode_id]
        # ステップ数取得
        steps = sorted([int(step) for step in episode_data["states"].keys()])
        # ステップ選択
        step_idx = st.select_slider("ステップを選択", options=steps)
        if step_idx is not None:
            # データ取得
            actions = np.array(h5_file[episode_id]["actions"][str(step_idx)])
            own_action = actions[0]
            opp_action = actions[1]
            states = []
            global_states = []
            for i in range(n_stack - 1, -1, -1):
                if step_idx - i >= 0:
                    state = np.array(h5_file[episode_id]["states"][str(step_idx - i)]).astype(np.float32)
                    global_state = np.array(h5_file[episode_id]["global_states"][str(step_idx - i)]).astype(np.float32)
                else:
                    state = np.zeros((len(State), EnvParams.map_height, EnvParams.map_width), dtype=np.float32)
                    global_state = np.zeros((len(GlobalState)), dtype=np.float32)
                states.append(state)
                global_states.append(global_state)
            state = np.stack(states, axis=0)
            global_state = np.stack(global_states, axis=0)
            hidden_state = np.array(h5_file[episode_id]["hidden_states"][str(step_idx)])
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                visualize_action(own_action, title="Own Action")
                visualize_action(opp_action, title="Opp Action")

            with col2:
                if model is not None:
                    torch_states = {
                        "state": torch.tensor(state).unsqueeze(0).float(),
                        "global_state": torch.tensor(global_state).unsqueeze(0).float(),
                    }
                    with torch.no_grad():
                        output = model(torch_states)
                        pred_own_action = to_np(output["own_policy"].argmax(dim=1).cpu().squeeze())
                        pred_opp_action = to_np(output["opp_policy"].argmax(dim=1).cpu().squeeze())
                        pred_state = to_np(output["state"].cpu().squeeze())

                    visualize_pred_action(pred_own_action, title="Predict Own Action")
                    visualize_pred_action(pred_opp_action, title="Predict Opp Action")

            with col3:
                last_state = states[-1]
                visualize_state(last_state, State, title="State")
                visualize_state(hidden_state, HiddenState, title="Hidden State")
                visualize_state(pred_state, HiddenState, title="Predict State")
    # h5_file.close()


if __name__ == "__main__":
    main()
