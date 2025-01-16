from pathlib import Path

import h5py
import polars as pl
import streamlit as st
import plotly.graph_objects as go

# ページ設定
st.set_page_config(layout="wide")


def main():
    st.title("Episode Data Visualizer")

    # exp選択
    exp_name = st.selectbox(
        "実験を選択", options=["exp001", "exp002", "exp003", "exp004", "exp005", "exp006", "exp007"]
    )

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

    if selected_episode:
        episode_data = h5_file[selected_episode]
        # ステップ数取得
        steps = list(episode_data["states"].keys())
        st.write(f"Total Steps: {len(steps)}")

        # ステップ選択
        selected_step = st.select_slider("ステップを選択", options=steps)

        if selected_step:
            # データ取得
            state = h5_file[selected_episode]["states"][selected_step][:]
            # action = h5_file[selected_episode]["actions"][selected_step][:]

            # データ表示
            st.subheader("State")
            # 全てのチャンネルを可視化する

            # ヒートマップ表示
            num_channels = state.shape[0]
            cols = st.columns(6)
            for i in range(num_channels):
                with cols[i % 6]:
                    fig = go.Figure(data=go.Heatmap(z=state[i], zmid=0))
                    fig.update_layout(title=f"Channel {i}", width=400, height=400)
                    st.plotly_chart(fig)

            st.subheader("Action")


if __name__ == "__main__":
    main()
