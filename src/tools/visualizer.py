import h5py
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ページ設定
st.set_page_config(layout="wide")


def load_episode_data(file_path):
    """HDF5ファイルからエピソードデータを読み込む"""
    with h5py.File(file_path, "r") as f:
        episode_ids = list(f.keys())
        return episode_ids, f


def main():
    st.title("Episode Data Visualizer")

    # ファイル選択
    file_path = st.text_input("HDF5ファイルパス", "output/feature_store/exp001/episodes.h5")

    # データ読み込み
    episode_ids, h5_file = load_episode_data(file_path)

    # エピソード選択
    selected_episode = st.selectbox("エピソードを選択", episode_ids)

    if selected_episode:
        # ステップ数取得
        steps = list(h5_file[selected_episode]["states"].keys())
        st.write(f"Total Steps: {len(steps)}")

        # ステップ選択
        selected_step = st.select_slider("ステップを選択", options=steps)

        if selected_step:
            # データ取得
            state = h5_file[selected_episode]["states"][selected_step][:]
            action = h5_file[selected_episode]["actions"][selected_step][:]

            # データ表示
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("State")

                # チャンネル情報
                channel_names = [
                    "OWN_UNIT_COUNT",
                    "OPPONENT_UNIT_COUNT",
                    "RESOURCE_COUNT",
                    "FACTORY_COUNT",
                    "UNIT_HEALTH",
                    "FACTORY_HEALTH",
                    "UNIT_ENERGY",
                    "FACTORY_ENERGY",
                ]

                # チャンネル選択
                selected_channels = st.multiselect(
                    "表示するチャンネルを選択(最大4つ)", options=channel_names, default=[channel_names[0]]
                )

                # カラーマップ選択
                cmap = st.selectbox(
                    "カラーマップを選択", options=["viridis", "plasma", "inferno", "magma", "cividis"], index=0
                )

                # ヒートマップ表示
                num_channels = min(4, len(selected_channels))  # 最大4チャンネルまで表示
                fig, axes = plt.subplots(1, num_channels, figsize=(6 * num_channels, 6))
                if num_channels == 1:
                    axes = [axes]

                for i, channel in enumerate(selected_channels[:4]):
                    channel_idx = channel_names.index(channel)
                    im = axes[i].imshow(state[channel_idx], cmap=cmap)
                    plt.colorbar(im, ax=axes[i])
                    axes[i].set_title(f"{channel} at Step {selected_step}")

                plt.tight_layout()
                st.pyplot(fig)

            with col2:
                st.subheader("Action")
                st.write(action)

                # Actionの分布
                st.subheader("Action Distribution")
                all_actions = np.array([h5_file[selected_episode]["actions"][step][:] for step in steps]).flatten()

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(all_actions, bins=20)
                ax.set_xlabel("Action")
                ax.set_ylabel("Count")
                st.pyplot(fig)


if __name__ == "__main__":
    main()
