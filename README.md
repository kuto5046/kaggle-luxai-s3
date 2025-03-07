# kaggle luxai season3
lux ai s3用のレポジトリ

## 環境構築
dockerで環境構築を行う。
```bash
docker compose up -d --build
```
あとはvscodeのdevcontainerでコンテナに入って作業する

dockerを使わない場合はuvを使って環境構築
uvをインストールした上で以下を実行
```
uv sync
```

## 初めにすること
pre-commitをinstall
```sh
uv sync
uv run pre-commit install
```

wandbのprojectをwebから作成
ターミナルで以下を実行
```sh
wandb login
```
authorizeすることでwandbが利用可能になる

## kaggleからepisodeデータを取得する
①以下のnotebookで対象のsubmissionのepisode情報をcsvで取得してローカルにダウンロード
https://www.kaggle.com/code/kuto0633/fork-of-lux-ai-s3-download-episodes-from-meta-kagg

②以下を実行してepisodeのjsonファイルをローカルに取得する(3000件が2時間くらい)
```sh
uv run python ./src/downloader.py
```

③ jsonファイルを特徴量変換してh5ファイルに保存
```sh
uv run python exp/exp017/data_processor.py
```


## 実験ファイルの実行
expフォルダに前の実験の結果をコピーして次の実験を実施している。
```sh
exp/exp017/
├── main.py              # kaggle提供のファイル
├── agent.py             # subに必要なagentファイル
├── data_processor.py    # 模倣学習用の特徴量生成を行う
├── train.py             # 模倣学習
├── rl.py                # 強化学習(Rllib)
├── setup.py             # cppでの最小費用流をつかうのに必要なsetupを行う
├── visualizer.py        # 実験結果を視覚化する Streamlit アプリ
└── lux/                 # ここに必要なモジュールやクラスを格納している
```
以下のように実行する。必要に応じて設定ファイルを変更する。
```sh
uv run python exp/exp017/data_processor.py
uv run python exp/exp017/train.py
```
cppでのflowを用いて対戦を行うためには以下のコマンドでセットアップする必要がある
```sh
uv run python exp/best/setup.py build
uv run python exp/best/setup.py install
```


## 強化学習の実行
はじめにrayを起動する必要があります
```sh
ray start --head
```
上記を実行すると表示されるlocalhostのURL(ex;`127.0.0.1:8265`)を開くとrayのdashboardが見れます。
ここではcpuの利用状況などが見れます。

ray startを実行したら以下を実行して強化学習を行います。
```sh
uv run python exp/best/rl.py
```

## その他便利タスク
justをタスクランナーとして使用しています
justをインストールするとjustfileにあるタスクを簡単に実行できます

### 提出
```sh
just sub exp017
```

### visualizer
streamlitを使った特徴量や予測の可視化ができる
```sh
just vis exp017
```
<img width="1425" alt="image" src="https://github.com/user-attachments/assets/dbe8f79d-3296-452b-bb96-7be5c0b211ed" />

### 試合対戦
luxai-s3環境での対戦
(justファイルのコマンドを見るとわかるが事前に対戦相手を用意しておく必要がある)
```sh
just game exp017
```
