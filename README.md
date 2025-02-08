# kaggle luxai season3
kaggleコンペ用のテンプレートレポジトリ

## 環境構築
dockerで環境構築を行う。
```bash
docker compose up -d --build
```
あとはvscodeのdevcontainerでコンテナに入って作業する

## dataset準備

datasetをdownload
```bash
cd input
kaggle competitions download -c
unzip lux-ai-season-3.zip -d lux-ai-season-3
```

## 初めにすること
pre-commitをinstall
```sh
uv sync
uv run pre-commit install
```

ツールをinstall
```sh
uv pip install -e Lux-Design-S3/src
```

wandbのprojectをwebから作成
ターミナルで以下を実行
```sh
wandb login
```
authorizeすることでwandbが利用可能になる

## luxai-s3の実行
```sh
uv run luxai-s3 Lux-Design-S3/kits/python/main.py exp/exp001/main.py --output replay.json
```

## rayのdebug
ちょっと面倒
1. ray start
```bash
ray start --head
```
実行するとnext stepsで指定すべき`ip:port`が表示される

2.vscodeのray debugger拡張機能をinstallしcluster設定
clusterは1で表示されたものを使う
```
172.19.0.2:6379
```

3. 以下のリンクのように初期設定とbreakpointをおいてファイルをターミナルで実行
https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html#create-a-ray-task
