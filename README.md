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
