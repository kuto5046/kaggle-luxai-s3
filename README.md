# kaggle-template
kaggleコンペ用のテンプレートレポジトリ

## 環境構築
dockerで環境構築を行う。
```bash
docker compose up -d --build
```
あとはvscodeのdevcontainerでコンテナに入って作業する

## dataset準備
docker内だとkaggle APIが有効になっている

dataset一覧チェック
```
kaggle datasets list
```

datasetをdownload
```bash
cd input
kaggle datasets download <DATASET_NAME>
```
解凍(同じ名前のディレクトリを作成してその中に解凍)
```bash
unzip <DATASET_NAME>.zip -d <DATASET_NAME>
```

## 初めにすること
pre-commitをinstall
```sh
uv run pre-commit install
```

wandbのprojectをwebから作成
ターミナルで以下を実行
```sh
wandb login
```
authorizeすることでwandbが利用可能になる
