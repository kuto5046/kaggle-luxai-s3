import random
from typing import Any
from pathlib import Path
from dataclasses import dataclass

import h5py
import numpy as np
import torch
import polars as pl
import torch.nn.functional as F
from torch import nn, optim
from lightning import LightningModule, LightningDataModule
from torchvision import transforms
from torchmetrics import Accuracy, MetricCollection
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

import wandb

from .utils import State, Action, GlobalState, HiddenState, HiddenGlobalState, to_np
from .params import EnvParams


class LuxAugmentBase:
    def __init__(self) -> None:
        self.p = 0.5

    def switch_action(self, action: int, i: int, j: int) -> int:
        action = np.where(action == i, -1, action)
        action = np.where(action == j, i, action)
        action = np.where(action == -1, j, action)
        return action

    def rotate_action(self, action: int, offset: int = 0) -> int:
        # right(2)->up(1)
        action = np.where(action == 1 + offset, -1, action)
        action = np.where(action == 2 + offset, 1 + offset, action)

        # up(1) -> left(4)
        action = np.where(action == 4 + offset, -2, action)
        action = np.where(action == -1, 4 + offset, action)

        # left(4) -> down(3)
        action = np.where(action == 3 + offset, -1, action)
        action = np.where(action == -2, 3 + offset, action)

        # down(3) -> right(2)
        action = np.where(action == -1, 2 + offset, action)
        return action


# 自陣を(0,0)にする
class LuxAugmentStandardize(LuxAugmentBase):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # x,yが実際のmapと行列で異なるので操作を直感的にするために転置後に処理する
        state = inputs["state"].copy()
        hidden_state = inputs["hidden_state"].copy()
        action = inputs["action"].copy()
        sap = inputs["sap"].copy()

        # 原点を自陣とする
        # TODO agent_id を用いて自陣を判定する
        visit_count = state[:, State.VISIT_COUNT]
        do_flip = np.sum(visit_count[:, 0, 0]) < np.sum(visit_count[:, -1, -1])

        if do_flip:
            # Flip vertically↑↓(# switch up(1) and down(3))
            # Flip horizontally →← (switch left(2) and right(4))
            state = np.flip(state, axis=(2, 3)).copy()
            hidden_state = np.flip(hidden_state, axis=(1, 2)).copy()
            action = np.flip(action, axis=(0, 1)).copy()
            action = self.switch_action(action, Action.UP, Action.DOWN)
            action = self.switch_action(action, Action.LEFT, Action.RIGHT)
            sap = np.flip(sap, axis=(0, 1)).copy()

        inputs["state"] = state
        inputs["hidden_state"] = hidden_state
        inputs["action"] = action
        inputs["sap"] = sap
        return inputs


class LuxAugmentTranspose(LuxAugmentBase):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # x,yが実際のmapと行列で異なるので操作を直感的にするために転置後に処理する
        state = inputs["state"].copy()
        hidden_state = inputs["hidden_state"].copy()
        action = inputs["action"].copy()
        sap = inputs["sap"].copy()

        if random.random() < self.p:
            state = np.transpose(state, (0, 1, 3, 2)).copy()
            hidden_state = np.transpose(hidden_state, (0, 2, 1)).copy()
            action = np.transpose(action, (1, 0)).copy()
            action = self.switch_action(action, Action.UP, Action.LEFT)
            action = self.switch_action(action, Action.DOWN, Action.RIGHT)
            sap = np.transpose(sap, (1, 0)).copy()

        inputs["state"] = state
        inputs["hidden_state"] = hidden_state
        inputs["action"] = action
        inputs["sap"] = sap
        return inputs


class LaxDataset(Dataset):
    def __init__(self, df: pl.DataFrame, cfg: dataclass, mode: str = "train") -> None:
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.ids = []
        for episode_id, max_step in df[["EpisodeId", "MaxStep"]].to_numpy():
            for step_idx in range(1, int(max_step)):  # step_idx=0は初期状態なのでスキップ
                self.ids.append((episode_id, step_idx))
        self.h5_file = h5py.File(self.cfg.feature_dir / "episodes.h5", "r")
        self.transform_standardize = transforms.Compose([LuxAugmentStandardize()])
        self.transform = transforms.Compose([LuxAugmentTranspose()])
        self.aug = cfg.aug

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        episode_id, step_idx = self.ids[idx]
        states = []
        global_states = []
        for i in range(self.cfg.n_stack - 1, -1, -1):
            if step_idx - i >= 0:
                state = np.array(self.h5_file[str(episode_id)]["states"][str(step_idx - i)]).astype(np.float32)
                global_state = np.array(self.h5_file[str(episode_id)]["global_states"][str(step_idx - i)]).astype(
                    np.float32
                )
            else:
                state = np.zeros((len(State), EnvParams.map_height, EnvParams.map_width), dtype=np.float32)
                global_state = np.zeros(len(GlobalState), dtype=np.float32)
            states.append(state)
            global_states.append(global_state)
        state = np.stack(states, axis=0)  # (n_stack, channel, x, y)
        global_state = np.stack(global_states, axis=0)  # (n_stack, channel)

        hidden_state = np.array(self.h5_file[str(episode_id)]["hidden_states"][str(step_idx)]).astype(np.float32)
        hidden_global_state = np.array(self.h5_file[str(episode_id)]["hidden_global_states"][str(step_idx)]).astype(
            np.float32
        )
        actions = np.array(self.h5_file[str(episode_id)]["actions"][str(step_idx)]).astype(np.float32)
        action = actions[0]
        sap = actions[1]
        win = np.array(self.h5_file[str(episode_id)]["win"][str(step_idx)]).astype(np.float32)
        inputs = {
            "state": state,
            "global_state": global_state,
            "hidden_state": hidden_state,
            "hidden_global_state": hidden_global_state,
            "action": action,
            "sap": sap,
            "win": win,
            "turn": step_idx,
        }
        inputs = self.transform_standardize(inputs)
        if self.mode == "train" and self.aug:
            inputs = self.transform(inputs)

        return inputs


class LaxLitDataModule(LightningDataModule):
    def __init__(self, cfg: dataclass):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str | None = None) -> None:
        df = pl.read_csv(self.cfg.feature_dir / "train.csv")
        df = df.filter(pl.col("Win"))  # 勝利したエピソードのみを使用
        train = df.filter(pl.col("fold") != self.cfg.use_fold)
        valid = df.filter(pl.col("fold") == self.cfg.use_fold)
        self.train_dataset = LaxDataset(train, self.cfg, mode="train")
        self.valid_dataset = LaxDataset(valid, self.cfg, mode="valid")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )


class LaxLitModel(LightningModule):
    def __init__(self, cfg: dataclass) -> None:
        super().__init__()
        self.cfg = cfg
        self.output_dir = self.cfg.output_dir
        self.model = LuxUNetModel(
            state_space_size=len(State),
            global_state_space_size=len(GlobalState),
            action_space_size=len(Action),
            hidden_state_space_size=len(HiddenState),
            n_stack=cfg.n_stack,
            res=cfg.res,
            num_turn_groups=EnvParams.match_count_per_episode * 2,
        )
        self.criterion1 = DiceLoss(n_classes=len(Action))
        # self.criterion1 = MaskedBCEWithLogitsLoss()
        self.criterion2 = nn.BCEWithLogitsLoss()
        self.criterion3 = nn.MSELoss()
        self.criterion4 = MaskedFocalLoss()

        metrics = self.get_metrics()
        self.train_metrics = metrics.clone(postfix="/train")
        self.valid_metrics = metrics.clone(postfix="/valid")
        self.valid_outputs = {"ground_truth": [], "predictions": []}

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(batch)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._share_step(batch, mode="train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._share_step(batch, mode="valid")

    def _share_step(self, batch: Any, mode: str = "train") -> torch.Tensor:
        outputs = self(batch)

        policy_preds = torch.softmax(outputs["policy"], dim=1)
        policy_targets = one_hot_encoder(batch["action"], n_classes=len(Action))
        # policy_mask = (batch["state"][:, -1, State.OWN_UNIT_COUNT] > 0).unsqueeze(1)  # (batch_size, 1, w, h)
        # policy_loss = self.criterion1(policy_preds, policy_targets, policy_mask)
        policy_loss = self.criterion1(policy_preds, policy_targets)

        # value_loss = self.criterion2(outputs["value"].flatten(), batch["win"])
        state_loss = self.criterion3(outputs["state"].flatten(), batch["hidden_state"].flatten())
        global_state_loss = self.criterion3(outputs["global_state"].flatten(), batch["hidden_global_state"].flatten())

        # sap_available_mask = (batch["state"][:, -1, State.SAP_AVAILABLE_AREA] > 0)  # (batch_size, w, h)
        # sap_loss = self.criterion4(outputs["sap"].squeeze(1), batch["sap"], sap_available_mask)
        loss = (
            policy_loss * self.cfg.loss_weight_policy
            + state_loss * self.cfg.loss_weight_state
            # + value_loss * self.cfg.loss_weight_value
            + global_state_loss * self.cfg.loss_weight_global_state
            # + sap_loss * self.cfg.loss_weight_sap
        )

        self.log(
            f"PolicyLoss/{mode}",
            policy_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        # self.log(
        #     f"SapLoss/{mode}",
        #     sap_loss,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=False,
        #     logger=True,
        # )
        # self.log(
        #     f"ValueLoss/{mode}",
        #     value_loss,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=False,
        #     logger=True,
        # )
        self.log(
            f"StateLoss/{mode}",
            state_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        self.log(
            f"GlobalStateLoss/{mode}",
            global_state_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f"Loss/{mode}",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        preds = torch.softmax(outputs["policy"], dim=1).argmax(dim=1).flatten()
        gts = batch["action"].flatten()
        unit_masks = (batch["state"][:, -1, State.OWN_UNIT_COUNT] > 0).flatten()  # unitが存在するところだけで計算する

        preds = preds[unit_masks]
        gts = gts[unit_masks]
        if mode == "train":
            self.train_metrics.update(preds, gts)
        else:
            self.valid_metrics.update(preds, gts)
            self.valid_outputs["ground_truth"].append(to_np(gts))
            self.valid_outputs["predictions"].append(to_np(preds))
        return loss

    def on_train_epoch_end(self) -> None:
        # スコア評価
        output = self.train_metrics.compute()
        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # メトリクスのリセット
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        # スコア評価
        output = self.valid_metrics.compute()
        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # best_valid_lossを更新した場合のみconfusion matrixをlogする
        if self.trainer.callback_metrics["Loss/valid"] < self.trainer.callback_metrics.get(
            "best_valid_loss", float("inf")
        ):
            # save_model(self.model, self.output_dir)
            self.trainer.callback_metrics["best_valid_loss"] = self.trainer.callback_metrics["Loss/valid"]
            wandb.log(
                {
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=np.concatenate(self.valid_outputs["ground_truth"]),
                        preds=np.concatenate(self.valid_outputs["predictions"]),
                        class_names=[action.name for action in Action],
                    )
                }
            )
        self.valid_outputs = {"ground_truth": [], "predictions": []}
        # メトリクスのリセット
        self.valid_metrics.reset()

    def configure_optimizers(self) -> optim.Optimizer | dict[str, Any] | None:
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
        if scheduler is None:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
            }

    def get_optimizer(self) -> optim.Optimizer:
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        return optimizer

    def get_scheduler(self, optimizer: optim.Optimizer) -> optim.lr_scheduler._LRScheduler | None:
        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(self.cfg.warmup_step_rate * num_training_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return scheduler

    def get_metrics(self) -> MetricCollection:
        return MetricCollection(
            [
                Accuracy(task="multiclass", num_classes=len(Action)),
            ]
        )


def one_hot_encoder(input_tensor: torch.Tensor, n_classes: int) -> torch.Tensor:
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i
        tensor_list.append(temp_prob.unsqueeze(1))
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None, res: bool = False) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.res = res
        # 入力と出力のチャンネル数が異なる場合のための1x1 convolution
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.res:
            return self.double_conv(x) + self.skip_conv(x)
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int, res: bool = False) -> None:
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, res=res)
        self.res = res
        # スキップコネクション用の1x1 convとダウンサンプリング
        if self.res:
            self.skip = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.AvgPool2d(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.maxpool(x)
        x1 = self.conv(x1)
        if self.res:
            return x1 + self.skip(x)
        return x1


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        x1 = F.pad(
            x1,
            [
                torch.div(diffX, 2, rounding_mode="floor"),
                diffX - torch.div(diffX, 2, rounding_mode="floor"),
                torch.div(diffY, 2, rounding_mode="floor"),
                diffY - torch.div(diffY, 2, rounding_mode="floor"),
            ],
        )
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LuxUNetModel(nn.Module):
    def __init__(
        self,
        state_space_size: int,  # 例: len(State)
        global_state_space_size: int,
        action_space_size: int,
        hidden_state_space_size: int,
        n_stack: int,
        num_turn_groups: int,  # 追加：ターンのグループ数（例：10など）
        bilinear: bool = True,
        res: bool = False,
    ) -> None:
        super().__init__()
        self.bilinear = bilinear
        self.n_stack = n_stack
        self.num_turn_groups = num_turn_groups

        # もともとの state に加え、"turn" の one-hot 分のチャネルを追加する
        in_channels = state_space_size + self.num_turn_groups
        self.inc = DoubleConv(in_channels, 64, res=res)
        self.down1 = Down(64, 128, res=res)
        self.down2 = Down(128, 256, res=res)
        self.down3 = Down(256, 256, res=res)

        #
        factor = 2 if bilinear else 1
        self.up1 = Up(256 * 2 + global_state_space_size, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.policy_net = OutConv(64 * n_stack, action_space_size)
        # self.sap_net = OutConv(64 * n_stack, 1)
        self.state_net = OutConv(64 * n_stack, hidden_state_space_size)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.value_net = nn.Sequential(
        #     nn.Linear((256 + global_state_space_size) * n_stack, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1),
        # )
        self.global_state_net = nn.Sequential(
            nn.Linear((256 + global_state_space_size) * n_stack, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(HiddenGlobalState)),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state = batch["state"]
        global_state = batch["global_state"]
        _n, _t, _c, _x, _y = state.shape

        # one-hot 化してブロードキャスト
        # batch["turn"] をテンソルに変換（shape: (_n,)）
        turn = batch["turn"].to(state.device).long()  # 例: step_idx
        # ターングループ = turn // 50
        # 0-50, 51-100, 101-151, ... 455-504 の 10 グループに分ける
        turn_group = turn // 101 * 2 + (turn % 101) // 51
        # assert turn >= 0 and turn < 505
        # one-hot 化：shape (_n, num_turn_groups)
        one_hot_turn = F.one_hot(turn_group, num_classes=self.num_turn_groups).float()
        # ここで、one_hot_turn は各サンプルの情報なので、ここで時間軸はstackしない
        # そのため、1枚分の盤面全体にブロードキャストする:
        one_hot_turn = one_hot_turn.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # shape: (_n, 1, num_turn_groups, 1, 1)
        one_hot_turn = one_hot_turn.expand(_n, _t, self.num_turn_groups, _x, _y)  # replicate for each time slice
        # state と連結
        state = torch.cat([state, one_hot_turn], dim=2)
        # 連結後のチャンネル数は state_space_size + num_turn_groups

        # ここで state の shape は (_n, n_stack, state_space_size + num_turn_groups, _x, _y)
        x_in = state.view(_n * _t, state.shape[2], _x, _y)
        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # sx, syのマップにグローバルステートをブロードキャスト
        sx, sy = x4.shape[2:]
        _n2, _t2, _c_global = global_state.shape
        gx = global_state.view(-1, _c_global, 1, 1)
        gx = gx.repeat(1, 1, sx, sy)

        x4 = torch.cat([x4, gx], dim=1)
        x = self.global_avg_pool(x4).view(_n, -1)
        # value_logits = self.value_net(x)
        global_state_logits = self.global_state_net(x)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        x = x.view(_n, -1, _x, _y)
        policy_logits = self.policy_net(x)
        # sap_logits = self.sap_net(x)
        state_logits = self.state_net(x)

        return {
            "policy": policy_logits,
            # "sap": sap_logits,
            "state": state_logits,
            "global_state": global_state_logits,
            # "value": value_logits,
        }


def save_model(model, output_dir: Path, latest: bool = False):
    if latest:
        torch.save(model.state_dict(), output_dir / "latest_model.pth")
    else:
        torch.save(model.state_dict(), output_dir / "best_model.pth")


class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_loss = self.bce(logits, targets) * mask
        return masked_loss.sum() / mask.sum()


class DiceLoss(nn.Module):
    def __init__(self, n_classes: int, weights: None | list[float] = None) -> None:
        super().__init__()
        self.n_classes = n_classes
        if weights is None:
            self.weights = torch.ones(n_classes)
        else:
            self.weights = torch.tensor(weights)

    def _dice_loss(self, score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # If binary classification, the output tensors might have a single channel.
        # In that case, we squeeze the channel dimension and calculate the binary dice loss.
        if self.n_classes == 1 or (inputs.ndim > 1 and inputs.size(1) == 1):
            # inputs = inputs.squeeze(1)
            # Squeeze the target too if needed. If the target has an extra channel dimension, remove it.
            # target = target.squeeze(1) if (target.ndim > 1 and target.size(1) == 1) else target
            dice = self._dice_loss(inputs, target)
            # If weights are provided, apply the weight of the single channel.
            return dice * self.weights[0] / torch.sum(self.weights)

        # assert inputs.size() == target.size(), f"predict {inputs.size()} & target {target.size()} shape do not match"
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * self.weights[i]  # Apply the class weight
        return loss / torch.sum(self.weights)


class MaskedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets, mask):
        bce_loss = self.bce(logits, targets)
        pt = torch.exp(-bce_loss)  # 確率の補正
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        masked_focal_loss = focal_loss * mask
        return masked_focal_loss.sum() / mask.sum()
