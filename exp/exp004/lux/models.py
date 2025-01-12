import random
from typing import Any
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
from lux.utils import State, Action, HiddenState, to_np


class LuxAugment:
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

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # x,yが実際のmapと行列で異なるので操作を直感的にするために転置後に処理する
        state = inputs["state"].transpose((0, 2, 1))
        hidden_state = inputs["hidden_state"].transpose((0, 2, 1))
        action = inputs["action"].T.copy()

        # Flip vertically↑↓(# switch up(1) and down(3))
        if random.random() < self.p:
            state = np.flip(state, axis=1).copy()
            hidden_state = np.flip(hidden_state, axis=1).copy()
            action = np.flip(action, axis=0)
            action = self.switch_action(action, Action.UP.value, Action.DOWN.value)

        # Flip horizontally →← (switch left(2) and right(4))
        if random.random() < self.p:
            state = np.flip(state, axis=2).copy()
            hidden_state = np.flip(hidden_state, axis=2).copy()
            action = np.flip(action, axis=1)
            action = self.switch_action(action, Action.LEFT.value, Action.RIGHT.value)

        # Rotate 90 degrees ↑→ (right->up, up->left left->down down->right)
        if random.random() < self.p:
            state = np.rot90(state, axes=(1, 2)).copy()
            action = np.rot90(action, axes=(0, 1))
            action = self.rotate_action(action)

        # TODO:
        # mapをランダムにずらす
        # 試合のindexを入れ替える

        inputs["state"] = state.transpose((0, 2, 1))
        inputs["hidden_state"] = hidden_state.transpose((0, 2, 1))
        inputs["action"] = action.T.copy()
        return inputs


class LaxDataset(Dataset):
    def __init__(self, df: pl.DataFrame, cfg: dataclass, mode: str = "train") -> None:
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.ids = []
        for episode_id, max_step in df[["EpisodeId", "MaxStep"]].to_numpy():
            for step_idx in range(1, int(max_step)):  # step_idx=0は初期状態なのでスキップ
                self.ids.append((str(episode_id), str(step_idx)))
        self.h5_file = h5py.File(self.cfg.feature_dir / "episodes.h5", "r")
        self.transform = transforms.Compose([LuxAugment()])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        episode_id, step_idx = self.ids[idx]
        inputs = {
            "state": np.array(self.h5_file[episode_id]["states"][step_idx]).astype(np.float32),
            "hidden_state": np.array(self.h5_file[episode_id]["hidden_states"][step_idx]).astype(np.float32),
            "action": np.array(self.h5_file[episode_id]["actions"][step_idx]).astype(np.float32),
        }
        if self.mode == "train":
            inputs = self.transform(inputs)

        return inputs


class LaxLitDataModule(LightningDataModule):
    def __init__(self, cfg: dataclass):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str | None = None) -> None:
        df = pl.read_csv(self.cfg.feature_dir / "train.csv")
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
        self.model = LuxUNetModel(
            state_space_size=len(State), action_space_size=len(Action), hidden_state_space_size=len(HiddenState)
        )
        self.criterion1 = DiceLoss(n_classes=len(Action))
        self.criterion2 = DiceLoss(n_classes=2)

        metrics = self.get_metrics()
        self.train_metrics = metrics.clone(postfix="/train")
        self.valid_metrics = metrics.clone(postfix="/valid")
        self.valid_outputs = {"ground_truth": [], "predictions": []}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._share_step(batch, mode="train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._share_step(batch, mode="valid")

    def _share_step(self, batch: Any, mode: str = "train") -> torch.Tensor:
        states = batch["state"]
        hidden_states = batch["hidden_state"]
        actions = batch["action"]
        outputs = self(states)
        policy_logits = outputs["policy"]
        state_logits = outputs["state"]
        _value_logits = outputs["value"]

        policy_preds = torch.softmax(policy_logits, dim=1)
        policy_targets = one_hot_encoder(actions, n_classes=len(Action))
        loss1 = self.criterion1(policy_preds, policy_targets)

        state_preds = torch.sigmoid(state_logits)
        loss2 = self.criterion2(state_preds, hidden_states)
        loss = loss1 + loss2

        self.log(
            f"Loss1/{mode}",
            loss1,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f"Loss2/{mode}",
            loss2,
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

        preds = torch.softmax(policy_logits, dim=1).argmax(dim=1).flatten()
        gts = actions.flatten()
        unit_masks = (states[:, State.UNIT_COUNT.value] > 0).flatten()  # unitが存在するところだけで計算する

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
        # assert inputs.size() == target.size(), f"predict {inputs.size()} & target {target.size()} shape do not match"
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * self.weights[i]  # Apply the class weight
        return loss / torch.sum(self.weights)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


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
        self, state_space_size: int, action_space_size: int, hidden_state_space_size: int, bilinear: bool = True
    ) -> None:
        super().__init__()
        self.bilinear = bilinear

        self.inc = DoubleConv(state_space_size, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)
        factor = 2 if bilinear else 1
        self.up1 = Up(256 * 2, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.policy_net = OutConv(64, action_space_size)
        self.state_net = OutConv(64, hidden_state_space_size)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.value_net = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        _n, _c, _x, _y = state.shape
        x = state.view(-1, _c, _x, _y)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        x = x.view(_n, -1, _x, _y)
        policy_logits = self.policy_net(x)
        state_logits = self.state_net(x)
        x = self.global_avg_pool(x4).view(_n, -1)
        value_logits = self.value_net(x)
        return {"policy": policy_logits, "state": state_logits, "value": value_logits}
