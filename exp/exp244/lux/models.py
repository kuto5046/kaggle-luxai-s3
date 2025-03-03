import sys
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
from .convlstm import ConvLSTM


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
        sap_count = inputs["sap_count"].copy()

        # 原点を自陣とする
        # TODO agent_id を用いて自陣を判定する
        visit_count = state[:, State.VISIT_COUNT]
        do_flip = np.sum(visit_count[:, 0, 0]) < np.sum(visit_count[:, -1, -1])

        if do_flip:
            # Flip vertically↑↓(# switch up(1) and down(3))
            # Flip horizontally →← (switch left(2) and right(4))
            state = np.flip(state, axis=(2, 3)).copy()
            hidden_state = np.flip(hidden_state, axis=(2, 3)).copy()
            action = np.flip(action, axis=(1, 2)).copy()
            action = self.switch_action(action, Action.UP, Action.DOWN)
            action = self.switch_action(action, Action.LEFT, Action.RIGHT)
            sap = np.flip(sap, axis=(1, 2)).copy()
            sap_count = np.flip(sap_count, axis=(1, 2)).copy()

        inputs["state"] = state
        inputs["hidden_state"] = hidden_state
        inputs["action"] = action
        inputs["sap"] = sap
        inputs["sap_count"] = sap_count
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
        sap_count = inputs["sap_count"].copy()

        if random.random() < self.p:
            state = np.transpose(state, (0, 1, 3, 2)).copy()
            hidden_state = np.transpose(hidden_state, (0, 1, 3, 2)).copy()
            action = np.transpose(action, (0, 2, 1)).copy()
            action = self.switch_action(action, Action.UP, Action.LEFT)
            action = self.switch_action(action, Action.DOWN, Action.RIGHT)
            sap = np.transpose(sap, (0, 2, 1)).copy()
            sap_count = np.transpose(sap_count, (0, 2, 1)).copy()

        inputs["state"] = state
        inputs["hidden_state"] = hidden_state
        inputs["action"] = action
        inputs["sap"] = sap
        inputs["sap_count"] = sap_count
        return inputs


class LaxDataset(Dataset):
    def __init__(self, df: pl.DataFrame, cfg: dataclass, mode: str = "train") -> None:
        super().__init__()
        if cfg.n_stack != 505:
            raise NotImplementedError("n_stack must be 505 when training LSTM!")
        self.cfg = cfg
        self.mode = mode
        self.ids = []
        # self.n_match = 101
        self.n_match = 16
        # self.n_match = 1
        for episode_id, max_step in df[["EpisodeId", "MaxStep"]].to_numpy():
            if max_step != 505:
                raise NotImplementedError("max_step must be 505 when training LSTM!")
            # for step in range(0, max_step):
            #     self.ids.append((episode_id, step))
            # for match in range(0, 505, self.n_match):
            #     self.ids.append((episode_id, match))
            for game in range(0, 505, 101):
                for match in range(0, 101, self.n_match):
                    if match + self.n_match > 101:
                        continue
                    match_begin = game + match
                    match_end = match_begin + self.n_match
                    self.ids.append((episode_id, match_begin, match_end))

        self.h5_file = h5py.File(self.cfg.feature_dir / "episodes.h5", "r")
        self.transform_standardize = transforms.Compose([LuxAugmentStandardize()])
        self.transform = transforms.Compose([LuxAugmentTranspose()])
        self.aug = cfg.aug

    def __len__(self) -> int:
        # return 100
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        # episode_id, step = self.ids[idx]
        episode_id, match_begin, match_end = self.ids[idx]
        states = []
        global_states = []
        hidden_states = []
        hidden_global_states = []
        actions = []
        saps = []
        sap_counts = []
        wins = []
        # for i in range(self.cfg.n_stack):
        # for i in [step]:
        for i in range(match_begin, match_end):
            state = np.array(self.h5_file[str(episode_id)]["states"][str(i)]).astype(np.float32)
            global_state = np.array(self.h5_file[str(episode_id)]["global_states"][str(i)]).astype(np.float32)
            hidden_state = np.array(self.h5_file[str(episode_id)]["hidden_states"][str(i)]).astype(np.float32)
            hidden_global_state = np.array(self.h5_file[str(episode_id)]["hidden_global_states"][str(i)]).astype(
                np.float32
            )
            action, sap, sap_count = np.array(self.h5_file[str(episode_id)]["actions"][str(i)]).astype(np.float32)
            win = np.array(self.h5_file[str(episode_id)]["win"][str(i)]).astype(np.float32)

            states.append(state)
            global_states.append(global_state)
            hidden_states.append(hidden_state)
            hidden_global_states.append(hidden_global_state)
            actions.append(action)
            saps.append(sap)
            sap_counts.append(sap_count)
            wins.append(win)

        states = np.stack(states, axis=0)  # (n_stack, channel, x, y)
        global_states = np.stack(global_states, axis=0)  # (n_stack, channel)
        hidden_states = np.stack(hidden_states, axis=0)  # (n_stack, channel, x, y)
        hidden_global_states = np.stack(hidden_global_states, axis=0)  # (n_stack, channel)
        actions = np.stack(actions, axis=0)  # (n_stack, x, y)
        saps = np.stack(saps, axis=0)  # (n_stack, x, y)
        sap_counts = np.stack(sap_counts, axis=0)  # (n_stack, x, y)
        wins = np.stack(wins, axis=0)  # (n_stack,)

        inputs = {
            "state": states,
            "global_state": global_states,
            "hidden_state": hidden_states,
            "hidden_global_state": hidden_global_states,
            "action": actions,
            "sap": saps,
            "sap_count": sap_counts,
            "win": wins,
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
        self.model = LuxLSTMModel(
            state_space_size=len(State),
            global_state_space_size=len(GlobalState),
            action_space_size=len(Action),
            n_stack=cfg.n_stack,
            res=cfg.res,
        )
        self.criterion1 = DiceLoss(n_classes=len(Action))
        # self.criterion1 = MaskedBCEWithLogitsLoss()
        self.criterion2 = nn.BCEWithLogitsLoss()
        self.criterion3 = nn.MSELoss()
        self.criterion4 = MaskedFocalTverskyLoss(
            alpha=0.3, beta=0.7, gamma=1.0, smooth=1e-3
        )  # sapを行わない背景が多数で学習が進まない問題を解決するための損失関数

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

        policy_preds = torch.softmax(outputs["policy"].flatten(0, 1), dim=1)
        policy_targets = one_hot_encoder(batch["action"].flatten(0, 1), n_classes=len(Action))
        policy_loss = self.criterion1(policy_preds, policy_targets)

        sap_available_mask = batch["state"][:, :, State.SAP_AVAILABLE_AREA] > 0  # (batch_size, n_stack, w, h)
        sap_output = outputs["sap"].squeeze(2)  # shape: (batch, n_stack, H, W)
        sap_output = sap_output.flatten(0, 1)  # shape: (batch_size * n_stack, H, W)
        # 各サンプルごとに空間軸 (H, W) の和を計算し、sap が行われているか判定
        sap_present_mask = batch["sap"].view(sap_output.shape[0], -1).sum(dim=1) > 0  # (batch_size * n_stack)

        sap_available_mask = sap_available_mask.flatten(0, 1)  # (batch_size * n_stack, w, h)

        if sap_present_mask.sum() > 0:
            # sap_loss は、sap が存在するサンプルのみで計算
            # 背景の部分が多すぎて学習が進みにくいので、sap_available_mask を使って背景をマスクする
            sap_loss = self.criterion4(
                sap_output[sap_present_mask],
                batch["sap"].flatten(0, 1)[sap_present_mask],
                sap_available_mask[sap_present_mask],
            )
        else:
            sap_loss = 0
        loss = (
            policy_loss * self.cfg.loss_weight_policy
            # + state_loss * self.cfg.loss_weight_state
            # + value_loss * self.cfg.loss_weight_value
            # + global_state_loss * self.cfg.loss_weight_global_state
            + sap_loss * self.cfg.loss_weight_sap
        )

        self.log(
            f"PolicyLoss/{mode}",
            policy_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f"SapLoss/{mode}",
            sap_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        # self.log(
        #     f"ValueLoss/{mode}",
        #     value_loss,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=False,
        #     logger=True,
        # )
        # self.log(
        #     f"StateLoss/{mode}",
        #     state_loss,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=False,
        #     logger=True,
        # )

        # self.log(
        #     f"GlobalStateLoss/{mode}",
        #     global_state_loss,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=False,
        #     logger=True,
        # )
        self.log(
            f"Loss/{mode}",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        preds = torch.softmax(outputs["policy"].flatten(0, 1), dim=1).argmax(dim=1).flatten()
        gts = batch["action"].flatten()
        unit_masks = (batch["state"][:, :, State.OWN_UNIT_COUNT] > 0).flatten()  # unitが存在するところだけで計算する

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

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        res: bool = False,
        kernel_size: int = 3,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        padding = kernel_size // 2
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=not batch_norm),
            nn.BatchNorm2d(mid_channels) if batch_norm is not None else nn.Identity(),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=not batch_norm),
            nn.BatchNorm2d(out_channels) if batch_norm is not None else nn.Identity(),
            nn.LeakyReLU(inplace=True),
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


def _to_tuple(x):
    if isinstance(x, int):
        return (x, x)
    return x


class OutConvWithNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return x


class LuxLSTMModel(nn.Module):
    def __init__(
        self,
        state_space_size: int,
        global_state_space_size: int,
        action_space_size: int,
        n_stack: int,
        num_layers: int = 14,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        return_hidden: bool = False,
        bilinear: bool = True,
        res: bool = True,
    ) -> None:
        super().__init__()
        self.bilinear = bilinear

        self.return_hidden = return_hidden

        self.hidden_channels = hidden_channels

        # TODO 引数で設定できるようにする
        # embed 部分の設定
        self.embed_configs = [
            {
                "in_channels": state_space_size + global_state_space_size,
                "out_channels": self.hidden_channels,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "use_relu": True,
            },
            {
                "in_channels": self.hidden_channels,
                "out_channels": self.hidden_channels,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "use_relu": True,
            },
        ]
        # recurrent_config の設定に repeats_per_step を追加
        self.recurrent_config = {
            "input_channels": self.hidden_channels,  # embed 層出力のチャネル数
            "hidden_channels": self.hidden_channels,
            "kernel_size": 5,
            "n_recurrent": 16,  # lstmのブロック数 論文中のD
            "repeats_per_step": 1,  # 1時刻あたりの内部更新回数 論文中のN
            "pool_and_inject": "horizontal",
            "pool_projection": "per-channel",
            "output_activation": "tanh",
            "forget_bias": -0.5,
            "fence_pad": "no",
            "residual": True,
            "skip_final": False,
        }

        self.convlstm = ConvLSTM(
            self.embed_configs,
            self.recurrent_config,
        )

        self.base_output_channels = self.hidden_channels

        self.sap_net1 = ResidualBlock(
            self.base_output_channels, 64, EnvParams.map_width, EnvParams.map_width, squeeze_excitation=False
        )
        self.sap_net2 = ResidualBlock(64, 64, EnvParams.map_width, EnvParams.map_width, squeeze_excitation=False)
        self.sap_net3 = OutConvWithNorm(64, 1)  # かなり極端な値を出力するので正規化することで学習を安定化させる
        # sap候補位置をpolicyの特徴マップに統合. sap rangeの最大値が7なのでkernel_size=15にしている
        self.policy_net1_from_sap = ResidualBlock(
            1, 16, EnvParams.map_width, EnvParams.map_width, kernel_size=15, squeeze_excitation=False
        )
        self.concat_norm = nn.BatchNorm2d(self.base_output_channels + 16)
        self.policy_net2 = ResidualBlock(
            self.base_output_channels + 16, 64, EnvParams.map_width, EnvParams.map_width, squeeze_excitation=False
        )
        self.policy_net3 = ResidualBlock(64, 64, EnvParams.map_width, EnvParams.map_width, squeeze_excitation=False)
        self.policy_net4 = OutConv(64, action_space_size)  # ここでWithNormを使うとCenterが全部Sapと予測されてしまった。

    def forward(
        self, batch: dict[str, torch.Tensor], hidden: None | list[tuple[torch.Tensor, torch.Tensor]] = None
    ) -> dict[str, torch.Tensor]:
        state = batch["state"]
        global_state = batch["global_state"]
        # print(f"state: {state.shape}")
        _n, _t, _c, _x, _y = state.shape
        _ng, _tg, _cg = global_state.shape

        gx = global_state.view(_ng, _tg, _cg, 1, 1).expand(_ng, _tg, _cg, _x, _y)
        # print(f"global_state: {gx.shape}")
        x = torch.cat([state, gx], dim=2)

        # print(f"_n: {_n}, _t: {_t}, _c: {_c}, _x: {_x}, _y: {_y}", file=sys.stderr)

        # print(f"x: {x.shape}", file=sys.stderr)
        x = x.transpose(1, 0).contiguous()

        # print(f"x: {x.shape}", file=sys.stderr)
        x, hidden = self.convlstm(x, hidden)
        # print(f"x: {x.shape}", file=sys.stderr)
        # print(f"hidden: {len(hidden)}, {len(hidden[0])}, {hidden[0][0].shape}, {hidden[0][1].shape}", file=sys.stderr)
        if _t > 1:
            x = x.transpose(0, 1)
            # print(f"x: {x.shape}", file=sys.stderr)

            x = x.flatten(0, 1)
        # else:
        #     x = x.unsqueeze(0)

        # print(f"x: {x.shape}", file=sys.stderr)

        sap_logits1 = self.sap_net1(x)
        sap_logits2 = self.sap_net2(sap_logits1)
        sap_logits = self.sap_net3(sap_logits2)

        # sap_net の出力は logits のまま扱う(ここで sigmoid はかけない)
        policy_logits1 = self.policy_net1_from_sap(sap_logits)
        # x は [N, base_output_channels, H, W]、policy_logits1 は [N, 16, H, W] なので連結後のチャネル数は base_output_channels+16
        policy_features = torch.cat([x, policy_logits1], dim=1)
        # 連結後に正規化を適用
        policy_features = self.concat_norm(policy_features)
        policy_logits = self.policy_net2(policy_features)
        policy_logits = self.policy_net3(policy_logits)
        policy_logits = self.policy_net4(policy_logits)

        policy_logits = policy_logits.view(_n, _t, -1, _x, _y)
        sap_logits = sap_logits.view(_n, _t, -1, _x, _y)
        assert policy_logits.shape == (state.shape[0], state.shape[1], len(Action), state.shape[3], state.shape[4])
        assert sap_logits.shape == (state.shape[0], state.shape[1], 1, state.shape[3], state.shape[4])

        output = {
            "policy": policy_logits,
            "sap": sap_logits,
            # "state": state_logits,
            # "global_state": global_state_logits,
            # "value": value_logits,
        }
        if self.return_hidden:
            return output, hidden
        else:
            return output


class MaskedFocalTverskyLoss(nn.Module):
    def __init__(
        self, alpha: float = 0.5, beta: float = 0.5, gamma: float = 1.0, smooth: float = 1e-6, reduction: str = "mean"
    ):
        """
        Focal Tversky Loss with mask support.
        :param alpha: False Positive に対する重み (通常 0.5)
        :param beta: False Negative に対する重み (通常 0.5)
        :param gamma: Focal項のパラメータ。gamma > 1 で難しい例に注目
        :param smooth: 数値安定性のためのスムージング項
        :param reduction: 'mean' もしくは 'sum'
        ※この損失関数は、モデルの出力として logitsd(シグモイド未適用値)を入力として受け取ります。
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: 予測値。logitsd(シグモイド未適用値)を想定。
                       形状は (batch, ...) であることを前提とする。
        :param targets: 教師ラベル。0/1のバイナリマスク。
        :param mask: 損失計算対象となる領域を示すバイナリマスク。inputs と同じ形状。
        :return: Focal Tversky Loss
        """
        # logits から確率値に変換
        inputs = torch.sigmoid(inputs)
        # 入力、ターゲット、mask を (batch, -1) にフラット化
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1).float()
        mask = mask.view(mask.size(0), -1).float()

        # マスクを考慮してTP, FP, FNを計算
        TP = (inputs * targets * mask).sum(dim=1)
        FP = (inputs * (1 - targets) * mask).sum(dim=1)
        FN = ((1 - inputs) * targets * mask).sum(dim=1)

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        focal_loss = (1 - Tversky) ** self.gamma

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class SELayer(nn.Module):
    def __init__(self, n_channels: int, reduction: int = 16):
        """
        Squeeze-and-Excitation (SE) Layer.
        Args:
            n_channels (int): 入力のチャネル数
            reduction (int): 圧縮率 (デフォルト: 16)
        """
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(n_channels // reduction, n_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 入力テンソル (B, C, H, W)

        Returns:
            torch.Tensor: チャネルごとのスケーリングを適用した出力テンソル
        """
        b, c, _, _ = x.shape

        # グローバル平均プーリング (B, C, 1, 1)
        y = x.mean(dim=[2, 3], keepdim=True)

        # FC層を適用してチャネルごとの重みを学習 (B, C, 1, 1)
        y = self.fc(y.view(b, c)).view(b, c, 1, 1)

        # スケール適用
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        kernel_size: int = 3,
        normalize: bool = False,
        activation=nn.LeakyReLU,
        squeeze_excitation: bool = True,
        rescale_se_input: bool = True,
        **conv2d_kwargs,
    ):
        super().__init__()

        # Calculate "same" padding
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # https://www.wolframalpha.com/input/?i=i%3D%28i%2B2x-k-%28k-1%29%28d-1%29%2Fs%29+%2B+1&assumption=%22i%22+-%3E+%22Variable%22
        assert "padding" not in conv2d_kwargs.keys()
        k = kernel_size
        d = conv2d_kwargs.get("dilation", 1)
        s = conv2d_kwargs.get("stride", 1)
        padding = (k - 1) * (d + s - 1) / (2 * s)
        assert padding == int(padding), f"padding should be an integer, was {padding:.2f}"
        padding = int(padding)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(padding, padding),
            **conv2d_kwargs,
        )
        # We use LayerNorm here since the size of the input "images" may vary based on the board size
        self.norm1 = nn.LayerNorm([out_channels, height, width]) if normalize else nn.Identity()
        self.act1 = activation()

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(padding, padding),
            **conv2d_kwargs,
        )
        self.norm2 = nn.LayerNorm([out_channels, height, width]) if normalize else nn.Identity()
        self.final_act = activation()

        if in_channels != out_channels:
            self.change_n_channels = nn.Conv2d(in_channels, out_channels, (1, 1))
        else:
            self.change_n_channels = nn.Identity()

        if squeeze_excitation:
            self.squeeze_excitation = SELayer(out_channels, rescale_se_input)
        else:
            self.squeeze_excitation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.act1(self.norm1(x))
        x = self.conv2(x)
        x = self.squeeze_excitation(self.norm2(x))
        x = x + self.change_n_channels(identity)
        return self.final_act(x)


class LuxValueConvModel(nn.Module):
    def __init__(
        self,
        state_space_size: int,
        global_state_space_size: int,
        n_stack: int,
        bilinear: bool = True,
        res: bool = True,
    ) -> None:
        super().__init__()
        self.bilinear = bilinear

        self.inc = DoubleConv(state_space_size, 64, res=res)
        self.down1 = Down(64, 128, res=res)
        self.down2 = Down(128, 256, res=res)
        self.down3 = Down(256, 256, res=res)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.value_net = nn.Sequential(
            nn.Linear((256 + global_state_space_size) * n_stack, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state = batch["state"]
        global_state = batch["global_state"]
        _n, _t, _c, _x, _y = state.shape
        x = state.view(-1, _c, _x, _y)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # sx, syのマップにグローバルステートをブロードキャスト
        sx, sy = x4.shape[2:]
        _n, _t, _c = global_state.shape
        gx = global_state.view(-1, _c, 1, 1)
        gx = gx.repeat(1, 1, sx, sy)

        x4 = torch.cat([x4, gx], dim=1)
        x = self.global_avg_pool(x4).view(_n, -1)
        value_logits = self.value_net(x)

        return {
            "value": value_logits,
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
