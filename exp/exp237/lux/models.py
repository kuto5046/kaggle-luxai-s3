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
            hidden_state = np.flip(hidden_state, axis=(2, 3)).copy()
            action = np.flip(action, axis=(1, 2)).copy()
            action = self.switch_action(action, Action.UP, Action.DOWN)
            action = self.switch_action(action, Action.LEFT, Action.RIGHT)
            sap = np.flip(sap, axis=(1, 2)).copy()

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
            hidden_state = np.transpose(hidden_state, (0, 1, 3, 2)).copy()
            action = np.transpose(action, (0, 2, 1)).copy()
            action = self.switch_action(action, Action.UP, Action.LEFT)
            action = self.switch_action(action, Action.DOWN, Action.RIGHT)
            sap = np.transpose(sap, (0, 2, 1)).copy()

        inputs["state"] = state
        inputs["hidden_state"] = hidden_state
        inputs["action"] = action
        inputs["sap"] = sap
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
        self.n_match = 4
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
        # return 10
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
            action, sap = np.array(self.h5_file[str(episode_id)]["actions"][str(i)]).astype(np.float32)
            win = np.array(self.h5_file[str(episode_id)]["win"][str(i)]).astype(np.float32)

            states.append(state)
            global_states.append(global_state)
            hidden_states.append(hidden_state)
            hidden_global_states.append(hidden_global_state)
            actions.append(action)
            saps.append(sap)
            wins.append(win)

        states = np.stack(states, axis=0)  # (n_stack, channel, x, y)
        global_states = np.stack(global_states, axis=0)  # (n_stack, channel)
        hidden_states = np.stack(hidden_states, axis=0)  # (n_stack, channel, x, y)
        hidden_global_states = np.stack(hidden_global_states, axis=0)  # (n_stack, channel)
        actions = np.stack(actions, axis=0)  # (n_stack, x, y)
        saps = np.stack(saps, axis=0)  # (n_stack, x, y)
        wins = np.stack(wins, axis=0)  # (n_stack,)

        inputs = {
            "state": states,
            "global_state": global_states,
            "hidden_state": hidden_states,
            "hidden_global_state": hidden_global_states,
            "action": actions,
            "sap": saps,
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
            hidden_state_space_size=len(HiddenState),
            n_stack=cfg.n_stack,
            # res=cfg.res,
        )
        # self.model = torch.compile(self.model)
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

        output_policy = outputs["policy"]
        batch_action = batch["action"]

        # first 2 dims to 1
        # print(f"output_policy.shape: {output_policy.shape}")
        # print(f"batch_action.shape: {batch_action.shape}")
        output_policy = output_policy.flatten(0, 1)
        batch_action = batch_action.flatten(0, 1)
        # print(f"output_policy.shape: {output_policy.shape}")
        # print(f"batch_action.shape: {batch_action.shape}")

        policy_preds = torch.softmax(output_policy, dim=1)
        policy_targets = one_hot_encoder(batch_action, n_classes=len(Action))
        policy_loss = self.criterion1(policy_preds, policy_targets)

        # 頭がバグったので policy loss 以外は一旦無視
        # value_loss = self.criterion2(outputs["value"].flatten(), batch["win"])
        # state_loss = self.criterion3(outputs["state"].flatten(), batch["hidden_state"].flatten())
        # global_state_loss = self.criterion3(outputs["global_state"].flatten(), batch["hidden_global_state"].flatten())

        # sap_available_mask = (batch["state"][:, -1, State.SAP_AVAILABLE_AREA] > 0)  # (batch_size, w, h)
        # sap_loss = self.criterion4(outputs["sap"].squeeze(1), batch["sap"], sap_available_mask)

        loss = (
            policy_loss * self.cfg.loss_weight_policy
            # + state_loss * self.cfg.loss_weight_state
            # + value_loss * self.cfg.loss_weight_value
            # + global_state_loss * self.cfg.loss_weight_global_state
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

        preds = output_policy.argmax(dim=1).flatten()
        gts = batch_action.flatten()
        unit_masks = (batch["state"][:, :, State.OWN_UNIT_COUNT] > 0).flatten()  # unitが存在するところだけで計算する

        # print(f"preds.shape: {preds.shape}")
        # print(f"gts.shape: {gts.shape}")
        # print(f"unit_masks.shape: {unit_masks.shape}")

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
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=not batch_norm),
            nn.BatchNorm2d(out_channels) if batch_norm is not None else nn.Identity(),
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


def _to_tuple(x):
    if isinstance(x, int):
        return (x, x)
    return x


# Written by ChatGPT
class CustomConvLSTMCell(nn.Module):
    def __init__(
        self, input_channels, hidden_channels, kernel_size, proj_channels=0, bias=True, device=None, dtype=None
    ):
        """
        input_channels : 入力テンソルのチャネル数
        hidden_channels: セル内部の隠れ状態・セル状態のチャネル数
        kernel_size    : 畳み込みカーネルサイズ（int または tuple）
        proj_channels  : 0 の場合は通常の ConvLSTM、0 より大きい場合は出力に 1x1 畳み込みによる射影を適用
        bias           : バイアスの有無
        device, dtype  : パラメータ作成時のデバイス、型
        """
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.proj_channels = proj_channels
        self.use_proj = proj_channels > 0

        kernel_size = _to_tuple(kernel_size)
        self.kernel_size = kernel_size
        # 空間サイズを変えずに出力するためのパディング（各辺半径）
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        # 畳み込みによるゲートの重み
        # 入力からゲートへの変換: 出力チャネルは 4 * hidden_channels
        self.weight_x = nn.Parameter(
            torch.empty(4 * hidden_channels, input_channels, kernel_size[0], kernel_size[1], device=device, dtype=dtype)
        )
        # 隠れ状態からゲートへの変換
        # ※projection を使う場合、前時刻の h のチャネル数は proj_channels となる
        hidden_dim = proj_channels if self.use_proj else hidden_channels
        self.weight_h = nn.Parameter(
            torch.empty(4 * hidden_channels, hidden_dim, kernel_size[0], kernel_size[1], device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(4 * hidden_channels, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        # projection 用のパラメータ（1x1 畳み込み）： hidden_channels → proj_channels
        if self.use_proj:
            self.weight_proj = nn.Parameter(
                torch.empty(proj_channels, hidden_channels, 1, 1, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("weight_proj", None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / (self.hidden_channels**0.5)
        for param in self.parameters():
            if param is not None:
                nn.init.uniform_(param, -stdv, stdv)

    def forward(self, x, hx):
        """
        x  : 入力テンソル (batch, input_channels, H, W)
        hx : タプル (h_prev, c_prev)
             h_prev: (batch, proj_channels if use_proj else hidden_channels, H, W)
             c_prev: (batch, hidden_channels, H, W)
        """
        h_prev, c_prev = hx

        # 入力と隠れ状態からの畳み込み
        conv_x = F.conv2d(x, self.weight_x, bias=None, padding=self.padding)
        conv_h = F.conv2d(h_prev, self.weight_h, bias=None, padding=self.padding)
        gates = conv_x + conv_h
        if self.bias is not None:
            # バイアスは (1, 4*hidden_channels, 1, 1) の形状にして加算
            gates = gates + self.bias.view(1, -1, 1, 1)

        # ゲートを 4 つに分割
        i_gate, f_gate, g_gate, o_gate = torch.chunk(gates, 4, dim=1)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)

        c_new = f_gate * c_prev + i_gate * g_gate
        h_new = o_gate * torch.tanh(c_new)

        if self.use_proj:
            # 1x1 畳み込みによる projection
            h_new = F.conv2d(h_new, self.weight_proj, bias=None, padding=0)

        return h_new, c_new


# Written by ChatGPT
class SimpleConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        """
        input_channels  : 入力のチャネル数
        hidden_channels : 隠れ状態のチャネル数
        kernel_size     : 畳み込みカーネルサイズ（int または tuple）
        bias            : バイアスの有無
        """
        super().__init__()
        # 畳み込みによるパディングは、カーネルサイズの半径
        padding = kernel_size // 2 if isinstance(kernel_size, int) else (kernel_size[0] // 2, kernel_size[1] // 2)
        self.hidden_channels = hidden_channels
        # 入力 x と隠れ状態 h をチャネル方向に連結して 4 倍のチャネル数で一括畳み込み
        self.conv = nn.Conv2d(
            input_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=padding, bias=bias
        )
        # self.conv = DoubleConv(
        #     input_channels + hidden_channels,
        #     4 * hidden_channels,
        #     mid_channels=hidden_channels * 2,
        #     res=True,
        #     kernel_size=kernel_size,
        #     batch_norm=False,
        # )
        self.h_init = nn.Parameter(torch.randn(1, hidden_channels, 24, 24))
        self.c_init = nn.Parameter(torch.randn(1, hidden_channels, 24, 24))

    def forward(self, x, hidden):
        """
        x     : 入力テンソル (batch, input_channels, H, W)
        hidden: タプル (h, c) 各テンソルの shape は (batch, hidden_channels, H, W)
        """
        h, c = hidden
        if h is None:
            h, c = self.h_init.repeat(x.size(0), 1, 1, 1), self.c_init.repeat(x.size(0), 1, 1, 1)
        combined = torch.cat([x, h], dim=1)
        conv_out = self.conv(combined)
        # 4 つのゲートに分割
        i, f, g, o = torch.chunk(conv_out, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class SimpleConvLSTMCellVer2(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        """
        input_channels  : 入力のチャネル数
        hidden_channels : 隠れ状態のチャネル数
        kernel_size     : 畳み込みカーネルサイズ（int または tuple）
        bias            : バイアスの有無
        """
        super().__init__()
        # 畳み込みによるパディングは、カーネルサイズの半径
        padding = kernel_size // 2 if isinstance(kernel_size, int) else (kernel_size[0] // 2, kernel_size[1] // 2)
        self.hidden_channels = hidden_channels
        # self.conv = nn.Conv2d(
        #     input_channels + hidden_channels, hidden_channels * 2, kernel_size, padding=padding, bias=bias
        # )
        self.conv = DoubleConv(
            input_channels + hidden_channels, hidden_channels * 2, res=True, kernel_size=kernel_size, batch_norm=False
        )

    def forward(self, x, hidden):
        """
        x     : 入力テンソル (batch, input_channels, H, W)
        hidden: タプル (h, c) 各テンソルの shape は (batch, hidden_channels, H, W)
        """
        h, c = hidden
        combined = torch.cat([x, h], dim=1)
        conv_out = self.conv(combined)
        i, f = torch.chunk(conv_out, 2, dim=1)
        return i, f


class ConvLSTM(nn.Module):
    def __init__(
        self, input_channels, hidden_channels, kernel_size, num_layers=1, bias=True, batch_first=False, res=False
    ):
        """
        input_channels  : 入力のチャネル数
        hidden_channels : 各層の隠れ状態のチャネル数
        kernel_size     : 畳み込みカーネルサイズ
        num_layers      : LSTM の層数
        bias            : バイアスの有無
        batch_first     : 入出力テンソルが (batch, seq, C, H, W) なら True
        """
        super().__init__()
        self.num_layers = num_layers
        self.batch_first = batch_first

        cell_list = []
        for i in range(num_layers):
            cur_in_channels = input_channels if i == 0 else hidden_channels
            # cell_list.append(SimpleConvLSTMCellVer2(cur_in_channels, hidden_channels, kernel_size, bias))
            cell_list.append(SimpleConvLSTMCell(cur_in_channels, hidden_channels, kernel_size, bias))
        self.cell_list = nn.ModuleList(cell_list)
        self.res = res

    def forward(self, x, hidden=None):
        """
        x : 入力テンソル
            batch_first=False の場合 (seq_len, batch, C, H, W)
            batch_first=True の場合 (batch, seq_len, C, H, W)
        hidden: オプションで各層の初期状態 [(h0, c0), ...]
                各状態の shape は (batch, hidden_channels, H, W)
                省略時はゼロで初期化
        """
        # if self.batch_first:
        #     # (batch, seq, C, H, W) -> (seq, batch, C, H, W)
        #     x = x.transpose(0, 1)
        # seq_len, batch_size, _, H, W = x.size()

        # # 各層の初期状態を用意
        # if hidden is None:
        #     hidden = []
        #     for i in range(self.num_layers):
        #         h = torch.zeros(batch_size, self.cell_list[i].hidden_channels, H, W, device=x.device)
        #         c = torch.zeros(batch_size, self.cell_list[i].hidden_channels, H, W, device=x.device)
        #         hidden.append((h, c))

        # layer_input = x
        # last_state_list = []
        # # 各層ごとに時系列を処理
        # for i in range(self.num_layers):
        #     h, c = hidden[i]
        #     outputs = []
        #     for t in range(seq_len):
        #         h, c = self.cell_list[i](layer_input[t], (h, c))
        #         outputs.append(h)
        #     layer_output = torch.stack(outputs, dim=0)
        #     layer_input = layer_output  # 次層への入力
        #     last_state_list.append((h, c))

        # if self.batch_first:
        #     layer_output = layer_output.transpose(0, 1)
        # return layer_output, last_state_list

        if self.batch_first:
            x = x.transpose(0, 1)  # (seq_len, batch, C, H, W)
        seq_len, batch_size, _, H, W = x.size()
        num_layers = self.num_layers

        # 各層の初期状態を用意（セルごとの初期値）
        hidden_states = []
        for i in range(num_layers):
            if hidden is None:
                # h0 = torch.zeros(batch_size, self.cell_list[i].hidden_channels, H, W, device=x.device)
                # c0 = torch.zeros(batch_size, self.cell_list[i].hidden_channels, H, W, device=x.device)
                h0, c0 = None, None
            else:
                h0, c0 = hidden[i]
            hidden_states.append((h0, c0))

        # 各セル (layer, t) の出力を格納するグリッド（2次元リスト）
        h_grid = [[None for _ in range(seq_len)] for _ in range(num_layers)]
        c_grid = [[None for _ in range(seq_len)] for _ in range(num_layers)]

        # wavefront parallelism:
        # 対角線 d = layer + time について、同じ d のセルは互いに依存しないため並列計算可能
        for d in range(num_layers + seq_len - 1):
            for layer in range(num_layers):
                t = d - layer
                if t < 0 or t >= seq_len:
                    continue
                # 入力は、layer == 0 の場合は x[t]、それ以外は下層の同時刻の出力
                cell_input = x[t] if layer == 0 else h_grid[layer - 1][t]
                # 同一層の前時刻の出力がなければ初期状態を用いる
                if t == 0:
                    h_prev, c_prev = hidden_states[layer]
                else:
                    h_prev, c_prev = h_grid[layer][t - 1], c_grid[layer][t - 1]
                # h_prev, c_prev = h0, c0  # DEBUG
                cell = self.cell_list[layer]
                h_new, c_new = cell(cell_input, (h_prev, c_prev))
                # if self.res and layer % 2 == 0 and layer > 0:
                #     h_new = h_new + h_grid[layer - 2][t]
                h_grid[layer][t] = h_new
                c_grid[layer][t] = c_new

        # 最終層の出力を結果としてまとめる
        outputs = torch.stack(h_grid[-1], dim=0)  # (seq_len, batch, hidden_channels, H, W)
        if self.batch_first:
            outputs = outputs.transpose(0, 1)  # (batch, seq_len, hidden_channels, H, W)
        # 各層の最終状態も返す
        final_states = []
        for layer in range(num_layers):
            final_states.append((h_grid[layer][-1], c_grid[layer][-1]))
            # final_states.append((h0, c0))  # DEBUG
        return outputs, final_states


# 使用例
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    input_channels = 3
    hidden_channels = 8
    kernel_size = 3
    # batch_first=True の場合の入力形状: (batch, seq, C, H, W)
    x = torch.randn(batch_size, seq_len, input_channels, 16, 16)

    convlstm = ConvLSTM(input_channels, hidden_channels, kernel_size, num_layers=2, batch_first=True)
    output, states = convlstm(x)
    print("出力 shape:", output.shape)  # (batch, seq, hidden_channels, H, W)


class LuxLSTMModel(nn.Module):
    def __init__(
        self,
        state_space_size: int,
        global_state_space_size: int,
        action_space_size: int,
        hidden_state_space_size: int,
        n_stack: int,
        num_layers: int = 14,
        hidden_channels: int = 128,
        kernel_size: int = 3,
        return_hidden: bool = False,
        # bilinear: bool = True,
        # res: bool = False,
    ) -> None:
        super().__init__()

        self.debug = False

        # DEBUG
        if self.debug:
            self.conv = nn.Sequential(
                DoubleConv(state_space_size + global_state_space_size, hidden_channels),
                *[DoubleConv(in_channels=hidden_channels, out_channels=hidden_channels, res=True) for _ in range(8)],
                nn.Conv2d(hidden_channels, action_space_size, kernel_size=1),
            )
            self.return_hidden = return_hidden
            return

        self.return_hidden = return_hidden
        res = True
        bilinear = True
        self.inc = DoubleConv(state_space_size + global_state_space_size, 64, res=res)
        self.down1 = Down(64, 128, res=res)
        self.down2 = Down(128, 256, res=res)
        self.down3 = Down(256, 256, res=res)
        factor = 2 if bilinear else 1
        self.up1 = Up(256 * 2 + global_state_space_size, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 128, bilinear)
        # self.policy_net = OutConv(64 * n_stack, action_space_size)

        # self.hidden_channels = hidden_channels
        # self.conv1 = nn.Sequential(
        #     DoubleConv(state_space_size + global_state_space_size, hidden_channels),
        #     *[
        #         DoubleConv(
        #             in_channels=hidden_channels, out_channels=hidden_channels, res=True, kernel_size=5, batch_norm=False
        #         )
        #         for _ in range(8)
        #     ],
        # )

        self.convlstm = ConvLSTM(
            input_channels=128,
            hidden_channels=256,
            kernel_size=kernel_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            res=True,
        )
        self.policy_net = OutConv(128 * 3, action_space_size)

        # self.policy_net = OutConv(128, action_space_size)

        # self.policy_net = nn.Sequential(
        #     DoubleConv(128 * 3, hidden_channels, res=res),
        #     OutConv(hidden_channels, action_space_size),
        # )

        # self.net_policy = nn.Sequential(nn.Conv2d(hidden_channels, action_space_size, kernel_size=1))
        # self.return_hidden = return_hidden

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
        # print(f"x: {x.shape}")

        # DEBUG
        if self.debug:
            x = x.view(-1, _c + _cg, _x, _y)
            x = self.conv(x)
            x = x.view(_n, _t, -1, _x, _y)
            # print(f"x: {x.shape}")
            output = {
                "policy": x,
            }
            if self.return_hidden:
                return output, hidden
            else:
                return output

        x = x.view(-1, _c + _cg, _x, _y)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        sx, sy = x4.shape[2:]
        _n, _t, _c = global_state.shape
        gx = global_state.view(-1, _c, 1, 1)
        gx = gx.repeat(1, 1, sx, sy)

        x4 = torch.cat([x4, gx], dim=1)
        # x = self.global_avg_pool(x4).view(_n, -1)
        # value_logits = self.value_net(x)
        # global_state_logits = self.global_state_net(x)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        x = x.view(_n, _t, -1, _x, _y)
        x_lstm, hidden = self.convlstm(x, hidden)
        # x = x_lstm
        # x = x + x_lstm
        x = torch.cat([x, x_lstm], dim=2)
        x = x.flatten(0, 1)

        # print(f"x: {x.shape}")
        policy_logits = self.policy_net(x)
        policy_logits = policy_logits.view(_n, _t, -1, _x, _y)
        # print(f"policy_logits: {policy_logits.shape}")

        # x = x.view(-1, _c + _cg, _x, _y)
        # x = self.conv1(x)
        # x = x.view(_n, _t, -1, _x, _y)
        # # x, hidden = self.convlstm(x, hidden)
        # # print(f"convlstm after reshape: {x.shape}")
        # # x = x.reshape(_n * _t, -1, _x, _y)
        # # print(f"convlstm before reshape: {x.shape}")
        # # x = self.conv2(x)
        # # x = x.reshape(_n, _t, -1, _x, _y)
        # # print(f"convlstm: {x.shape}")
        # # flatten
        # x = x.flatten(0, 1)
        # policy_logits = self.net_policy(x)
        # # reshape
        # policy_logits = policy_logits.view(_n, _t, len(Action), _x, _y)

        # print(f"policy_logits: {policy_logits.shape}")
        assert policy_logits.shape == (state.shape[0], state.shape[1], len(Action), state.shape[3], state.shape[4])
        output = {
            "policy": policy_logits,
            # "sap": sap_logits,
            # "state": state_logits,
            # "global_state": global_state_logits,
            # "value": value_logits,
        }
        if self.return_hidden:
            return output, hidden
        else:
            return output


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
