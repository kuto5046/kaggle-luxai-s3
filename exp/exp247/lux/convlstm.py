# written original code with pytorch
# https://github.com/AlignmentResearch/train-learned-planner/blob/main/cleanba/convlstm.py

import torch
import torch.nn.functional as F
from torch import nn


class ConvLSTMCell(nn.Module):
    """
    ConvLSTMCell
      - x: 現在の入力 (B, input_channels, H, W)
      - state: (c, h) のタプル。各テンソルは (B, hidden_channels, H, W)
      - prev_layer_hidden: 前段セルの出力（または別の情報） (B, hidden_channels, H, W)
    """

    def __init__(
        self,
        input_channels,
        hidden_channels,
        kernel_size,
        pool_and_inject="horizontal",
        pool_projection="per-channel",
        output_activation="sigmoid",
        forget_bias=0.0,
        fence_pad="same",
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.pool_and_inject = pool_and_inject
        self.pool_projection = pool_projection
        self.output_activation = output_activation
        self.forget_bias = forget_bias
        self.fence_pad = fence_pad

        # 入力と prev_layer_hidden の結合後のチャンネル数
        if pool_and_inject == "no":
            conv_in_channels = input_channels + hidden_channels
        else:
            conv_in_channels = input_channels + 2 * hidden_channels

        padding = kernel_size // 2  # "same" padding

        self.conv_i = nn.Conv2d(conv_in_channels, 4 * hidden_channels, kernel_size, padding=padding, bias=True)
        self.conv_h = nn.Conv2d(hidden_channels, 4 * hidden_channels, kernel_size, padding=padding, bias=True)

        if fence_pad != "no":
            self.conv_fence = nn.Conv2d(1, 4 * hidden_channels, kernel_size, padding=padding, bias=True)
        else:
            self.conv_fence = None

        if self.pool_and_inject != "no":
            if self.pool_projection == "per-channel":
                self.project = nn.Parameter(torch.ones(2, hidden_channels))
            elif self.pool_projection == "overall":
                self.project = nn.Parameter(torch.ones(2, 1))
            elif self.pool_projection == "full":
                self.fc_full = nn.Linear(2 * hidden_channels, hidden_channels)

    def pool_and_project_fn(self, tensor):
        """
        tensor: (B, hidden_channels, H, W)
        チャンネルごとの最大値と平均値を計算し、pool_projection の設定に応じて出力し、
        空間次元に展開する。
        """
        B, C, H, W = tensor.size()
        h_max = tensor.view(B, C, -1).max(dim=2)[0]  # (B, C)
        h_mean = tensor.view(B, C, -1).mean(dim=2)  # (B, C)
        if self.pool_projection == "max":
            pooled = h_max
        elif self.pool_projection == "mean":
            pooled = h_mean
        elif self.pool_projection == "full":
            concat = torch.cat([h_max, h_mean], dim=1)  # (B, 2C)
            pooled = self.fc_full(concat)
        elif self.pool_projection == "per-channel":
            pooled = self.project[0] * h_max + self.project[1] * h_mean
        elif self.pool_projection == "overall":
            pooled = self.project[0] * h_max + self.project[1] * h_mean
        else:
            raise ValueError("Unknown pool_projection")
        pooled = pooled.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)
        return pooled

    def forward(self, x, state, prev_layer_hidden):
        """
        Args:
          x: 現在の入力 (B, input_channels, H, W)
          state: (c, h) タプル、各 (B, hidden_channels, H, W)
          prev_layer_hidden: 前段セルの出力 (B, hidden_channels, H, W)
        Returns:
          new_state: (new_c, new_h)
          new_h: 出力（隠れ状態）
        """
        c, h = state

        # fence の処理
        if self.fence_pad == "same" and self.conv_fence is not None:
            fence = torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            if x.size(2) > 2 and x.size(3) > 2:
                fence[:, :, 1:-1, 1:-1] = 0
            processed_fence = self.conv_fence(fence)
        elif self.fence_pad == "valid" and self.conv_fence is not None:
            kernel_size = self.conv_fence.kernel_size[0]
            valid_H = x.size(2) + (kernel_size - 1)
            valid_W = x.size(3) + (kernel_size - 1)
            fence = torch.ones(x.size(0), 1, valid_H, valid_W, device=x.device, dtype=x.dtype)
            if valid_H > 2 and valid_W > 2:
                fence[:, :, 1:-1, 1:-1] = 0
            processed_fence = self.conv_fence(fence)
        else:
            processed_fence = 0

        if self.pool_and_inject == "no":
            cell_inputs = torch.cat([x, prev_layer_hidden], dim=1)
        else:
            if self.pool_and_inject == "horizontal":
                to_pool = h
            elif self.pool_and_inject == "vertical":
                to_pool = prev_layer_hidden
            else:
                raise ValueError("Unknown pool_and_inject mode")
            pooled = self.pool_and_project_fn(to_pool)
            cell_inputs = torch.cat([x, prev_layer_hidden, pooled], dim=1)

        gates = self.conv_i(cell_inputs) + self.conv_h(h)
        if self.conv_fence is not None:
            gates = gates + processed_fence

        i_gate, j_gate, f_gate, o_gate = torch.split(gates, self.hidden_channels, dim=1)
        i_gate = torch.tanh(i_gate)
        j_gate = torch.sigmoid(j_gate)
        f_gate = torch.sigmoid(f_gate + self.forget_bias)
        if self.output_activation == "sigmoid":
            o_gate = torch.sigmoid(o_gate)
        elif self.output_activation == "tanh":
            o_gate = torch.tanh(o_gate)
        else:
            raise ValueError("Unknown output_activation")
        new_c = c * f_gate + i_gate * j_gate
        new_h = torch.tanh(new_c) * o_gate
        return (new_c, new_h), new_h


class ConvLSTM(nn.Module):
    """
    ConvLSTM モジュール
      - embed_layers: 複数の畳み込み層で入力画像を埋め込み
      - recurrent_cells: 複数の ConvLSTMCell をスタックして時系列処理
      - repeats_per_step: 各時刻で内部セル更新を何回繰り返すか
    """

    def __init__(self, embed_configs, recurrent_config):
        """
        Args:
          embed_configs: 畳み込み層の設定リスト（各要素は dict）
          recurrent_config: dict。以下のキーを含む:
              "input_channels": embed後のチャネル数,
              "hidden_channels": セルの隠れ状態チャネル数,
              "kernel_size": 畳み込みカーネルサイズ,
              "n_recurrent": セルの層数,
              "repeats_per_step": 各時刻における再帰回数,
              その他のオプション（pool_and_inject など）
        """
        super().__init__()
        # embed 部分の構築
        layers = []
        for i, cfg in enumerate(embed_configs):
            in_channels = cfg.get("in_channels", 3) if i == 0 else embed_configs[i - 1]["out_channels"]
            padding = cfg.get("padding", cfg["kernel_size"] // 2)
            layers.append(
                nn.Conv2d(
                    in_channels,
                    cfg["out_channels"],
                    kernel_size=cfg["kernel_size"],
                    stride=cfg.get("stride", 1),
                    padding=padding,
                )
            )
            if cfg.get("use_relu", True) and i < len(embed_configs) - 1:
                layers.append(nn.ReLU())
        self.embed_layers = nn.Sequential(*layers)

        # recurrent_cells の構築
        self.n_recurrent = recurrent_config["n_recurrent"]
        self.recurrent_cells = nn.ModuleList(
            [
                ConvLSTMCell(
                    input_channels=recurrent_config["input_channels"],
                    hidden_channels=recurrent_config["hidden_channels"],
                    kernel_size=recurrent_config["kernel_size"],
                    pool_and_inject=recurrent_config.get("pool_and_inject", "horizontal"),
                    pool_projection=recurrent_config.get("pool_projection", "per-channel"),
                    output_activation=recurrent_config.get("output_activation", "sigmoid"),
                    forget_bias=recurrent_config.get("forget_bias", 0.0),
                    fence_pad=recurrent_config.get("fence_pad", "same"),
                )
                for _ in range(self.n_recurrent)
            ]
        )
        self.residual = recurrent_config.get("residual", False)
        self.skip_final = recurrent_config.get("skip_final", True)
        self.repeats_per_step = recurrent_config.get("repeats_per_step", 1)

    def forward(self, x, carry=None, episode_starts=None):
        """
        Args:
          x: 入力シーケンス (T, B, C, H, W) または (B, C, H, W)（単一時刻）
          carry: 各セルの状態 [(c, h), ...] のリスト。指定がなければゼロ状態で初期化。
          episode_starts: 未使用（オプション）
        Returns:
          outputs: 各時刻の出力 (T, B, hidden_channels, H_new, W_new)
          carry: 最終状態のリスト
        """
        if x.dim() == 4:
            x = x.unsqueeze(0)  # (1, B, C, H, W)
        T, B, C, H, W = x.size()

        # embed 部分: 時刻とバッチ軸をまとめて処理
        x_reshaped = x.view(T * B, C, H, W)
        x_embedded = self.embed_layers(x_reshaped)
        _, C_new, H_new, W_new = x_embedded.size()
        x_embedded = x_embedded.view(T, B, C_new, H_new, W_new)

        # carry がない場合はゼロ状態で初期化
        if carry is None:
            carry = []
            hidden_channels = self.recurrent_cells[0].hidden_channels
            for _ in range(self.n_recurrent):
                c0 = torch.zeros(B, hidden_channels, H_new, W_new, device=x.device, dtype=x.dtype)
                h0 = torch.zeros(B, hidden_channels, H_new, W_new, device=x.device, dtype=x.dtype)
                carry.append((c0, h0))

        outputs = []
        for t in range(T):
            x_t = x_embedded[t]  # (B, C_new, H_new, W_new)
            # repeats_per_step 回の再帰更新を実施
            for r in range(self.repeats_per_step):
                new_carry = []
                # 初回は前段として、最後のセルの出力を利用
                prev = carry[-1][1]
                for i, cell in enumerate(self.recurrent_cells):
                    if i == 0:
                        prev_layer_hidden = carry[-1][1]
                    else:
                        prev_layer_hidden = prev
                    state, out = cell(x_t, carry[i], prev_layer_hidden)
                    if self.residual:
                        out = out + prev_layer_hidden
                    prev = out
                    new_carry.append(state)
                carry = new_carry
            # skip_final の場合、出力に元の x_t を加算（残差接続）
            if self.skip_final:
                output = carry[-1][1] + x_t
            else:
                output = carry[-1][1]
            outputs.append(output)
        outputs = torch.stack(outputs, dim=0)
        if outputs.size(0) == 1:
            outputs = outputs.squeeze(0)
        return outputs, carry


# ------------------------------------------------------------------------------
# 使用例（repeats_per_step を 3 として設定）
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    T, B = 5, 2
    x = torch.randn(T, B, 3, 64, 64)

    # embed 部分の設定
    embed_configs = [
        {"in_channels": 3, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1, "use_relu": True},
        {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1, "use_relu": False},
    ]
    # recurrent_config の設定に repeats_per_step を追加
    recurrent_config = {
        "input_channels": 32,  # embed 層出力のチャネル数
        "hidden_channels": 32,
        "kernel_size": 3,
        "n_recurrent": 2,
        "repeats_per_step": 3,  # 1時刻あたりの内部更新回数
        "pool_and_inject": "horizontal",
        "pool_projection": "per-channel",
        "output_activation": "sigmoid",
        "forget_bias": 0.0,
        "fence_pad": "same",
        "residual": False,
        "skip_final": True,
    }

    model = ConvLSTM(embed_configs, recurrent_config)
    outputs, final_state = model(x)
    print("出力の形状:", outputs.shape)
