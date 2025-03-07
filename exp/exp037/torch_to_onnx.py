from pathlib import Path

import torch
from torch import nn
from lux.utils import State, Action, GlobalState
from lux.models import LuxUNetModel
from lux.params import EnvParams


def torch_to_onnx(model, onnx_path: Path, n_stack: int):
    model.eval()
    state = torch.randn(1, n_stack, len(State), EnvParams.map_width, EnvParams.map_height)
    global_state = torch.randn(1, n_stack, len(GlobalState))
    input_names = ["batch"]
    output_names = ["outputs"]
    torch.onnx.export(
        model=model,
        args=(state, global_state),
        f=onnx_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=11,
    )


def load_model(model: nn.Module, model_path: Path):
    ckpt = torch.load(model_path, weights_only=False, map_location="cpu")
    state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)


def main():
    model_dir = Path("/home/kyohei.uto/kaggle-luxai-s3/exp/rl_best/output/")
    model_path = model_dir / "best_model.ckpt"
    onnx_path = model_dir / "best_model.onnx"
    n_stack = 4

    model = LuxUNetModel(
        state_space_size=len(State),
        global_state_space_size=len(GlobalState),
        action_space_size=len(Action),
        n_stack=n_stack,
        res=True,
    )
    load_model(model, model_path)
    torch_to_onnx(model, onnx_path, n_stack)


if __name__ == "__main__":
    main()
