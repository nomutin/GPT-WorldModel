"""
Execute lightning cli.

References
----------
- https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html

"""

import torch
from lightning.pytorch.cli import LightningCLI


def main() -> None:
    """Execute lightning cli."""
    torch.set_float32_matmul_precision("medium")
    LightningCLI(save_config_kwargs={"overwrite": True, "config_filename": "config_override.yaml"})


if __name__ == "__main__":
    main()
