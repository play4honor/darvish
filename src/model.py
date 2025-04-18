"""Primarily for the LightningModule that contains the model.

Other network modules, etc. can go here or not, depending on how confusing it is.
"""

from typing import Any

import torch
import lightning.pytorch as pl


class DarvishLightning(pl.LightningModule):
    """Model for whatever we're doing."""

    def __init__(
        self,
        optimizer_params: dict[str, Any],
        *args,
        **kwargs,
    ):
        """Initialize the model.

        - optimizer_params: dictionary of parameters to initialize optimizer with.
        """

        super().__init__()
        self.save_hyperparameters(logger=False)
        self.optimizer_params = optimizer_params

        raise NotImplementedError

    def configure_optimizers(self):
        """Lightning hook for optimizer setup.

        Body here is just an example, although it probably works okay for a lot of things.
        We can't pass arguments to this directly, so they need to go to the init.
        """

        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_params)
        return optimizer

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for the model.

        - x is compatible with the output of data.TrainingDataset.__getitem__, literally the output
          from a dataloader.

        Returns whatever the output of the model is.
        """

        raise NotImplementedError

    def step(self, stage: str, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Generic step for training or validation, assuming they're similar.

        - stage: one of "train" or "valid"
        - x: dictionary of torch tensors, input to model, targets, etc.

        This MUST log "valid_loss" during the validation step in order to have the model checkpointing work as written.

        Returns loss as one-element tensor.
        """

        raise NotImplementedError

    def training_step(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Lightning hook for training."""
        return self.step("train", x)

    def validation_step(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Lightning hook for validation."""
        return self.step("valid", x)
