import wandb
import wandb.wandb_run

import polars as pl

from .base import BaseTorchModel


class TorchEmbModel(BaseTorchModel):
    def __init__(
        self,
        run: wandb.wandb_run.Run,
    ):
        pass

    def fit(self, df_train: pl.DataFrame, df_events: pl.DataFrame) -> None:
        pass

    def predict(self, user_to_pred: list[int | str], N: int = 40) -> pl.DataFrame:
        pass