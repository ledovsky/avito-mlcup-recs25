from typing import Self

import joblib
import polars as pl
from scipy.sparse import csr_matrix
import torch


class BaseModel:
    def __init__(self) -> None:
        self.user_id_to_index: dict[str, int] = {}
        self.item_id_to_index: dict[int, int] = {}
        self.index_to_item_id: dict[int, int] = {}
        self.sparse_matrix: csr_matrix = None

    def predict(self, user_to_pred: list[int | str], N: int = 40) -> pl.DataFrame:
        raise NotImplementedError()

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> Self:
        return joblib.load(path)

    def filter_rare_events(self, df_train: pl.DataFrame, thr=3) -> pl.DataFrame:
        return (
            df_train.group_by(["cookie", "node"])
            .agg(pl.len().alias("event_count"))
            .filter(pl.col("event_count") >= thr)
            .select("cookie", "node")
            .join(df_train, on=["cookie", "node"], how="inner")
        )

    def dedupe(self, df_train: pl.DataFrame) -> pl.DataFrame:
        return df_train.sort(
            ["cookie", "node", "is_contact", "event_date"],
            descending=[False, False, False, True],
        ).unique(subset=["cookie", "node"])

    def filter_top_k_items(
        self, df: pl.DataFrame, top_k_items: int
    ) -> pl.DataFrame:
        top_items = (
            df["node"]
            .value_counts()
            .sort("count", descending=True)
            .head(top_k_items)["node"]
            .to_list()
        )
        return df.filter(pl.col("node").is_in(top_items))

    def build_interaction_matrix(
        self,
        df: pl.DataFrame,
        event_col: str = "event",
        event_weights: dict | None = None,
        use_week_discount: bool = False,
    ) -> None:
        """Fills:
        - self.user_id_to_index,
        - self.item_id_to_index,
        - self.index_to_item_id,
        - self.sparse_matrix,
        """
        users = df["cookie"].unique().to_list()
        items = df["node"].unique().to_list()
        items = df["node"].unique().to_list()
        self.user_id_to_index = {u: i for i, u in enumerate(users)}
        self.item_id_to_index = {j: i for i, j in enumerate(items)}
        self.index_to_item_id = {i: j for j, i in self.item_id_to_index.items()}

        rows = df["cookie"].replace_strict(self.user_id_to_index).to_list()
        cols = df["node"].replace_strict(self.item_id_to_index).to_list()
        base_vals = (
            [event_weights.get(ev, 1) for ev in df[event_col].to_list()]
            if event_weights
            else [1] * len(rows)
        )
        if use_week_discount and "week" in df.columns:
            weeks = df["week"].to_list()
            values = [
                base_vals[i] * (2 if weeks[i] >= 4 else 1) for i in range(len(rows))
            ]
        else:
            values = base_vals

        self.sparse_matrix = csr_matrix((values, (rows, cols)), shape=(len(users), len(items)))


class BaseTorchModel(BaseModel):
    def save(self, path: str) -> None:
        torch.save(self, path)

    @classmethod
    def load(cls, path: str, weights_only: bool = True) -> Self:
        return torch.load(path, weights_only=weights_only)