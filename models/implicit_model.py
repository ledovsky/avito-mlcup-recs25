from typing import Self

import wandb
import wandb.wandb_run
import implicit
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
import joblib

import implicit.cpu.als
import implicit.gpu.als

from .base import BaseModel, FaissPredict


class ALSRecommender(BaseModel):
    """
    ALS-based recommender using implicit library.
    """

    def __init__(
        self,
        run: wandb.wandb_run.Run = wandb.init(mode="disabled"),
        do_dedupe: bool = False,
        use_week_discount: bool = False,
        filter_rare_events: bool = False,
        contact_weight: int = 10,
        als_factors: int = 60,
        iterations: int = 10,
        top_k_items: int | None = None,
    ):
        super().__init__()
        # coniguration
        self.als_kwargs = {
            "factors": als_factors,
            "iterations": iterations,
            "calculate_training_loss": True,
        }
        self.do_dedupe = do_dedupe
        self.use_week_discount = use_week_discount
        self.flag_filter_rare_events = filter_rare_events
        self.run = run
        # log device type to wandb
        device = "gpu" if "gpu" in implicit.als.AlternatingLeastSquares.__module__ else "cpu"
        self.run.config.update({"device": device})

        self.contact_weight = contact_weight
        self.model: (
            implicit.cpu.als.AlternatingLeastSquares
            | implicit.gpu.als.AlternatingLeastSquares
            | None
        ) = None
        self.top_k_items = top_k_items
        self.loss_ar: list[float] = []

    def fit(self, df_train: pl.DataFrame, df_events: pl.DataFrame) -> None:
        event_weights = {
            row["event"]: self.contact_weight if row["is_contact"] else 1
            for row in df_events.select(["event", "is_contact"]).to_dicts()
        }

        if self.flag_filter_rare_events:
            df_train = self.filter_rare_events(df_train)

        if self.do_dedupe:
            df_train = self.dedupe(df_train)

        if self.top_k_items:
            df_train = self.filter_top_k_items(df_train, self.top_k_items)

        # Fill
        # self.user_id_to_index,
        # self.item_id_to_index,
        # self.index_to_item_id,
        # self.sparse_matrix,
        self.build_interaction_matrix(
            df_train,
            event_weights=event_weights,
            use_week_discount=self.use_week_discount,
        )

        def callback(iteration: int, elapsed: float, loss: float):
            self.loss_ar.append(loss)
            self.run.log({"loss": loss}, step=iteration)

        self.model = implicit.als.AlternatingLeastSquares(**self.als_kwargs)
        self.model.fit(self.sparse_matrix, callback=callback)

    def predict(self, user_to_pred: list[int | str], N: int = 40) -> pl.DataFrame:
        if self.model is None or self.sparse_matrix is None:
            raise ValueError("Model is not fitted. Call fit() before predict().")

        valid_users = [u for u in user_to_pred if u in self.user_id_to_index]
        if not valid_users:
            return pl.DataFrame(columns=["cookie", "node", "scores"])

        user_indices = np.array([self.user_id_to_index[u] for u in valid_users])

        recommendations, scores = self.model.recommend(
            user_indices,
            self.sparse_matrix[user_indices],
            N=N,
            filter_already_liked_items=True,
        )

        df_pred = pl.DataFrame(
            {
                "cookie": valid_users,
                "node": [
                    [self.index_to_item_id[i] for i in recs]
                    for recs in recommendations.tolist()
                ],
                "scores": scores.tolist(),
            }
        )
        return df_pred.explode(["node", "scores"])

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> Self:
        return joblib.load(path)


class ALS2(FaissPredict, ALSRecommender):
    def fit(self, df_train: pl.DataFrame, df_events: pl.DataFrame) -> None:
        super().fit(df_train, df_events)
        self.set_seen_items(self.get_seen_nodes(df_train))
        self.set_embeddings(self.model.user_factors, self.model.item_factors)
        self.set_is_cos_dist(False)
        self.set_populars(self.get_populars(df_train))
