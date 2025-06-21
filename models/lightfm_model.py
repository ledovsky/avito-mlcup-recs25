import random

import faiss
import numpy as np
import polars as pl
from lightfm import LightFM
from lightfm.evaluation import auc_score, recall_at_k
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score

import wandb
import wandb.wandb_run

from .base import BaseModel


class LightFMRecommender(BaseModel):
    """
    LightFM-based recommender using LightFM library.
    """

    def __init__(
        self,
        run: wandb.wandb_run.Run,
        top_k_items: int | None = None,
        no_components: int = 30,
        contact_weight: int = 10,
        loss: str = "bpr",
    ) -> None:

        super().__init__()
        self.epochs = 5

        self.top_k_items = top_k_items
        self.no_components = no_components
        self.contact_weight = contact_weight
        self.loss = loss
        self.model: LightFM = None

    def fit(self, df_train: pl.DataFrame, df_events: pl.DataFrame) -> None:
        event_weights = {
            row["event"]: self.contact_weight if row["is_contact"] else 1
            for row in df_events.select(["event", "is_contact"]).to_dicts()
        }
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
            use_week_discount=False,
        )

        user_ids = df_train["cookie"].unique().to_list()
        item_ids = df_train["node"].unique().to_list()
        val_users = random.sample(user_ids, min(100, len(user_ids)))

        # Initialize and train LightFM model with partial fit and loss logging
        # prepare validation dataset
        # map each val_user to positive and negative items
        user_groups = (
            df_train.filter(pl.col("cookie").is_in(val_users))
            .group_by("cookie")
            .agg(pl.col("node").unique().alias("pos_items"))
            .to_dicts()
        )
        self._val_data = {
            row["cookie"]: {
                "pos": row["pos_items"],
                "neg": random.sample(
                    list(set(item_ids) - set(row["pos_items"])), len(row["pos_items"])
                ),
            }
            for row in user_groups
        }

        self.model = LightFM(no_components=self.no_components, loss=self.loss)

        for epoch in range(1, self.epochs + 1):
            self.model.fit_partial(
                self.sparse_matrix,
                epochs=1,
            )
            # compute validation ROC AUC
            y_true, y_score = [], []
            for u in val_users:
                u_idx = self.user_id_to_index[u]
                # positives
                for item in self._val_data[u]["pos"]:
                    idx = self.item_id_to_index[item]
                    y_true.append(1)
                    y_score.append(
                        np.dot(
                            self.model.user_embeddings[u_idx],
                            self.model.item_embeddings[idx],
                        )
                    )
                # negatives
                for item in self._val_data[u]["neg"]:
                    idx = self.item_id_to_index[item]
                    y_true.append(0)
                    y_score.append(
                        np.dot(
                            self.model.user_embeddings[u_idx],
                            self.model.item_embeddings[idx],
                        )
                    )
            roc = roc_auc_score(y_true, y_score)
            print(f"Epoch {epoch}/{self.epochs} - val_roc_auc: {roc:.4f}")

    def predict(self, user_to_pred: list[int | str], N: int = 40) -> pl.DataFrame:
        if self.model is None or self.sparse_matrix is None:
            raise ValueError("Model is not fitted. Call fit() before predict().")

        valid_users = [u for u in user_to_pred if u in self.user_id_to_index]
        if not valid_users:
            raise ValueError("No valid users")

        # retrieve user embeddings and item embeddings
        user_indices = np.array([self.user_id_to_index[u] for u in valid_users])
        user_embs = self.model.user_embeddings[user_indices]
        item_embs = self.model.item_embeddings
        # use dot product (inner product) similarity
        dim = item_embs.shape[1]
        # build Faiss HNSW index with inner product metric
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.add(item_embs)
        # search nearest neighbors
        D, I = index.search(user_embs, N)
        predictions = []
        for i, u in enumerate(valid_users):
            for rank, item_idx in enumerate(I[i]):
                predictions.append(
                    {
                        "cookie": u,
                        "node": self.index_to_item_id[int(item_idx)],
                        "scores": float(D[i][rank]),
                    }
                )
        df_pred = pl.DataFrame(predictions)
        return df_pred
