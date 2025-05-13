import polars as pl
from scipy.sparse import csr_matrix
import numpy as np
import faiss
import random
from sklearn.metrics import roc_auc_score
from lightfm import LightFM
from lightfm.evaluation import auc_score, recall_at_k

class LightFMRecommender:
    """
    LightFM-based recommender using LightFM library.
    """
    def __init__(self, df_events: pl.DataFrame):
        self.lfm_init_kwargs = {
            "no_components": 60, 
            "loss": "bpr", 
        }
        self.lfm_fit_kwargs = {
            "num_threads": 8,
        }
        self.epochs = 10
        self.event_weights = {
            row["event"]: 4 if row["is_contact"] else 1
            for row in df_events.select(["event", "is_contact"]).to_dicts()
        }

        self.model = None
        self.user_id_to_index = {}
        self.item_id_to_index = {}
        self.index_to_item_id = {}
        self.sparse_matrix = None

    def fit(self, users: pl.Series, nodes: pl.Series, events: pl.Series, weeks=None) -> None:
        user_ids = users.unique().to_list()
        item_ids = nodes.unique().to_list()

        self.user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self.item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
        self.index_to_item_id = {v: k for k, v in self.item_id_to_index.items()}

        rows = users.replace_strict(self.user_id_to_index).to_list()
        cols = nodes.replace_strict(self.item_id_to_index).to_list()
        values = [self.event_weights.get(ev, 1) for ev in events.to_list()]

        self.sparse_matrix = csr_matrix(
            (values, (rows, cols)), shape=(len(user_ids), len(item_ids))
        )

        # Initialize and train LightFM model with partial fit and loss logging
        # prepare validation dataset
        val_users = random.sample(user_ids, min(100, len(user_ids)))
        # map each val_user to positive and negative items
        df_inter = pl.DataFrame({"user": users, "item": nodes})
        user_groups = df_inter.filter(pl.col("user").is_in(val_users))\
            .group_by("user")\
            .agg(pl.col("item").unique().alias("pos_items"))\
            .to_dicts()
        self._val_data = {
            row["user"]: {
                "pos": row["pos_items"],
                "neg": random.sample(
                    list(set(item_ids) - set(row["pos_items"])),
                    len(row["pos_items"])
                )
            }
            for row in user_groups
        }

        self.model = LightFM(**self.lfm_init_kwargs)
        for epoch in range(1, self.epochs + 1):
            self.model.fit_partial(
                self.sparse_matrix,
                epochs=1,
                **self.lfm_fit_kwargs,
            )
            # compute validation ROC AUC
            y_true, y_score = [], []
            for u in val_users:
                u_idx = self.user_id_to_index[u]
                # positives
                for item in self._val_data[u]["pos"]:
                    idx = self.item_id_to_index[item]
                    y_true.append(1)
                    y_score.append(np.dot(
                        self.model.user_embeddings[u_idx],
                        self.model.item_embeddings[idx]
                    ))
                # negatives
                for item in self._val_data[u]["neg"]:
                    idx = self.item_id_to_index[item]
                    y_true.append(0)
                    y_score.append(np.dot(
                        self.model.user_embeddings[u_idx],
                        self.model.item_embeddings[idx]
                    ))
            roc = roc_auc_score(y_true, y_score)
            print(f"Epoch {epoch}/{self.epochs} - val_roc_auc: {roc:.4f}")

    def predict(self, user_to_pred: list[int | str], N: int = 40) -> pl.DataFrame:
        if self.model is None or self.sparse_matrix is None:
            raise ValueError("Model is not fitted. Call fit() before predict().")

        valid_users = [u for u in user_to_pred if u in self.user_id_to_index]
        if not valid_users:
            return pl.DataFrame(columns=["cookie", "node", "scores"])

        # retrieve user embeddings and item embeddings
        user_indices = np.array([self.user_id_to_index[u] for u in valid_users])
        user_embs = self.model.user_embeddings[user_indices]
        item_embs = self.model.item_embeddings
        # normalize embeddings for cosine similarity
        faiss.normalize_L2(user_embs)
        faiss.normalize_L2(item_embs)
        dim = item_embs.shape[1]
        # build Faiss index
        index = faiss.IndexFlatIP(dim)
        index.add(item_embs)
        # search nearest neighbors
        D, I = index.search(user_embs, N)
        predictions = []
        for i, u in enumerate(valid_users):
            for rank, item_idx in enumerate(I[i]):
                predictions.append({
                    "cookie": u,
                    "node": self.index_to_item_id[int(item_idx)],
                    "scores": float(D[i][rank]),
                })
        df_pred = pl.DataFrame(predictions)
        return df_pred
