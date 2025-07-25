from typing import Self

import faiss
import joblib
import numpy as np
import os
import polars as pl
import torch
from scipy.sparse import csr_matrix
from tqdm import tqdm


class BaseModel:
    def __init__(self, *args, **kwargs) -> None:
        self.contact_weight: int = 1
        self.user_id_to_index: dict[str, int] = {}
        self.item_id_to_index: dict[int, int] = {}
        self.index_to_item_id: dict[int, int] = {}
        self.sparse_matrix: csr_matrix = None
        self.user_embeddings_np: np.typing.NDArray | None = None
        self.item_embeddings_np: np.typing.NDArray | None = None
        self.num_users: int | None = None
        self.num_items: int | None = None

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

    def filter_top_k_items(self, df: pl.DataFrame, top_k_items: int) -> pl.DataFrame:
        top_items = (
            df["node"]
            .value_counts()
            .sort("count", descending=True)
            .head(top_k_items)["node"]
            .to_list()
        )
        return df.filter(pl.col("node").is_in(top_items))

    def get_seen_nodes(self, df_train: pl.DataFrame) -> dict[int, list[int]]:
        seen_nodes_df = df_train.group_by("cookie").agg(
            pl.col("node").unique().alias("seen_nodes")
        )
        seen_nodes: dict[int, list[int]] = {
            row["cookie"]: row["seen_nodes"] for row in seen_nodes_df.to_dicts()
        }
        return seen_nodes

    def get_populars(self, df_train: pl.DataFrame) -> list[int]:
        return (
            df_train.group_by("node")
            .agg(pl.col("cookie").len().alias("len"))
            .sort("len", descending=True)
            .get_column("len")
            .to_list()[:1000]
        )

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
        self.fill_indices(df)

        rows = df["cookie"].replace_strict(self.user_id_to_index).to_list()
        cols = df["node"].replace_strict(self.item_id_to_index).to_list()


        base_vals = [
                self.contact_weight if is_contact else 1
                for is_contact in df["is_contact"].to_list()
        ]
        
        if use_week_discount and "week" in df.columns:
            weeks = df["week"].to_list()
            values = [
                base_vals[i] * (2 if weeks[i] >= 4 else 1) for i in range(len(rows))
            ]
        else:
            values = base_vals

        self.sparse_matrix = csr_matrix(
            (values, (rows, cols)), shape=(self.num_users, self.num_items)
        )
    
    def fill_indices(self, df):
        """Fills:
        - self.user_id_to_index,
        - self.item_id_to_index,
        - self.index_to_item_id,
        """
        users = df["cookie"].unique().to_list()
        items = df["node"].unique().to_list()
        self.user_id_to_index = {u: i for i, u in enumerate(users)}
        self.item_id_to_index = {j: i for i, j in enumerate(items)}
        self.index_to_item_id = {i: j for j, i in self.item_id_to_index.items()}
        self.num_users = len(users)
        self.num_items = len(items)

    def set_embeddings(self, user_embeddings_np, item_embeddings_np):
        self.user_embeddings_np = user_embeddings_np
        self.item_embeddings_np = item_embeddings_np

    def set_user_mappings(self, user_id_to_index, index_to_item_id):
        self.user_id_to_index = user_id_to_index
        self.index_to_item_id = index_to_item_id

    def save_embeddings(self, path: str, run_name: str) -> None:
        np.save(os.path.join(path, f"{run_name}-user-emb.npy"), self.user_embeddings_np)
        np.save(os.path.join(path, f"{run_name}-item-emb.npy"), self.item_embeddings_np)

    def load_embeddings(self, path: str, run_name: str) -> None:
        self.user_embeddings_np = np.load(os.path.join(path, f"{run_name}-user-emb.npy"))
        self.item_embeddings_np = np.load(os.path.join(path, f"{run_name}-item-emb.npy"))


class BaseTorchModel(BaseModel):
    def save(self, path: str) -> None:
        torch.save(self, path)

    @classmethod
    def load(cls, path: str, weights_only: bool = True) -> Self:
        return torch.load(path, weights_only=weights_only)


class FaissPredict(BaseModel):
    """
    Mixin providing FAISS-based predict functionality.
    """

    def __init__(self, *args, **kwargs):
        self.seen_items: dict[int, set[int]] = {}
        self.populars: list[int] = []
        self.is_cos_dist: bool = False
        super().__init__(*args, **kwargs)

    def set_seen_items(self, seen_items: dict[int, set[int]]):
        self.seen_items = seen_items

    def set_populars(self, populars: list[int]):
        self.populars = populars

    def set_is_cos_dist(self, is_cos_dist: bool):
        self.is_cos_dist = is_cos_dist

    def predict(
        self, user_to_pred: list[int | str], N: int = 40, batch_size: int = 100
    ) -> pl.DataFrame:
        """
        Predict top-N items per user using FAISS HNSW index and return as Polars DataFrame.
        """

        dim = self.user_embeddings_np.shape[1]
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        item_embs = self.item_embeddings_np.astype(np.float32)
        user_embs = self.user_embeddings_np.astype(np.float32)

        if self.is_cos_dist:
            # normalize
            item_embs /= np.linalg.norm(item_embs, axis=1, keepdims=True)
            user_embs /= np.linalg.norm(user_embs, axis=1, keepdims=True)

        index.add(item_embs)

        all_user_ids, all_item_ids, all_scores = [], [], []

        for i in tqdm(
            range(0, len(user_to_pred), batch_size),
            total=len(user_to_pred) // batch_size,
        ):
            batch_user_ids = user_to_pred[i : i + batch_size]
            batch_user_ids = [u for u in batch_user_ids if u in self.user_id_to_index]

            user_indices = [self.user_id_to_index[u] for u in batch_user_ids]
            batch_embs = user_embs[user_indices]
            scores, indices = index.search(batch_embs, N * 3)

            for j, user_id in enumerate(batch_user_ids):
                seen = set(self.seen_items.get(user_id, []))
                recs, rec_scores = [], []
                for idx, score in zip(indices[j], scores[j]):
                    item_id = self.index_to_item_id[idx]
                    if item_id not in seen and item_id not in recs:
                        recs.append(item_id)
                        rec_scores.append(score)
                    if len(recs) >= N:
                        break
                if len(recs) < N:
                    for pid in self.populars:
                        if pid not in seen and pid not in recs:
                            recs.append(pid)
                            rec_scores.append(0.0)
                        if len(recs) >= N:
                            break
                recs, rec_scores = recs[:N], rec_scores[:N]
                assert (len(recs) == N)
                all_user_ids += [user_id] * N
                all_item_ids += recs
                all_scores += rec_scores

        return pl.DataFrame(
            {"cookie": all_user_ids, "node": all_item_ids, "scores": all_scores}
        )

    def get_hard_negs(self, train_df: pl.DataFrame, user_list: list[int | str], top_k: int) -> pl.DataFrame:
        """
        Return hard negative samples by predicting top_k for each user and filtering out interactions present in train_df.
        """
        # get top-k predictions per user
        preds = self.predict(user_list, N=top_k)
        # remove any (cookie, node) present in train_df
        negs = preds.join(
            train_df.select(['cookie', 'node']).unique(),
            on=['cookie', 'node'],
            how='anti'
        )
        # return only cookie and node_id columns
        return negs.select(['cookie', pl.col('node')])
