import polars as pl
from scipy.sparse import csr_matrix
import numpy as np
from lightfm import LightFM

class LightFMRecommender:
    """
    LightFM-based recommender using LightFM library.
    """
    def __init__(self, df_events: pl.DataFrame, no_components: int = 60, epochs: int = 10, dedupe: bool = True, **lfm_kwargs):
        self.lfm_kwargs = {"no_components": no_components, "epochs": epochs, **lfm_kwargs}
        self.dedupe = dedupe
        self.event_weights = {
            row["event"]: 4 if row["is_contact"] else 1
            for row in df_events.select(["event", "is_contact"]).to_dicts()
        }
        self.model = None
        self.user_id_to_index = {}
        self.item_id_to_index = {}
        self.index_to_item_id = {}
        self.sparse_matrix = None

    def fit(self, users: pl.Series, nodes: pl.Series, events: pl.Series) -> None:
        user_ids = users.unique().to_list()
        item_ids = nodes.unique().to_list()

        self.user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self.item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
        self.index_to_item_id = {v: k for k, v in self.item_id_to_index.items()}

        rows = users.replace_strict(self.user_id_to_index).to_list()
        cols = nodes.replace_strict(self.item_id_to_index).to_list()
        values = [self.event_weights.get(ev, 1) for ev in events.to_list()]

        if self.dedupe:
            pair_max = {}
            for u_idx, i_idx, w in zip(rows, cols, values):
                key = (u_idx, i_idx)
                pair_max[key] = max(pair_max.get(key, w), w)
            final_rows, final_cols, final_vals = zip(*[(u, i, w) for (u, i), w in pair_max.items()])
            self.sparse_matrix = csr_matrix(
                (final_vals, (final_rows, final_cols)), shape=(len(user_ids), len(item_ids))
            )
        else:
            self.sparse_matrix = csr_matrix(
                (values, (rows, cols)), shape=(len(user_ids), len(item_ids))
            )

        # Initialize and train LightFM model
        self.model = LightFM(no_components=self.lfm_kwargs["no_components"], **{k: v for k, v in self.lfm_kwargs.items() if k not in ["no_components", "epochs"]})
        self.model.fit(self.sparse_matrix, epochs=self.lfm_kwargs["epochs"], num_threads=4)

    def predict(self, user_to_pred: list[int | str], N: int = 40) -> pl.DataFrame:
        if self.model is None or self.sparse_matrix is None:
            raise ValueError("Model is not fitted. Call fit() before predict().")

        valid_users = [u for u in user_to_pred if u in self.user_id_to_index]
        if not valid_users:
            return pl.DataFrame(columns=["cookie", "node", "scores"])

        predictions = []
        for u in valid_users:
            u_idx = self.user_id_to_index[u]
            scores = self.model.predict(u_idx, np.arange(self.sparse_matrix.shape[1]))
            top_indices = np.argpartition(-scores, N)[:N]
            top_scores = scores[top_indices]
            items = [self.index_to_item_id[i] for i in top_indices]
            for item, score in zip(items, top_scores):
                predictions.append({"cookie": u, "node": item, "scores": score})

        df_pred = pl.DataFrame(predictions)
        return df_pred
