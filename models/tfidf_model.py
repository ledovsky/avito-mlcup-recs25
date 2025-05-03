import polars as pl
from scipy.sparse import csr_matrix
import numpy as np
from implicit.nearest_neighbours import TFIDFRecommender as _TfIdfRecommender
from implicit.nearest_neighbours import CosineRecommender as _CosineRecommender

class TfidfRecommender:
    """
    TF-IDF based recommender with a similar interface to ALSRecommender.
    """
    def __init__(self, df_events: pl.DataFrame, dedupe: bool = True, **kwargs):
        self.event_weights = {
            row["event"]: 2.0 if row["is_contact"] else 1.0
            for row in df_events.select(["event", "is_contact"]).to_dicts()
        }
        self.model = _TfIdfRecommender(K=40, **kwargs)
        # self.model = _CosineRecommender(K=40, **kwargs)
        self.dedupe = dedupe
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
            # collapse duplicate user-item pairs keeping only the maximum weight
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

        self.model.fit(self.sparse_matrix)
        col_sums = np.array(self.sparse_matrix.sum(axis=0)).ravel()
        popular_indices = np.argsort(-col_sums)
        self.popular_item_indices = popular_indices.tolist()

        self.seen = {
            user_id: set(self.sparse_matrix[idx].indices.tolist())
            for user_id, idx in self.user_id_to_index.items()
        }
        
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
        rec_lists = recommendations.tolist()
        for rec in rec_lists:
            popular_idx = 0
            seen = set()
            for idx, rec_idx in enumerate(rec):
                if rec_idx == -1:
                    while True:
                        popular_rec = self.popular_item_indices[popular_idx]
                        if popular_rec not in seen and popular_rec not in self.seen[valid_users[idx]]:
                            break
                        popular_idx += 1
                    rec[idx] = popular_rec
                    popular_idx += 1
                else:
                    seen.add(rec_idx)

        df_pred = pl.DataFrame(
            {
                "cookie": valid_users,
                "node": [[self.index_to_item_id[i] for i in recs] for recs in rec_lists],
                "scores": scores.tolist(),
            }
        )
        return df_pred.explode(["node", "scores"])
