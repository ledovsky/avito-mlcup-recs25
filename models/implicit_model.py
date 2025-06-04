import polars as pl
from scipy.sparse import csr_matrix
import numpy as np
import implicit
from .base import BaseModel


class ALSRecommender(BaseModel):
    """
    ALS-based recommender using implicit library.
    """

    def __init__(
        self,
        df_events: pl.DataFrame,
        do_dedupe: bool,
        use_week_discount: bool,
        filter_rare_events: bool,
        contact_weight: int,
        als_factors: int,
        iterations: int,
        top_k_items: int | None = None,
    ):
        # coniguration
        self.als_kwargs = {
            "factors": als_factors,
            "iterations": iterations,
        }
        self.do_dedupe = do_dedupe
        self.use_week_discount = use_week_discount
        self.filter_rare_events = filter_rare_events

        self.event_weights = {
            row["event"]: contact_weight if row["is_contact"] else 1
            for row in df_events.select(["event", "is_contact"]).to_dicts()
        }

        self.model = None
        self.user_id_to_index = {}
        self.item_id_to_index = {}
        self.index_to_item_id = {}
        self.sparse_matrix = None
        self.top_k_items = top_k_items

    def fit(self, df_train: pl.DataFrame) -> None:
        if self.filter_rare_events:
            df_train = self.filter_rare_events(df_train)

        if self.do_dedupe:
            df_train = self.dedupe(df_train)

        df_train = self.filter_top_k_items(df_train, self.top_k_items)

        (
            self.user_id_to_index,
            self.item_id_to_index,
            self.index_to_item_id,
            self.sparse_matrix,
        ) = self.build_interaction_matrix(
            df_train,
            event_weights=self.event_weights,
            use_week_discount=self.use_week_discount,
        )

        self.model = implicit.als.AlternatingLeastSquares(**self.als_kwargs)
        self.model.fit(self.sparse_matrix)

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
