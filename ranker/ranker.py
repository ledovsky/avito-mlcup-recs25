import polars as pl
from catboost import CatBoostRanker, CatBoostError
from ranker.features import enrich, add_target


class Ranker:

    default_params = {
        "iterations": 50,
    }

    def __init__(
        self,
        df_clickstream: pl.DataFrame,
        df_cat: pl.DataFrame,
        catboost_params: dict | None = None,
    ):
        self.df_clickstream = df_clickstream
        self.df_cat = df_cat
        params = self.default_params.copy()
        if catboost_params:
            params.update(catboost_params)
        self.model = CatBoostRanker(**params)

    def fit(self, df_pred: pl.DataFrame) -> None:
        # prepare training data with target
        df_train = enrich(df_pred, self.df_clickstream, self.df_cat)
        df_train = add_target(df_train, self.df_clickstream)
        # ensure queryIds are grouped for CatBoostRanker
        df_train = df_train.sort("user_id")
        y = df_train["target"].to_list()
        group_id = df_train["user_id"].to_list()
        X = df_train.drop(["target", "user_id"])
        # fit CatBoost
        self.model.fit(X.to_pandas(), group_id=group_id,  y=y, verbose=True)

    def predict(self, df_pred: pl.DataFrame, N: int = 40) -> pl.DataFrame:
        # prepare feature data without target
        df_feat = enrich(df_pred, self.df_clickstream, self.df_cat)
        user_ids = df_feat["user_id"].to_list()
        df_feat = df_feat.drop(["user_id"])
        preds = self.model.predict(df_feat.to_pandas())
        df_out = pl.DataFrame(
            {
                "cookie": user_ids,
                "node": df_feat["node"].to_list(),
                "score": preds.tolist() if hasattr(preds, "tolist") else list(preds),
            }
        )
        # select top N scores per cookie
        df_out = df_out.sort("score", descending=True).group_by("cookie", maintain_order=True).head(N)
        return df_out
