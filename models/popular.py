import polars as pl

from .base import BaseModel


class Popular(BaseModel):
    def fit(self, df_train):
        self.popular_nodes = (
            df_train.group_by("node")
            .agg(pl.col("cookie").count())
            .sort("cookie")
            .tail(40)["node"]
            .to_list()
        )

    def predict(self, user_to_pred: list[int | str], N: int = 40) -> pl.DataFrame:
        df_pred_pop = pl.DataFrame(
            {"node": [self.popular_nodes for i in range(len(user_to_pred))], "cookie": user_to_pred}
        )
        return df_pred_pop.explode("node")
