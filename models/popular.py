import polars as pl

from .base import BaseModel


class PopularLocCat(BaseModel):
    def __init__(self, df_cat):
        self.df_cat = df_cat
        self.populars = None
        self.recs_dict = None

    def fit(self, df_train: pl.DataFrame):
        self.df_cat.sort("node")
        df_train.group_by("node").agg(pl.col("cookie").count())
        counts = df_train.group_by("node").agg(pl.len().alias("count"))
        counts_by_loc_cat = (
            self.df_cat.select("node", "location", "category")
            .unique()
            .join(counts, on="node", how="left")
        )
        counts_by_loc_cat = counts_by_loc_cat.fill_null(0)

        top5_by_loc_cat = (
            counts_by_loc_cat.sort(
                ["location", "category", "count"], descending=[False, False, True]
            )  # sort by location, then count descending
            .group_by(["location", "category"], maintain_order=True)
            .head(5)  # take top 5 rows per location
        )

        node2node = (
            counts_by_loc_cat.select("node", "location", "category")
            .join(
                top5_by_loc_cat.rename({"node": "recommended_node"}),
                on=["location", "category"],
            )
            .select("node", "recommended_node", "count")
            .sort(["node", "count"], descending=[False, True])
            .group_by("node", maintain_order=True)
            .head(40)
        )

        recs = (
            df_train.select("cookie", "node")
            .unique()
            .join(node2node, on="node")
            .unique(subset=["cookie", "recommended_node"])
            .sort(["cookie", "count"], descending=[False, True])
            .group_by("cookie", maintain_order=True)
            .head(40)
            .group_by("cookie", maintain_order=True)
            .agg(pl.col("recommended_node").alias("recs"))
        )

        recs_dict = {}
        for cookie, recs in recs.iter_rows():
            recs_dict[cookie] = recs

        self.recs_dict = recs_dict
        self.populars = counts.sort("count", descending=True).head(100)["node"].to_list()

    def predict(self, user_to_pred: list[int | str], N: int = 40) -> pl.DataFrame:
        """
        Predict top N recommendations for given users.
        """
        records: list[tuple[int | str, list[int | str]]] = []
        for user in user_to_pred:
            recs = list(self.recs_dict.get(user, []))
            if len(recs) < N:
                # fill up with popular items
                fill_items = [item for item in self.populars if item not in recs]
                recs += fill_items[: N - len(recs)]
            else:
                recs = recs[:N]
            records.append((user, recs))
        df = pl.DataFrame(records, schema=["cookie", "recs"])
        return df.explode("recs").rename({"recs": "node"})


def get_popular(df, eval_users):
    popular_node = (
        df.group_by("node")
        .agg(pl.col("cookie").count())
        .sort("cookie")
        .tail(40)["node"]
        .to_list()
    )
    df_pred_pop = pl.DataFrame(
        {"node": [popular_node for i in range(len(eval_users))], "cookie": eval_users}
    )
    df_pred_pop = df_pred_pop.explode("node")
    return df_pred_pop
