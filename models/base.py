import polars as pl


class BaseModel:
    def filter_rare_events(self, df_train: pl.DataFrame, thr=3) -> pl.DataFrame:
        return (
            df_train
            .group_by(["cookie", "node"])
            .agg(pl.len().alias("event_count"))
            .filter(pl.col("event_count") >= thr)
            .select("cookie", "node")
            .join(df_train, on=["cookie", "node"], how="inner")
        )
    
    def dedupe(self, df_train: pl.DataFrame) -> pl.DataFrame():
        return (
            df_train
            .sort(["cookie", "node", "is_contact", "event_date"], descending=[False, False, False, True])
            .unique(subset=["cookie", "node"])
        )