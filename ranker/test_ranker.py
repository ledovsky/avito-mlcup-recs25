import polars as pl
import pytest
import random

from ranker.ranker import Ranker

def test_ranker_fit_predict_trivial():
    # load clickstream and category features from test_data
    df_click = pl.read_parquet("./test_data/train.pq")
    df_cat = pl.read_parquet("./test_data/cat_features.pq")
    # build dummy predictions: one random node per user, zero score
    cookies = df_click["cookie"].unique().to_list()
    nodes = df_click["node"].unique().to_list()
    pairs = [(c, random.choice(nodes)) for _ in range(10) for c in cookies]
    nodes = [n for _, n in pairs]
    scores = [0.0] * len(pairs)
    df_pred = pl.DataFrame({
        "cookie": [c for c, _ in pairs],
        "node":   [n for _, n in pairs],
        "scores": scores
    })
    # use minimal iterations for a constant model
    ranker = Ranker(df_click, df_cat, catboost_params={"iterations": 1, "depth": 1})
    ranker.fit(df_pred)
    preds = ranker.predict(df_pred)

    assert set(preds["cookie"].to_list()) == set(cookies)
    assert set(preds["node"].to_list()) == set(df_pred["node"].to_list())
    assert len(preds["score"]) == len(df_pred)
