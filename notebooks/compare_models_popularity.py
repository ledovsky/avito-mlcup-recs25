# %%
import sys; sys.path.append("../")
import polars as pl
from utils import get_data
from models.implicit_model import ALSRecommender
from models.tfidf_model import TfidfRecommender
import numpy as np

# %%
df_test_users, df_clickstream, df_cat, df_text, df_events, df_train, df_eval = get_data(data_dir="../data")

# %%

# Compute item popularity from training data
pop_counts = df_train.select("cookie", "node").unique().group_by('node').len().rename({'len':'freq'})
pop_counts = pop_counts.sort('freq', descending=True)
total_items = pop_counts.shape[0]
quantiles = [0.33, 0.66]
freqs = pop_counts['freq'].to_list()
th1 = freqs[int(total_items * quantiles[0])]
th2 = freqs[int(total_items * quantiles[1])]

# Evaluation by bucket function
def bucket(freq):
    if freq >= th2:
        return 'high'
    elif freq >= th1:
        return 'medium'
    else:
        return 'low'

# %%

def eval_by_bucket(df_eval: pl.DataFrame, preds: pl.Series):
    df_true = df_eval.select(['cookie','node']).with_columns(pl.lit(1).alias('true'))
    df_pred = preds.with_columns(pl.lit(1).alias('pred'))
    df_merge = df_true.join(df_pred, on=['cookie','node'], how='left').with_columns(pl.col("pred").fill_nan(0))
    df_merge = df_merge.join(pop_counts, "node", how="left")
    df_merge = df_merge.with_columns(pl.col("freq").map_elements(bucket).alias("bucket"))
    result = {}
    for b in ['high','medium','low']:
        df_b = df_merge.filter(pl.col('bucket') == b)
        result[b] = df_b['pred'].sum() / df_b.shape[0] if df_b.shape[0] > 0 else None
    return result, df_merge


# %%

# Run for ALS
als = ALSRecommender(df_events)
als.fit(df_train['cookie'], df_train['node'], df_train['event'])

# %%
als_preds = als.predict(df_eval['cookie'].to_list(), N=40)

# %%
res_als, df_merge = eval_by_bucket(df_eval, als_preds)


# %%

# Run for TF-IDF
tfidf = TfidfRecommender(df_events)
tfidf.fit(df_train['cookie'], df_train['node'], df_train['event'])

# %% 

res_tfidf = eval_by_bucket(tfidf, df_eval)

# %%

df_merge.group_by("bucket").agg(pl.len())
# int(total_items * quantiles[0])

# %%
pop_counts
# %%


nodes_train = df_train.group_by("node").agg(pl.len().alias("n_train"))
nodes_eval = df_eval.group_by("node").agg(pl.len().alias("n_eval"))

nodes_stats = (
    nodes_train
    .join(nodes_eval, "node", "left")
    .join(pop_counts, "node", "left")
    .with_columns(pl.col("freq").map_elements(bucket).alias("bucket"))
)


nodes_stats.group_by("")