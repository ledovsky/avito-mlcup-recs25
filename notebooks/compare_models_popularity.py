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
# Evaluation by bucket function
def eval_by_bucket(model, df_eval, N=40):
    preds = model.predict(df_eval['cookie'].to_list(), N=N)
    df_true = df_eval.select(['cookie','node']).with_columns(pl.lit(1).alias('true'))
    df_pred = preds.with_columns(pl.lit(1).alias('pred'))
    df_merge = df_true.join(df_pred, on=['cookie','node'], how='left').fill_null({'pred': 0})
    df_merge = df_merge.with_columns(pl.col('node').apply(bucket).alias('bucket'))
    result = {}
    for b in ['high','medium','low']:
        df_b = df_merge.filter(pl.col('bucket') == b)
        result[b] = df_b['pred'].sum() / df_b.shape[0] if df_b.shape[0] > 0 else None
    return result

# %%

# Compute item popularity from training data
pop_counts = df_train.group_by('node').len().rename({'len':'freq'})
pop_counts = pop_counts.sort('freq', descending=True)
total_items = pop_counts.shape[0]
quantiles = [0.33, 0.66]
freqs = pop_counts['freq'].to_list()
th1 = freqs[int(total_items * quantiles[0])]
th2 = freqs[int(total_items * quantiles[1])]
def bucket(node):
    f = pop_counts.filter(pl.col('node') == node)['freq'][0]
    if f >= th2:
        return 'high'
    elif f >= th1:
        return 'medium'
    else:
        return 'low'

# %%

# Run for ALS
als = ALSRecommender(df_events)
als.fit(df_train['cookie'], df_train['node'], df_train['event'])
preds = als.predict(df_eval['cookie'].to_list(), N=N)

# %%
res_als = eval_by_bucket(als, df_eval)


# %%

# Run for TF-IDF
tfidf = TfidfRecommender(df_events)
tfidf.fit(df_train['cookie'], df_train['node'], df_train['event'])

# %% 

res_tfidf = eval_by_bucket(tfidf, df_eval)

# %%
