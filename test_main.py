import polars as pl
import wandb


from main import fit

def get_test_data():
    df_train = pl.read_parquet("./test_data/train.pq")
    df_cat = pl.read_parquet("./test_data/cat_features.pq")
    df_events = pl.read_parquet("./test_data/events.pq")
    return df_train, df_cat, df_events


def test_popular():
    df_train, df_cat, df_events = get_test_data()
    run = wandb.init(mode="disabled")
    fit("popular", run, df_train, df_events, df_cat)

def test_als():
    df_train, df_cat, df_events = get_test_data()
    run = wandb.init(mode="disabled")
    fit("als", run, df_train, df_events, df_cat)
