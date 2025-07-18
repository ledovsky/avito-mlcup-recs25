import polars as pl

import wandb
from main import fit_model, initialize_model


def get_test_data():
    df_train = pl.read_parquet("./test_data/train.pq")
    df_cat = pl.read_parquet("./test_data/cat_features.pq")
    df_events = pl.read_parquet("./test_data/events.pq")
    return df_train, df_cat, df_events


def test_popular():
    df_train, df_cat, df_events = get_test_data()
    run = wandb.init(mode="disabled")
    model = initialize_model("popular", run, df_events, df_cat)
    fit_model(model, "popular", df_train, df_events)


def test_als():
    df_train, df_cat, df_events = get_test_data()
    run = wandb.init(mode="disabled")
    model = initialize_model("als", run, df_events, df_cat)
    fit_model(model, "als", df_train, df_events)
    eval_cookies = df_train["cookie"].to_list()
    model.predict(eval_cookies, N=40)

def test_torch_emb_load_embeddings():
    df_train, df_cat, df_events = get_test_data()
    run = wandb.init(mode="disabled")
    model = initialize_model("torch-emb", run, df_events, df_cat)
    model.load_embeddings("./test_data", "peachy-voice")
    assert model.user_embeddings_np is not None
    assert model.item_embeddings_np is not None


def test_als_2():
    df_train, df_cat, df_events = get_test_data()
    run = wandb.init(mode="disabled")
    model = initialize_model("als-2", run, df_events, df_cat)
    fit_model(model, "als-2", df_train, df_events)
    eval_cookies = df_train["cookie"].to_list()
    model.predict(eval_cookies, N=40)


def test_lightfm():
    df_train, df_cat, df_events = get_test_data()
    run = wandb.init(mode="disabled")
    model = initialize_model("lightfm", run, df_events, df_cat)
    fit_model(model, "lightfm", df_train, df_events)


def test_torch_emb():
    df_train, df_cat, df_events = get_test_data()
    run = wandb.init(mode="disabled")
    model = initialize_model("torch-emb", run, df_events, df_cat)
    model.batch_size=50
    fit_model(model, "torch-emb", df_train, df_events)
    eval_cookies = df_train["cookie"].to_list()
    model.predict(eval_cookies, N=40)
