#!/usr/bin/env python3
import argparse

import polars as pl

import wandb
import wandb.wandb_run

from utils import get_data, recall_at, make_pred_df
from models.implicit_model import ALSRecommender
from models.tfidf_model import TfidfRecommender
from models.lightfm_model import LightFMRecommender
from models.popular import PopularLocCat, Popular
from models.base import BaseModel
from ranker.ranker import Ranker


def main():
    parser = argparse.ArgumentParser(
        description="Train/evaluate recommender and (optionally) produce submission"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["als", "tfidf", "lightfm", "popular-loc-cat", "popular"],
        help="Which model to run ('als', 'tfidf', 'lightfm', 'popular' or 'popular-loc-cat').",
    )
    args = parser.parse_args()

    # fixed number of recommendations
    TOP_K = 40

    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="alexander-ledovsky-just-myself",
        # Set the wandb project where this run will be logged.
        project="avito-mlcup-recs25",
        # Track hyperparameters and run metadata.
        config={
            "model": args.model,
        },
    )

    # 1) load data
    df_test_users, df_clickstream, df_cat, df_text, df_events, df_train, df_eval = (
        get_data()
    )

    model = fit(args.model, run, df_train, df_events, df_cat)

    # 3) fit on training split and evaluate
    print("Fitting on train split...")

    print(f"Generating top {TOP_K} predictions on eval split with ranker...")
    eval_preds = fit_ranker(model, df_clickstream, df_cat, df_train, df_eval, TOP_K)

    pred_df = make_pred_df(df_train, df_eval, eval_preds)
    pred_df.write_csv(f"pred_{run.name}.csv")

    recall = recall_at(df_eval, eval_preds, k=TOP_K)
    run.summary["Recall@{TOP_K}"] = recall
    print(f"Recall@{TOP_K} on eval: {recall:.4f}")

    print("Done.")

    run.finish()


def fit(
    model_name: str,
    run: wandb.wandb_run.Run,
    df_train: pl.DataFrame,
    df_events: pl.DataFrame,
    df_cat: pl.DataFrame,
) -> BaseModel:
    # 2) initialize model
    if model_name == "als":
        als_config = {
            "do_dedupe": False,
            "use_week_discount": False,
            "filter_rare_events": False,
            "contact_weight": 10,
            "als_factors": 60,
            "iterations": 10,
        }
        model = ALSRecommender(**als_config)
        run.config.update(als_config)
        model.fit(df_train, df_events)
    elif model_name == "tfidf":
        model = TfidfRecommender(df_events)
        model.fit(
            df_train["cookie"], df_train["node"], df_train["event"], df_train["week"]
        )
    elif model_name == "lightfm":
        model = LightFMRecommender(df_events)
        model.fit(
            df_train["cookie"], df_train["node"], df_train["event"], df_train["week"]
        )
    elif model_name == "popular-loc-cat":
        model = PopularLocCat(df_cat)
        model.fit(df_train)
    elif model_name == "popular":
        model = Popular()
        model.fit(df_train)
    else:
        raise NotImplementedError(f"Model '{model_name}' is not implemented")

    return model


def fit_ranker(
    model: BaseModel,
    df_clickstream: pl.DataFrame,
    df_cat: pl.DataFrame,
    df_train: pl.DataFrame,
    df_eval: pl.DataFrame,
    N: int,
) -> pl.DataFrame:
    """Train a per-user ranker on model predictions and generate final eval predictions."""
    # Generate base predictions on training split
    train_cookies = df_train["cookie"].sample(10_000).to_list()
    df_pred_train = model.predict(train_cookies, N=40)
    # Initialize and train ranker
    ranker = Ranker(df_clickstream, df_cat)
    ranker.fit(df_pred_train)
    # Generate base predictions on evaluation split
    eval_cookies = df_eval["cookie"].to_list()
    df_pred_eval_base = model.predict(eval_cookies, N=3*N)
    # Apply ranker to evaluation predictions
    df_pred_eval = ranker.predict(df_pred_eval_base)
    return df_pred_eval

if __name__ == "__main__":
    main()
