#!/usr/bin/env python3
import argparse

import wandb

from utils import get_data, recall_at, make_pred_df
from models.implicit_model import ALSRecommender
from models.tfidf_model import TfidfRecommender
from models.lightfm_model import LightFMRecommender


def main():
    parser = argparse.ArgumentParser(
        description="Train/evaluate recommender and (optionally) produce submission"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["als", "tfidf", "lightfm"],
        help="Which model to run ('als', 'tfidf', or 'lightfm').",
    )
    parser.add_argument(
        "--submission",
        type=str,
        default=None,
        help="Path to write the final submission CSV. If omitted, skips retraining & submission.",
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
    df_test_users, df_clickstream, df_cat, df_text, df_events, df_train, df_eval = get_data()

    # 2) initialize model
    if args.model == "als":
        als_config = {
            "do_dedupe": False,
            "use_week_discount": False,
            "filter_rare_events": False,
            "contact_weight": 10,
            "als_factors": 120,
            "iterations": 20,
        }
        model = ALSRecommender(df_events, **als_config)
        run.config.update(als_config)
        model.fit(df_train)
    elif args.model == "tfidf":
        model = TfidfRecommender(df_events)
        model.fit(df_train["cookie"], df_train["node"], df_train["event"], df_train["week"])
    elif args.model == "lightfm":
        model = LightFMRecommender(df_events)
        model.fit(df_train["cookie"], df_train["node"], df_train["event"], df_train["week"])
    else:
        raise NotImplementedError(f"Model '{args.model}' is not implemented")

    # 3) fit on training split and evaluate
    print("Fitting on train split...")

    print(f"Generating top {TOP_K} predictions on eval split...")
    eval_preds = model.predict(df_eval["cookie"].to_list(), N=TOP_K)

    pred_df = make_pred_df(df_train, df_eval, eval_preds)
    pred_df.write_csv(f"pred_{run.name}.csv")

    recall = recall_at(df_eval, eval_preds, k=TOP_K)
    run.summary["Recall@{TOP_K}"] = recall
    print(f"Recall@{TOP_K} on eval: {recall:.4f}")

    # 4) optionally retrain on full data & write submission
    if args.submission:
        print("Retraining on full clickstream...")
        model.fit(df_clickstream["cookie"], df_clickstream["node"], df_clickstream["event"], df_clickstream["week"])

        print(f"Generating top {TOP_K} for test users...")
        test_users = df_test_users["cookie"].to_list()
        submission_df = model.predict(test_users, N=TOP_K).select(["cookie", "node"])

        print(f"Writing submission to {args.submission} …")
        submission_df.write_csv(args.submission)
        print("Submission written.")
    else:
        print("No --submission provided; skipping retraining & submission step.")

    print("Done.")

    run.finish()


if __name__ == "__main__":
    main()
