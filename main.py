#!/usr/bin/env python3
import argparse

from utils import get_data, recall_at
from models.implicit_model import ALSRecommender


def main():
    parser = argparse.ArgumentParser(
        description="Train/evaluate recommender and (optionally) produce submission"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["als"],
        help="Which model to run (only 'als' supported today).",
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

    # 1) load data
    df_test_users, df_clickstream, df_cat, df_text, df_events, df_train, df_eval = get_data()

    # 2) initialize model
    if args.model == "als":
        model = ALSRecommender(df_events)
    else:
        raise NotImplementedError(f"Model '{args.model}' is not implemented")

    # 3) fit on training split and evaluate
    print("Fitting on train split...")
    model.fit(df_train["cookie"], df_train["node"], df_train["event"])

    print(f"Generating top {TOP_K} predictions on eval split...")
    eval_preds = model.predict(df_eval["cookie"].to_list(), N=TOP_K)

    recall = recall_at(df_eval, eval_preds, k=TOP_K)
    print(f"Recall@{TOP_K} on eval: {recall:.4f}")

    # 4) optionally retrain on full data & write submission
    if args.submission:
        print("Retraining on full clickstream...")
        model.fit(df_clickstream["cookie"], df_clickstream["node"], df_clickstream["event"])

        print(f"Generating top {TOP_K} for test users...")
        test_users = df_test_users["cookie"].to_list()
        submission_df = model.predict(test_users, N=TOP_K).select(["cookie", "node"])

        print(f"Writing submission to {args.submission} …")
        submission_df.write_csv(args.submission)
        print("Submission written.")
    else:
        print("No --submission provided; skipping retraining & submission step.")

    print("Done.")


if __name__ == "__main__":
    main()
