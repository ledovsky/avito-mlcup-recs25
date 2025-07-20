#!/usr/bin/env python3
import argparse
import os
import time
from typing import Optional

import polars as pl

import wandb
import wandb.wandb_run
from models.base import BaseModel
from models.implicit_model import ALSRecommender, ALS2
from models.lightfm_model import LightFMRecommender
from models.popular import Popular
from models.tfidf_model import TfidfRecommender
from models.pytorch_emb import TorchEmbModel, TorchEmbCrossEnt
from ranker.ranker import Ranker
from utils import TimeTracker, get_data, recall_at

MODEL_PATH = "checkpoints"


def main():
    parser = argparse.ArgumentParser(
        description="Train/evaluate recommender and ranker models"
    )

    # Create subparsers for different tasks
    subparsers = parser.add_subparsers(dest="task", help="Task to perform")

    # 1. Train candidate generation model
    train_candgen_parser = subparsers.add_parser(
        "train-candgen", help="Train candidate generation model"
    )
    train_candgen_parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["als", "als-2", "tfidf", "lightfm", "popular-loc-cat", "popular", "torch-emb", "torch-emb-ce"],
        help="Which model to run",
    )
    train_candgen_parser.add_argument(
        "--save-path",
        type=str,
        help="Path to save the trained model",
        default="./trained",
    )
    train_candgen_parser.add_argument(
        "--load-emb",
        type=str,
        default="",
        help="Run name to load embeddings from",
    )

    # 2. Create ranking dataset
    create_ranking_dataset_parser = subparsers.add_parser(
        "create-ranking-dataset", help="Create dataset for ranker training"
    )
    create_ranking_dataset_parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the pretrained candidate generation model",
    )
    create_ranking_dataset_parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["als", "tfidf", "lightfm", "popular-loc-cat", "popular", "torch-emb"],
        help="Type of the pretrained model",
    )
    create_ranking_dataset_parser.add_argument(
        "--output-path",
        type=str,
        default="./data/ranking_dataset.parquet",
        help="Path to save the ranking dataset",
    )

    # 3. Train ranker
    train_ranker_parser = subparsers.add_parser(
        "train-ranker", help="Train ranker model"
    )
    train_ranker_parser.add_argument(
        "--candgen-model-path",
        type=str,
        required=True,
        help="Path to the pretrained candidate generation model",
    )
    train_ranker_parser.add_argument(
        "--candgen-model-type",
        type=str,
        required=True,
        choices=["als", "tfidf", "lightfm", "popular-loc-cat", "popular", "torch-emb"],
        help="Type of the pretrained candidate generation model",
    )
    train_ranker_parser.add_argument(
        "--ranking-dataset",
        type=str,
        help="Path to the pregenerated ranking dataset (optional)",
    )

    args = parser.parse_args()

    # fixed number of recommendations
    TOP_K = 40

    is_offline = os.getenv("WANDB_MODE", "").lower() == "offline"
    run_name = f"offline-{int(time.time())}" if is_offline else None

    # Start a new wandb run to track this script.
    run = wandb.init(
        entity="alexander-ledovsky-just-myself",
        project="avito-mlcup-recs25",
        name=run_name,
        config={
            "task": args.task,
        },
    )
    print(f"Wandb run name - {run.name}, run id - {run.id}")

    # Initialize time tracker
    timer = TimeTracker(run)

    # Load data
    timer.start("data_loading")
    df_test_users, df_clickstream, df_cat, df_text, df_events, df_train, df_eval = (
        get_data()
    )
    timer.stop("data_loading")

    if args.task == "train-candgen":
        timer.start("train_candgen")
        train_candidate_generation_model(
            args, run, df_train, df_events, df_cat, df_eval, TOP_K
        )
        timer.stop("train_candgen")
    elif args.task == "create-ranking-dataset":
        timer.start("create_ranking_dataset")
        create_ranking_dataset(args, df_train, df_events, df_cat, df_clickstream)
        timer.stop("create_ranking_dataset")
    elif args.task == "train-ranker":
        timer.start("train_ranker")
        train_ranker_model(
            args, run, df_train, df_events, df_cat, df_clickstream, df_eval, TOP_K
        )
        timer.stop("train_ranker")
    else:
        parser.print_help()

    timer.print_total()
    run.finish()


def train_candidate_generation_model(
    args,
    run: wandb.wandb_run.Run,
    df_train: pl.DataFrame,
    df_events: pl.DataFrame,
    df_cat: pl.DataFrame,
    df_eval: pl.DataFrame,
    top_k: int,
):
    """Train a candidate generation model and evaluate it on validation data"""
    run.config.update({"model": args.model})

    # Initialize and train model
    timer = TimeTracker(run)

    timer.start("model_init")
    model = initialize_model(args.model, run, df_events, df_cat)
    if args.load_emb:
        run.config.update({"embed_from": args.load_emb})
        model.load_embeddings("embeddings", args.load_emb)
        print(f"Loaded embeddings from embeddings/{args.load_emb}-user-emb.npy and embeddings/{args.load_emb}-item-emb.npy")
    timer.stop("model_init")

    timer.start("model_fit")
    fit_model(model, args.model, df_train, df_events)
    timer.stop("model_fit")

    # Save model
    timer.start("model_save")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    model_path = f"{MODEL_PATH}/{args.model}-{run.name}.model"
    run.summary["model_path"] = model_path
    model.save(model_path)
    print(f"Model saved to {model_path}")
    timer.stop("model_save")

    # Save embeddings
    timer.start("embeddings_save")
    os.makedirs("embeddings", exist_ok=True)
    model.save_embeddings("embeddings", run.name)
    print(f"Embeddings saved to embeddings/{run.name}-user-emb.npy and embeddings/{run.name}-item-emb.npy")
    timer.stop("embeddings_save")

    # Evaluate model
    timer.start("model_eval")
    eval_cookies = df_eval["cookie"].to_list()
    eval_preds = model.predict(eval_cookies, N=top_k)

    recall = recall_at(df_eval, eval_preds, k=top_k)
    run.summary[f"Recall@{top_k}"] = recall
    print(f"Recall@{top_k} on eval: {recall:.4f}")
    timer.stop("model_eval")


def create_ranking_dataset(
    args,
    df_train: pl.DataFrame,
    df_events: pl.DataFrame,
    df_cat: pl.DataFrame,
    df_clickstream: pl.DataFrame,
):
    """Create a dataset for training the ranker"""
    # Load pretrained model
    model = load_model(args.model_type, args.model_path, df_events, df_cat)

    # Generate predictions for a sample of training users
    train_cookies = df_train["cookie"].sample(10_000).to_list()
    df_pred_train = model.predict(train_cookies, N=40)

    # Enrich predictions with features for ranking
    ranker = Ranker(df_clickstream, df_cat)
    df_ranking_dataset = ranker.prepare_dataset(df_pred_train)

    # Save the dataset
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df_ranking_dataset.write_parquet(args.output_path)
    print(f"Ranking dataset saved to {args.output_path}")


def train_ranker_model(
    args,
    run: wandb.wandb_run.Run,
    df_train: pl.DataFrame,
    df_events: pl.DataFrame,
    df_cat: pl.DataFrame,
    df_clickstream: pl.DataFrame,
    df_eval: pl.DataFrame,
    top_k: int,
):
    """Train a ranker model and evaluate the combined candidate generation + ranking approach"""
    run.config.update({"candgen_model": args.candgen_model_type})

    # Load pretrained candidate generation model
    model = load_model(
        args.candgen_model_type, args.candgen_model_path, df_events, df_cat
    )

    # Either load existing ranking dataset or create a new one
    if args.ranking_dataset and os.path.exists(args.ranking_dataset):
        df_ranking_dataset = pl.read_parquet(args.ranking_dataset)
    else:
        # Generate predictions for a sample of training users
        train_cookies = df_train["cookie"].sample(10_000).to_list()
        df_pred_train = model.predict(train_cookies, N=40)

        # Initialize ranker
        ranker = Ranker(df_clickstream, df_cat)
        df_ranking_dataset = df_pred_train

    # Initialize and train ranker
    ranker = Ranker(df_clickstream, df_cat)
    ranker.fit(df_ranking_dataset)

    # Generate base predictions on evaluation split
    eval_cookies = df_eval["cookie"].to_list()
    df_pred_eval_base = model.predict(eval_cookies, N=3 * top_k)

    # Apply ranker to evaluation predictions
    df_pred_eval = ranker.predict(df_pred_eval_base, N=top_k)

    # Evaluate
    recall = recall_at(df_eval, df_pred_eval, k=top_k)
    run.summary[f"Recall@{top_k}"] = recall
    print(f"Recall@{top_k} on eval with ranker: {recall:.4f}")


def initialize_model(
    model_name: str,
    run: wandb.wandb_run.Run,
    df_events: pl.DataFrame,
    df_cat: pl.DataFrame,
) -> BaseModel:
    """Initialize a model based on the model name"""
    if model_name == "als":
        als_config = {
            "dedupe": False,
            "use_week_discount": False,
            "filter_rare_events": False,
            "contact_weight": 10,
            "embedding_dim": 64,
            "iterations": 10,
            "top_k_items": 40_000,
        }
        run.config.update(als_config)
        return ALSRecommender(run, **als_config)
    if model_name == "als-2":
        als_config = {
            "dedupe": False,
            "use_week_discount": False,
            "filter_rare_events": False,
            "contact_weight": 10,
            "embedding_dim": 120,
            "iterations": 10,
            "top_k_items": 40_000,
        }
        run.config.update(als_config)
        return ALS2(run, **als_config)
    elif model_name == "tfidf":
        return TfidfRecommender(df_events)
    elif model_name == "lightfm":
        lightfm_config = {
            "epochs": 10,
            "top_k_items": 40_000,
            # "loss": "bpr",
            "loss": "warp",
            "no_components": 64,
        }
        return LightFMRecommender(run, **lightfm_config)
    elif model_name == "torch-emb":
        torchemb_config = {
            "embedding_dim": 64,
            "epochs": 12,
            "batch_size": 4096,
            "lr": 1e-3,
            "alpha": 0.01,
            "top_k_items": 40_000,
            "k_inbatch_negs": 200,
            "dedupe": True,
            "loss_fn": "BCEWithLogits"
        }
        run.config.update(torchemb_config)
        model = TorchEmbModel(run, **torchemb_config)
        run.summary["device"] = str(model.device)
        return model
    elif model_name == "torch-emb-ce":
        torchemb_config = {
            "embedding_dim": 64,
            "epochs": 12,
            "batch_size": 4096,
            "lr": 1e-3,
            "alpha": 0.01,
            "top_k_items": 40_000,
            "k_inbatch_negs": 200,
            "dedupe": True,
            "temperature": 0.05,
            "loss_fn": "InBatchCrossEnt",
        }
        run.config.update(torchemb_config)
        model = TorchEmbCrossEnt(run, **torchemb_config)
        run.summary["device"] = str(model.device)
        return model
    elif model_name == "popular":
        return Popular()
    else:
        raise NotImplementedError(f"Model '{model_name}' is not implemented")


def fit_model(
    model: BaseModel, model_name: str, df_train: pl.DataFrame, df_events: pl.DataFrame
) -> None:
    """Fit a model based on the model name"""
    if model_name == "popular":
        model.fit(df_train)
        return
    elif model_name in ["tfidf"]:
        model.fit(
            df_train["cookie"], df_train["node"], df_train["event"], df_train["week"]
        )
        return
    else:
        model.fit(df_train, df_events)


def load_model(
    model_type: str, model_path: str, df_events: pl.DataFrame, df_cat: pl.DataFrame
) -> BaseModel:
    """Load a pretrained model"""
    if model_type == "als":
        return ALSRecommender.load(model_path)
    elif model_type == "tfidf":
        model = TfidfRecommender(df_events)
        # Assuming TfidfRecommender has a load method
        return model.load(model_path)
    elif model_type == "lightfm":
        model = LightFMRecommender(df_events)
        # Assuming LightFMRecommender has a load method
        return model.load(model_path)
    elif model_type == "torch-emb":
        # Load TorchEmbModel with torch.load
        return TorchEmbModel.load(model_path)
    elif model_type == "popular":
        model = Popular()
        # Assuming Popular has a load method
        return model.load(model_path)
    else:
        raise NotImplementedError(f"Model '{model_type}' is not implemented")


if __name__ == "__main__":
    main()
