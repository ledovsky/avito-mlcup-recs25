import torch
from torch import nn
import wandb
import wandb.wandb_run
import numpy as np

import polars as pl
import faiss
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, record_function
from tqdm import tqdm

from .base import BaseTorchModel, FaissPredict


class TorchEmbModel(FaissPredict, BaseTorchModel):
    def __init__(
        self,
        run: wandb.wandb_run.Run,
        embedding_dim: int = 64,
        epochs: int = 5,
        batch_size: int = 1024,
        lr: float = 1e-3,
        alpha: float = 0.1,
        top_k_items: int | None = None,
        k_inbatch_negs: int = 10,
        debug: bool = False,
        dedupe: bool = False
    ):
        super().__init__()
        self.run = run
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.alpha = alpha
        self.top_k_items = top_k_items
        self.k_inbatch_negs = k_inbatch_negs
        self.debug = debug
        self.dedupe = debug
        self.user_embeddings: nn.Embedding | None = None
        self.item_embeddings: nn.Embedding | None = None
        self.user_embeddings_np: np.typing.NDArray | None = None
        self.item_embeddings_np: np.typing.NDArray | None = None
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"TorchEmbModel using device {self.device}")
        self.seen_items: dict[int, set[int]] = {}

        self.populars = []
        self.set_is_cos_dist(False)

    def fit(self, df_train: pl.DataFrame, df_events: pl.DataFrame) -> None:
        if self.top_k_items:
            df_train = self.filter_top_k_items(df_train, self.top_k_items)
        
        if self.dedupe:
            df_train = self.dedupe(df_train)

        users = df_train["cookie"].unique().to_list()
        items = df_train["node"].unique().to_list()
        self.user_id_to_index = {u: i for i, u in enumerate(users)}
        self.item_id_to_index = {j: i for i, j in enumerate(items)}
        self.index_to_item_id = {i: j for j, i in self.item_id_to_index.items()}

        num_users = len(self.user_id_to_index)
        num_items = len(self.item_id_to_index)

        self.user_embeddings = nn.Embedding(num_users, self.embedding_dim).to(
            self.device
        )
        self.item_embeddings = nn.Embedding(num_items, self.embedding_dim).to(
            self.device
        )
        if self.device != torch.device("mps"):
            nn.init.orthogonal_(self.user_embeddings.weight)
            nn.init.orthogonal_(self.item_embeddings.weight)

        optimizer = torch.optim.Adam(
            list(self.user_embeddings.parameters())
            + list(self.item_embeddings.parameters()),
            lr=self.lr,
        )
        # loss_fn = nn.CosineEmbeddingLoss(reduction="mean")
        loss_fn = nn.BCEWithLogitsLoss()
        self.run.config.update({"loss_fn": "BCEWithLogits"})

        user_indices = torch.tensor(
            [self.user_id_to_index[u] for u in df_train["cookie"].to_list()],
            dtype=torch.long,
        )
        item_indices = torch.tensor(
            [self.item_id_to_index[i] for i in df_train["node"].to_list()],
            dtype=torch.long,
        )

        # record seen items per user for filtering during prediction
        self.seen_items = {}
        for u_idx, i_idx in zip(user_indices.tolist(), item_indices.tolist()):
            self.seen_items.setdefault(u_idx, set()).add(i_idx)

        if self.debug:
            user_indices = user_indices[:10000]
            item_indices = item_indices[:10000]

        dataset = torch.utils.data.TensorDataset(user_indices, item_indices)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_users, batch_items in tqdm(dataloader, desc=f"Epoch {epoch}"):
                # with record_function("load_to_device"):
                batch_users = batch_users.to(self.device)
                batch_items = batch_items.to(self.device)

                user_emb = self.user_embeddings(batch_users)
                pos_item_emb = self.item_embeddings(batch_items)

                # compute positive loss via dot product
                pos_labels = torch.ones(len(batch_users), device=self.device)
                pos_scores = (user_emb * pos_item_emb).sum(dim=1)
                pos_loss = loss_fn(pos_scores, pos_labels)

                # in-batch negatives: sample k negatives per positive
                k = min(self.k_inbatch_negs, len(batch_users) // 2)
                B = len(batch_users)
                # shape (B, B, D)
                expanded_user_mat = user_emb.unsqueeze(1).expand(-1, B, -1)
                expanded_item_mat = pos_item_emb.unsqueeze(0).expand(B, -1, -1)

                # mask out positive pairs
                mask = ~torch.eye(B, dtype=torch.bool, device=self.device)
                neg_user_all = expanded_user_mat[mask].reshape(
                    B, B - 1, self.embedding_dim
                )
                neg_item_all = expanded_item_mat[mask].reshape(
                    B, B - 1, self.embedding_dim
                )
                # sample k negatives per positive
                idx = torch.randperm(B - 1, device=self.device)[:k]
                sampled_user = neg_user_all[:, idx, :].reshape(-1, self.embedding_dim)
                sampled_neg = neg_item_all[:, idx, :].reshape(-1, self.embedding_dim)

                neg_labels = torch.zeros(B * k, device=self.device)
                neg_scores = (sampled_user * sampled_neg).sum(dim=1)
                neg_loss = loss_fn(neg_scores, neg_labels)

                loss = pos_loss + self.alpha * neg_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if self.run:
                self.run.log({"epoch": epoch, "loss": total_loss})

        self.set_embeddings(
            self.user_embeddings.weight.detach().cpu().numpy(),
            self.item_embeddings.weight.detach().cpu().numpy(),
        )

