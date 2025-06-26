import torch
from torch import nn
import wandb
import wandb.wandb_run

import polars as pl
import faiss
import torch.nn.functional as F

from .base import BaseTorchModel


class TorchEmbModel(BaseTorchModel):
    def __init__(
        self,
        run: wandb.wandb_run.Run,
        embedding_dim: int = 64,
        epochs: int = 5,
        batch_size: int = 1024,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.run = run
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.user_embeddings: nn.Embedding | None = None
        self.item_embeddings: nn.Embedding | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, df_train: pl.DataFrame, df_events: pl.DataFrame) -> None:
        users = df_train["cookie"].unique().to_list()
        items = df_train["node"].unique().to_list()
        self.user_id_to_index = {u: i for i, u in enumerate(users)}
        self.item_id_to_index = {j: i for i, j in enumerate(items)}
        self.index_to_item_id = {i: j for j, i in self.item_id_to_index.items()}

        num_users = len(self.user_id_to_index)
        num_items = len(self.item_id_to_index)

        self.user_embeddings = nn.Embedding(num_users, self.embedding_dim).to(self.device)
        self.item_embeddings = nn.Embedding(num_items, self.embedding_dim).to(self.device)

        optimizer = torch.optim.Adam(
            list(self.user_embeddings.parameters()) + list(self.item_embeddings.parameters()),
            lr=self.lr,
        )
        loss_fn = nn.CosineEmbeddingLoss()

        user_indices = torch.tensor(
            [self.user_id_to_index[u] for u in df_train["cookie"].to_list()],
            dtype=torch.long,
        )
        item_indices = torch.tensor(
            [self.item_id_to_index[i] for i in df_train["node"].to_list()],
            dtype=torch.long,
        )

        dataset = torch.utils.data.TensorDataset(user_indices, item_indices)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_users, batch_items in dataloader:
                batch_users = batch_users.to(self.device)
                batch_items = batch_items.to(self.device)

                user_emb = self.user_embeddings(batch_users)
                pos_item_emb = self.item_embeddings(batch_items)

                # in-batch negatives: reuse pos_item_emb as negatives
                labels = torch.cat([
                    torch.ones(len(batch_users), device=self.device),
                    -torch.ones(len(batch_users) * len(batch_items), device=self.device),
                ])

                pos_scores = F.cosine_similarity(user_emb, pos_item_emb)
                expanded_user = user_emb.unsqueeze(1).expand(-1, len(batch_items), -1).reshape(-1, self.embedding_dim)
                expanded_neg = pos_item_emb.unsqueeze(0).expand(len(batch_users), -1, -1).reshape(-1, self.embedding_dim)
                neg_scores = F.cosine_similarity(expanded_user, expanded_neg)

                loss = loss_fn(
                    torch.cat([user_emb, expanded_user], dim=0),
                    torch.cat([pos_item_emb, expanded_neg], dim=0),
                    labels,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if self.run:
                self.run.log({"epoch": epoch, "loss": total_loss})

        self.user_embeddings = self.user_embeddings.cpu()
        self.item_embeddings = self.item_embeddings.cpu()

    def predict(self, user_to_pred: list[int | str], N: int = 40) -> pl.DataFrame:
        item_matrix = self.item_embeddings.weight.detach().cpu().numpy()
        faiss.normalize_L2(item_matrix)
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(item_matrix)

        results = []
        for u in user_to_pred:
            if u not in self.user_id_to_index:
                continue
            u_idx = self.user_id_to_index[u]
            u_emb = self.user_embeddings.weight[u_idx].detach().cpu().numpy().reshape(1, -1)
            faiss.normalize_L2(u_emb)
            _, I = index.search(u_emb, N)
            top_items = [self.index_to_item_id[i] for i in I[0]]
            results.append(pl.DataFrame({"cookie": [u] * len(top_items), "node": top_items}))

        return pl.concat(results) if results else pl.DataFrame({"cookie": [], "node": []})
