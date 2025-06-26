import torch

# 1) Create a batch of size B=2, emb dim D=4
user_emb = torch.tensor([[1., 2., 3., 4.],
                         [11., 22., 33., 44.]])   # shape (2,4)
pos_item_emb = user_emb.clone()           # same shape (2,4)

B, D = user_emb.shape

# 2) Build expanded_user (B×B, D):
#    each user repeated for each item in batch
expanded_user = (
    user_emb
    .unsqueeze(1)                # (2,1,4)
    .expand(-1, B, -1)           # (2,2,4)
    .reshape(-1, D)              # (4,4)
)

# 3) Build expanded_neg (B×B, D):
#    each item repeated for each user in batch
expanded_neg = (
    pos_item_emb
    .unsqueeze(0)                # (1,2,4)
    .expand(B, -1, -1)           # (2,2,4)
    .reshape(-1, D)              # (4,4)
)

print("user_emb:\n", user_emb, "\n")
print("pos_item_emb:\n", pos_item_emb, "\n")
print(f"expanded_user (shape={expanded_user.shape}):\n", expanded_user, "\n")
print(f"expanded_neg   (shape={expanded_neg.shape}):\n", expanded_neg)
