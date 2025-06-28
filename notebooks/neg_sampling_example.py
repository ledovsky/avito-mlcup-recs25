import torch

# 1) Create a batch of size B=2, emb dim D=4
user_emb = torch.tensor([[1., 2., 3., 4.],
                         [11., 22., 33., 44.],
                         [111., 222., 333., 444.]])
pos_item_emb = user_emb * 10

B, D = user_emb.shape

# shape (B, B, D)
expanded_user_mat = user_emb.unsqueeze(1).expand(-1, B, -1)
expanded_item_mat = pos_item_emb.unsqueeze(0).expand(B, -1, -1)
# mask out positive pairs
mask = ~torch.eye(B, dtype=torch.bool)
neg_user_all = expanded_user_mat[mask].reshape(B, B-1, D)
neg_item_all = expanded_item_mat[mask].reshape(B, B-1, D)

print("user_emb:\n", user_emb, "\n")
print("pos_item_emb:\n", pos_item_emb, "\n")
print(f"mask (shape={mask.shape}):\n", mask, "\n")
print(f"neg_user_all (shape={neg_user_all.shape}):\n", neg_user_all, "\n")
print(f"neg_item_all   (shape={neg_item_all.shape}):\n", neg_item_all)
