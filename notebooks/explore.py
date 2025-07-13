# %%
print(123)

# %%
import sys
sys.path.append("../")
from utils import get_data
import polars as pl

df_test_users, df_clickstream, df_cat, df_text, df_events, df_train, df_eval = get_data("../data")

# %%

print(df_train.head())

# extract seen_nodes dict: cookie → list of unique nodes
seen_nodes_df = df_train.group_by("cookie").agg(
    pl.col("node").unique().alias("seen_nodes")
)
seen_nodes: dict[int, list[int]] = {
    row["cookie"]: row["seen_nodes"] for row in seen_nodes_df.to_dicts()
}
print(f"Extracted seen_nodes for {len(seen_nodes)} users")
print("Example entry:", next(iter(seen_nodes.items())))


# %%

populars = df_train.group_by("node").agg(pl.col("cookie").len().alias("len")).sort("len", descending=True).get_column("len").to_list()[:1000]

# %%

from models.implicit_model import ALSRecommender
from models.base import FaissPredict
import inspect

# %%

class Bar(FaissPredict, ALSRecommender):
    def fit(self, df_train: pl.DataFrame, df_events: pl.DataFrame) -> None:
        super().fit(df_train, df_events)
        self.set_seen_items(self.get_seen_nodes(df_train))

# %%

bar = Bar()

# %%

bar.model

# %%

print(inspect.getsource(bar.predict))