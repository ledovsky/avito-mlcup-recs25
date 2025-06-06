import polars as pl


def get_node_features(
    df_clickstream: pl.DataFrame, df_cat: pl.DataFrame
) -> pl.DataFrame:
    # compute per-node conversion rate (contacts / non-contacts)
    item_stats = (
        df_clickstream.group_by("node")
        .agg(
            total_views=pl.len(),
            contact_count=pl.col("is_contact").sum(),
        )
        .with_columns(
            [
                (pl.col("total_views") - pl.col("contact_count")).alias(
                    "non_contact_count"
                ),
                pl.when(pl.col("total_views") - pl.col("contact_count") > 0)
                .then(
                    pl.col("contact_count")
                    / (pl.col("total_views") - pl.col("contact_count"))
                )
                .alias("node_view_contact_rate"),
            ]
        )
        .select(["node", "node_view_contact_rate"])
    )
    # compute share of contact events per category
    df_with_cat = df_clickstream.join(
        df_cat.select(["node", "category"]).unique(subset=["node"]),
        on="node",
        how="inner",
    )
    category_contacts = df_with_cat.group_by("category").agg(
        category_contact_count=pl.col("is_contact").sum()
    )
    total_contacts = category_contacts["category_contact_count"].sum()
    category_share = category_contacts.with_columns(
        (pl.col("category_contact_count") / total_contacts).alias(
            "category_contact_share"
        )
    ).select(["category", "category_contact_share"])
    # combine item stats with category share
    features = (
        df_cat.select(["node", "category"])
        .unique(subset=["node"])
        .join(item_stats, on="node", how="left")
        .join(category_share, on="category", how="left")
        .select(
            ["node", "category", "node_view_contact_rate", "category_contact_share"]
        )
    )
    return features


def get_user_category_features(
    df_clickstream: pl.DataFrame, df_cat: pl.DataFrame
) -> pl.DataFrame:
    # join clickstream with category info
    df_with_cat = df_clickstream.join(
        df_cat.select(["item", "category"]), on="item", how="inner"
    )
    # compute per-user-category share of contacts
    user_cat_stats = (
        df_with_cat.group_by("cookie", "category")
        .agg(total_events=pl.len(), contact_count=pl.col("is_contact").sum())
        .with_columns(
            (pl.col("contact_count") / pl.col("total_events")).alias(
                "user_category_contact_share"
            )
        )
    )
    # rename and select final columns
    return user_cat_stats.rename(
        {"cookie": "user_id", "category": "category_id"}
    ).select(["user_id", "category_id", "user_category_contact_share"])


def add_target(
    df: pl.DataFrame, df_clickstream: pl.DataFrame
) -> pl.DataFrame:
    """
    Add target column to DataFrame based on clickstream:
    10 if any contact, 1 if any non-contact, else 0.
    """
    df_click_agg = (
        df_clickstream.group_by(["cookie", "node"])
        .agg(contact_sum=pl.col("is_contact").sum(), total_events=pl.len())
        .with_columns(non_contact_sum=pl.col("total_events") - pl.col("contact_sum"))
        .with_columns(
            pl.when(pl.col("contact_sum") > 0)
            .then(10)
            .when(pl.col("non_contact_sum") > 0)
            .then(1)
            .otherwise(0)
            .alias("target")
        )
        .select(["cookie", "node", "target"])
    )
    return df.join(
        df_click_agg.rename({"cookie": "user_id"}), on=["user_id", "node"], how="left"
    ).fill_null(0)

def enrich(
    df_pred: pl.DataFrame,
    df_clickstream: pl.DataFrame,
    df_cat: pl.DataFrame,
) -> pl.DataFrame:
    # rename prediction columns
    df = df_pred.rename({"cookie": "user_id", "scores": "score_als"})
    # load features
    node_feats = get_node_features(df_clickstream, df_cat)
    user_cat_feats = get_user_category_features(df_clickstream, df_cat)
    # enrich with category
    df = df.join(
        df_cat.select(["node", "category"]).unique(subset=["node"]),
        on="node",
        how="left",
    )
    # enrich with node-level features
    df = df.join(node_feats, on=["node", "category"], how="left")
    # enrich with user-category features
    df = df.join(
        user_cat_feats.rename({"category_id": "category"}),
        on=["user_id", "category"],
        how="left",
    )
    # fill missing values with zero
    return df.fill_null(0)
