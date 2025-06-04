import polars as pl
import pytest

from ranker.features import get_node_features, get_user_category_features, enrich

def test_get_node_features():
    # sample clickstream: item 1 has 2 views, 1 contact; item 2 has 3 views, 1 contact; item 3 has 1 view, 0 contacts
    df_click = pl.DataFrame({
        "cookie": [0, 0, 1, 1, 1, 2],
        "item": [1, 1, 2, 2, 2, 3],
        "node": [1, 1, 2, 2, 2, 3],
        "event": [0, 0, 0, 0, 0, 0],
        "is_contact": [0, 1, 0, 0, 1, 0]
    })
    # sample categories: items 1 and 2 in category 10; item 3 in category 20
    df_cat = pl.DataFrame({
        "item": [1, 2, 3],
        "location": [0, 0, 0],
        "category": [10, 10, 20],
        "clean_params": ["", "", ""],
        "node": [1, 2, 3]
    })

    features = get_node_features(df_click, df_cat).sort("node")
    # expected per-item contact rates: item1:1/1=1.0, item2:1/2=0.5, item3:0/1=0.0
    expected_rates = [1.0, 0.5, 0.0]
    # expected category shares: category10 has 2 contacts of total 2 =>1.0, category20 has 0 =>0.0
    expected_shares = [1.0, 1.0, 0.0]

    assert features["node"].to_list() == [1, 2, 3]
    assert features["category"].to_list() ==  [10, 10, 20]
    assert features["node_view_contact_rate"].to_list() == pytest.approx(expected_rates)
    assert features["category_contact_share"].to_list() == pytest.approx(expected_shares)


def test_get_user_category_features():
    # sample clickstream: user 1 has 3 events in category 10 with 2 contacts; user 2 has 1 event in category 10 contact and 1 in category 20 no contact
    df_click = pl.DataFrame({
        "cookie": [1, 1, 1, 2, 2],
        "item": [1, 2, 1, 2, 3],
        "event": [0, 0, 0, 0, 0],
        "is_contact": [1, 0, 1, 1, 0]
    })
    df_cat = pl.DataFrame({
        "item": [1, 2, 3],
        "location": [0, 0, 0],
        "category": [10, 10, 20],
        "clean_params": ["", "", ""],
        "node": [0, 0, 0]
    })

    features = get_user_category_features(df_click, df_cat).sort(["user_id", "category_id"])
    # expected shares: user1-cat10:2/3, user2-cat10:1/1, user2-cat20:0/1
    expected_user = [1, 2, 2]
    expected_cat = [10, 10, 20]
    expected_share = [2/3, 1.0, 0.0]

    assert features["user_id"].to_list() == expected_user
    assert features["category_id"].to_list() == expected_cat
    assert features["user_category_contact_share"].to_list() == pytest.approx(expected_share)


def test_enrich():
    # prepare sample predictions, clickstream, and category data
    df_pred = pl.DataFrame({
        "cookie": [1, 1, 2, 2],
        "node":   [10, 20, 10, 20],
        "scores": [0.9, 0.8, 0.7, 0.6]
    })
    df_click = pl.DataFrame({
        "cookie":     [1, 1, 2, 2],
        "node":       [10, 20, 10, 20],
        "event":      [0, 0, 0, 0],
        "is_contact": [1, 0, 0, 1]
    })
    df_cat = pl.DataFrame({
        "node":     [10, 20],
        "category": [100, 200]
    })

    features = enrich(df_pred, df_click, df_cat).sort(["user_id", "node"])

    # check renaming and target assignment
    assert features["score_als"].to_list() == [0.9, 0.8, 0.7, 0.6]
    assert features["target"].to_list() == [10, 1, 1, 10]

    # node-level contact rate for both nodes is 1/1 => 1.0
    assert features["node_view_contact_rate"].unique().to_list() == [1.0]

    # each category has 1 contact out of 2 total => 0.5
    assert features["category_contact_share"].unique().to_list() == [0.5]

    # user-category shares: user1-cat100=1/1=1.0?, user1-cat200=0/1=0.0, user2-cat100=0/1=0.0, user2-cat200=1/1=1.0
    exp = {
        (1, 100): 1.0,
        (1, 200): 0.0,
        (2, 100): 0.0,
        (2, 200): 1.0
    }
    got = {
        (r["user_id"], r["category"]): r["user_category_contact_share"]
        for r in features.select(["user_id", "category", "user_category_contact_share"]).iter_rows(named=True)
    }
    assert got == exp
