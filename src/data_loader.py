"""Load NCAA basketball data from CSVs and build a PyG HeteroData graph."""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_csvs(data_dir=DATA_DIR):
    """Load teams, games, and tournament CSVs."""
    teams = pd.read_csv(os.path.join(data_dir, "teams.csv"))
    games = pd.read_csv(os.path.join(data_dir, "games.csv"))
    tournament = pd.read_csv(os.path.join(data_dir, "tournament.csv"))
    return teams, games, tournament


def build_graph(teams: pd.DataFrame, games: pd.DataFrame) -> HeteroData:
    """Build a heterogeneous graph from team and game data.

    Node types:
        - team: features = [wins, losses, ppg, opp_ppg, off_eff, def_eff, sos, seed]
        - conference: features = [avg_wins, num_tournament_teams, avg_sos]

    Edge types:
        - (team, belongs_to, conference)
        - (conference, has_team, team)
        - (team, defeated, team): directed, with weight and avg_margin attributes
        - (team, played, team): bidirectional, for all teams that faced each other
    """
    data = HeteroData()

    # --- Team node features ---
    feature_cols = ["wins", "losses", "ppg", "opp_ppg", "off_efficiency", "def_efficiency", "sos", "net_rating"]
    team_features = teams[feature_cols].values.astype(np.float32)

    # Normalize each feature to [0, 1]
    mins = team_features.min(axis=0)
    maxs = team_features.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # avoid division by zero
    team_features = (team_features - mins) / ranges

    data["team"].x = torch.tensor(team_features, dtype=torch.float)
    data["team"].num_nodes = len(teams)

    # --- Conference node features ---
    conferences = sorted(teams["conference"].unique())
    conf_to_idx = {c: i for i, c in enumerate(conferences)}

    conf_features = []
    for conf in conferences:
        conf_teams = teams[teams["conference"] == conf]
        avg_wins = conf_teams["wins"].mean()
        num_tourney = (conf_teams["seed"] > 0).sum() if "seed" in conf_teams.columns else 0
        avg_sos = conf_teams["sos"].mean()
        conf_features.append([avg_wins, num_tourney, avg_sos])

    conf_features = np.array(conf_features, dtype=np.float32)
    cmins = conf_features.min(axis=0)
    cmaxs = conf_features.max(axis=0)
    cranges = cmaxs - cmins
    cranges[cranges == 0] = 1
    conf_features = (conf_features - cmins) / cranges

    data["conference"].x = torch.tensor(conf_features, dtype=torch.float)
    data["conference"].num_nodes = len(conferences)

    # --- Team <-> Conference edges ---
    team_to_conf_src = []
    team_to_conf_dst = []
    for _, row in teams.iterrows():
        team_to_conf_src.append(int(row["team_id"]))
        team_to_conf_dst.append(conf_to_idx[row["conference"]])

    data["team", "belongs_to", "conference"].edge_index = torch.tensor(
        [team_to_conf_src, team_to_conf_dst], dtype=torch.long
    )
    data["conference", "has_team", "team"].edge_index = torch.tensor(
        [team_to_conf_dst, team_to_conf_src], dtype=torch.long
    )

    # --- Aggregate game stats per team pair ---
    # Track both defeated stats and played stats
    defeat_stats = defaultdict(lambda: {"count": 0, "total_margin": 0.0})
    played_stats = defaultdict(lambda: {"num_games": 0, "total_diff": 0.0})

    for _, row in games.iterrows():
        a, b = int(row["team_a_id"]), int(row["team_b_id"])
        sa, sb = row["score_a"], row["score_b"]

        # Played edges: bidirectional, track from both perspectives
        pair = (min(a, b), max(a, b))
        played_stats[pair]["num_games"] += 1

        if sa > sb:
            defeat_stats[(a, b)]["count"] += 1
            defeat_stats[(a, b)]["total_margin"] += sa - sb
            played_stats[pair]["total_diff"] += (sa - sb) if a == pair[0] else -(sa - sb)
        elif sb > sa:
            defeat_stats[(b, a)]["count"] += 1
            defeat_stats[(b, a)]["total_margin"] += sb - sa
            played_stats[pair]["total_diff"] += (sa - sb) if a == pair[0] else -(sa - sb)

    # --- Defeated edges (directed) ---
    if defeat_stats:
        src, dst, weights, margins = [], [], [], []
        for (w, l), stats in defeat_stats.items():
            src.append(w)
            dst.append(l)
            weights.append(stats["count"])
            margins.append(stats["total_margin"] / stats["count"])

        data["team", "defeated", "team"].edge_index = torch.tensor(
            [src, dst], dtype=torch.long
        )
        data["team", "defeated", "team"].edge_attr = torch.tensor(
            list(zip(weights, margins)), dtype=torch.float
        )
    else:
        data["team", "defeated", "team"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["team", "defeated", "team"].edge_attr = torch.zeros((0, 2), dtype=torch.float)

    # --- Played edges (bidirectional) ---
    # Every pair of teams that played each other gets edges in both directions.
    # This ensures signal can flow even from losers to winners.
    if played_stats:
        played_src, played_dst, played_attr = [], [], []
        for (a, b), stats in played_stats.items():
            num_games = stats["num_games"]
            avg_diff = stats["total_diff"] / num_games  # positive if a scored more overall

            # Edge a -> b: features = [num_games, avg_score_diff from a's perspective]
            played_src.append(a)
            played_dst.append(b)
            played_attr.append([num_games, avg_diff])

            # Edge b -> a: features = [num_games, avg_score_diff from b's perspective]
            played_src.append(b)
            played_dst.append(a)
            played_attr.append([num_games, -avg_diff])

        data["team", "played", "team"].edge_index = torch.tensor(
            [played_src, played_dst], dtype=torch.long
        )
        data["team", "played", "team"].edge_attr = torch.tensor(
            played_attr, dtype=torch.float
        )
    else:
        data["team", "played", "team"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["team", "played", "team"].edge_attr = torch.zeros((0, 2), dtype=torch.float)

    return data


def build_training_pairs(games: pd.DataFrame):
    """Build (team_a, team_b, label) pairs from game results.

    For each game where A beat B: (A, B) -> label 1, (B, A) -> label 0.
    Returns tensors for team_a_ids, team_b_ids, labels.
    """
    a_ids, b_ids, labels = [], [], []
    for _, row in games.iterrows():
        a, b = int(row["team_a_id"]), int(row["team_b_id"])
        sa, sb = row["score_a"], row["score_b"]
        if sa > sb:
            a_ids.append(a)
            b_ids.append(b)
            labels.append(1.0)
            a_ids.append(b)
            b_ids.append(a)
            labels.append(0.0)
        elif sb > sa:
            a_ids.append(b)
            b_ids.append(a)
            labels.append(1.0)
            a_ids.append(a)
            b_ids.append(b)
            labels.append(0.0)

    return (
        torch.tensor(a_ids, dtype=torch.long),
        torch.tensor(b_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float),
    )
