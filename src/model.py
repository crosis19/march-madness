"""Heterogeneous GNN model for NCAA basketball matchup prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class MarchMadnessGNN(nn.Module):
    """Two-layer heterogeneous GNN with a matchup prediction head.

    The GNN produces team embeddings by message-passing over:
        - team <-> conference edges
        - team -defeated-> team edges

    The matchup predictor concatenates two team embeddings and
    outputs P(team_a wins).
    """

    def __init__(self, team_features: int = 8, conf_features: int = 3, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()

        # Layer 1: project each node type into hidden_dim, then message-pass
        self.conv1 = HeteroConv({
            ("team", "belongs_to", "conference"): SAGEConv((team_features, conf_features), hidden_dim),
            ("conference", "has_team", "team"): SAGEConv((conf_features, team_features), hidden_dim),
            ("team", "defeated", "team"): SAGEConv(team_features, hidden_dim),
        }, aggr="sum")

        # Layer 2: hidden_dim -> hidden_dim
        self.conv2 = HeteroConv({
            ("team", "belongs_to", "conference"): SAGEConv(hidden_dim, hidden_dim),
            ("conference", "has_team", "team"): SAGEConv(hidden_dim, hidden_dim),
            ("team", "defeated", "team"): SAGEConv(hidden_dim, hidden_dim),
        }, aggr="sum")

        self.dropout = dropout

        # Matchup prediction MLP: concat two team embeddings -> P(team_a wins)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def get_team_embeddings(self, data):
        """Run GNN layers and return team node embeddings."""
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        # Layer 1
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(F.dropout(x, p=self.dropout, training=self.training)) for key, x in x_dict.items()}

        # Layer 2
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(F.dropout(x, p=self.dropout, training=self.training)) for key, x in x_dict.items()}

        return x_dict["team"]

    def predict_matchup(self, team_embeddings, team_a_ids, team_b_ids):
        """Predict P(team_a wins) for each (team_a, team_b) pair."""
        emb_a = team_embeddings[team_a_ids]
        emb_b = team_embeddings[team_b_ids]
        combined = torch.cat([emb_a, emb_b], dim=-1)
        return torch.sigmoid(self.predictor(combined)).squeeze(-1)

    def forward(self, data, team_a_ids, team_b_ids):
        """Full forward pass: graph -> team embeddings -> matchup predictions."""
        team_embeddings = self.get_team_embeddings(data)
        return self.predict_matchup(team_embeddings, team_a_ids, team_b_ids)
