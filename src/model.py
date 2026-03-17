"""Heterogeneous GNN model for NCAA basketball matchup prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv


class MarchMadnessGNN(nn.Module):
    """Two-layer heterogeneous GNN with GAT layers and a matchup prediction head.

    Uses Graph Attention Networks so the model can learn that beating a strong
    team matters more than beating a weak team. Incorporates edge features
    (win count, margin) into the attention computation.

    Edge types:
        - team -defeated-> team: directed, with edge features [weight, avg_margin]
        - team -played-> team: bidirectional, with edge features [num_games, avg_score_diff]
        - team <-> conference: membership edges
    """

    def __init__(self, team_features: int = 8, conf_features: int = 3,
                 hidden_dim: int = 64, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        head_dim = hidden_dim // heads

        # Project each node type to hidden_dim before message passing
        self.team_proj = nn.Linear(team_features, hidden_dim)
        self.conf_proj = nn.Linear(conf_features, hidden_dim)

        # Layer 1: GAT with edge features for team-team edges, SAGE for conference edges
        self.conv1 = HeteroConv({
            ("team", "belongs_to", "conference"): SAGEConv(hidden_dim, hidden_dim),
            ("conference", "has_team", "team"): SAGEConv(hidden_dim, hidden_dim),
            ("team", "defeated", "team"): GATConv(
                hidden_dim, head_dim, heads=heads, edge_dim=2, add_self_loops=False,
            ),
            ("team", "played", "team"): GATConv(
                hidden_dim, head_dim, heads=heads, edge_dim=2, add_self_loops=False,
            ),
        }, aggr="sum")

        # Layer 2
        self.conv2 = HeteroConv({
            ("team", "belongs_to", "conference"): SAGEConv(hidden_dim, hidden_dim),
            ("conference", "has_team", "team"): SAGEConv(hidden_dim, hidden_dim),
            ("team", "defeated", "team"): GATConv(
                hidden_dim, head_dim, heads=heads, edge_dim=2, add_self_loops=False,
            ),
            ("team", "played", "team"): GATConv(
                hidden_dim, head_dim, heads=heads, edge_dim=2, add_self_loops=False,
            ),
        }, aggr="sum")

        self.dropout = dropout

        # Matchup prediction MLP using richer pairwise features:
        # concat(emb_a, emb_b, emb_a - emb_b, emb_a * emb_b) = 4 * hidden_dim
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def get_team_embeddings(self, data):
        """Run GNN layers and return team node embeddings."""
        # Project node features to hidden_dim
        x_dict = {
            "team": self.team_proj(data["team"].x),
            "conference": self.conf_proj(data["conference"].x),
        }

        edge_index_dict = data.edge_index_dict

        # Build edge_attr dict for conv layers that use it
        edge_attr_dict = {}
        for edge_type in [("team", "defeated", "team"), ("team", "played", "team")]:
            if edge_type in data.edge_attr_dict:
                edge_attr_dict[edge_type] = data.edge_attr_dict[edge_type]

        # Layer 1
        x_dict = self.conv1(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
        x_dict = {key: F.relu(F.dropout(x, p=self.dropout, training=self.training))
                  for key, x in x_dict.items()}

        # Layer 2
        x_dict = self.conv2(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
        x_dict = {key: F.relu(F.dropout(x, p=self.dropout, training=self.training))
                  for key, x in x_dict.items()}

        return x_dict["team"]

    def predict_matchup(self, team_embeddings, team_a_ids, team_b_ids):
        """Predict P(team_a wins) for each (team_a, team_b) pair.

        Uses four pairwise representations:
        - concat: raw asymmetric information
        - difference: who is stronger
        - product: interaction features
        """
        emb_a = team_embeddings[team_a_ids]
        emb_b = team_embeddings[team_b_ids]
        combined = torch.cat([
            emb_a,
            emb_b,
            emb_a - emb_b,
            emb_a * emb_b,
        ], dim=-1)
        return torch.sigmoid(self.predictor(combined)).squeeze(-1)

    def forward(self, data, team_a_ids, team_b_ids):
        """Full forward pass: graph -> team embeddings -> matchup predictions."""
        team_embeddings = self.get_team_embeddings(data)
        return self.predict_matchup(team_embeddings, team_a_ids, team_b_ids)
