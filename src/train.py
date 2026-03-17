"""Training loop for the March Madness GNN."""

import torch
import torch.nn as nn

from .data_loader import load_csvs, build_graph, build_training_pairs
from .model import MarchMadnessGNN


def train(epochs: int = 100, lr: float = 0.01, val_split: float = 0.2, hidden_dim: int = 64, save_path: str = "model.pt"):
    """Train the GNN on season game results."""
    teams, games, _ = load_csvs()
    graph = build_graph(teams, games)
    team_a_ids, team_b_ids, labels = build_training_pairs(games)

    # Train/val split (random shuffle)
    n = len(labels)
    perm = torch.randperm(n)
    split = int(n * (1 - val_split))
    train_idx, val_idx = perm[:split], perm[split:]

    model = MarchMadnessGNN(
        team_features=graph["team"].x.shape[1],
        conf_features=graph["conference"].x.shape[1],
        hidden_dim=hidden_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        optimizer.zero_grad()
        preds = model(graph, team_a_ids[train_idx], team_b_ids[train_idx])
        loss = criterion(preds, labels[train_idx])
        loss.backward()
        optimizer.step()

        train_acc = ((preds > 0.5).float() == labels[train_idx]).float().mean().item()

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            val_preds = model(graph, team_a_ids[val_idx], team_b_ids[val_idx])
            val_loss = criterion(val_preds, labels[val_idx]).item()
            val_acc = ((val_preds > 0.5).float() == labels[val_idx]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {loss.item():.4f} Acc: {train_acc:.3f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}"
            )

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to {save_path}")
    return model
