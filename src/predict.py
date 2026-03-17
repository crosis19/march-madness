"""Predict March Madness bracket outcomes using the trained GNN."""

import torch
import pandas as pd

from .data_loader import load_csvs, build_graph
from .model import MarchMadnessGNN


def predict_bracket(model_path: str = "model.pt"):
    """Load a trained model and simulate the tournament bracket."""
    teams, games, tournament = load_csvs()
    graph = build_graph(teams, games)

    # Build team lookup
    team_names = {int(row["team_id"]): row["name"] for _, row in teams.iterrows()}
    team_seeds = {int(row["team_id"]): int(row["seed"]) for _, row in teams.iterrows()}

    # Load model
    model = MarchMadnessGNN(
        team_features=graph["team"].x.shape[1],
        conf_features=graph["conference"].x.shape[1],
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Get team embeddings
    with torch.no_grad():
        team_embeddings = model.get_team_embeddings(graph)

    # Simulate bracket round by round
    rounds = sorted(tournament["round"].unique())
    current_matchups = tournament.copy()

    for round_num in rounds:
        round_games = current_matchups[current_matchups["round"] == round_num]
        round_name = _round_name(round_num)
        print(f"\n{'='*60}")
        print(f"  {round_name}")
        print(f"{'='*60}")

        winners = []
        for _, game in round_games.iterrows():
            a_id, b_id = int(game["team_a_id"]), int(game["team_b_id"])

            with torch.no_grad():
                prob = model.predict_matchup(
                    team_embeddings,
                    torch.tensor([a_id]),
                    torch.tensor([b_id]),
                ).item()

            a_name = team_names.get(a_id, f"Team {a_id}")
            b_name = team_names.get(b_id, f"Team {b_id}")
            a_seed = team_seeds.get(a_id, "?")
            b_seed = team_seeds.get(b_id, "?")

            winner_id = a_id if prob > 0.5 else b_id
            winner_name = a_name if prob > 0.5 else b_name
            confidence = prob if prob > 0.5 else 1 - prob

            print(f"  ({a_seed:>2}) {a_name:<22} vs ({b_seed:>2}) {b_name:<22} -> {winner_name} ({confidence:.1%})")
            winners.append(winner_id)

        # Build next round matchups (pair consecutive winners)
        if len(winners) >= 2:
            next_round = round_num + 1
            next_matchups = []
            for i in range(0, len(winners), 2):
                if i + 1 < len(winners):
                    next_matchups.append({
                        "round": next_round,
                        "team_a_id": winners[i],
                        "team_b_id": winners[i + 1],
                    })
            if next_matchups:
                next_df = pd.DataFrame(next_matchups)
                current_matchups = pd.concat([current_matchups, next_df], ignore_index=True)

    if winners:
        champion_name = team_names.get(winners[-1], f"Team {winners[-1]}")
        champion_seed = team_seeds.get(winners[-1], "?")
        print(f"\n{'='*60}")
        print(f"  PREDICTED CHAMPION: ({champion_seed}) {champion_name}")
        print(f"{'='*60}")


def _round_name(round_num: int) -> str:
    names = {
        1: "FIRST ROUND (Round of 64)",
        2: "SECOND ROUND (Round of 32)",
        3: "SWEET 16",
        4: "ELITE 8",
        5: "FINAL FOUR",
        6: "CHAMPIONSHIP",
    }
    return names.get(round_num, f"Round {round_num}")
