"""Streamlit web UI for the March Madness GNN predictor."""

import os
import streamlit as st
import torch
import pandas as pd

from src.data_loader import load_csvs, build_graph
from src.model import MarchMadnessGNN
from src.train import train


MODEL_PATH = "model.pt"


@st.cache_resource
def load_data():
    """Load and cache the CSV data."""
    return load_csvs()


@st.cache_resource
def load_model_and_graph():
    """Load the trained model and build the graph."""
    teams, games, _ = load_data()
    graph = build_graph(teams, games)

    model = MarchMadnessGNN(
        team_features=graph["team"].x.shape[1],
        conf_features=graph["conference"].x.shape[1],
    )

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    return model, graph


def predict_matchup(model, graph, team_a_id, team_b_id):
    """Predict P(team_a wins) for a single matchup."""
    with torch.no_grad():
        team_embeddings = model.get_team_embeddings(graph)
        prob = model.predict_matchup(
            team_embeddings,
            torch.tensor([team_a_id]),
            torch.tensor([team_b_id]),
        ).item()
    return prob


def simulate_bracket(model, graph, tournament, teams_df):
    """Simulate the full tournament bracket. Returns list of round results."""
    team_names = {int(r["team_id"]): r["name"] for _, r in teams_df.iterrows()}
    team_seeds = {int(r["team_id"]): int(r["seed"]) for _, r in teams_df.iterrows()}
    team_confs = {int(r["team_id"]): r["conference"] for _, r in teams_df.iterrows()}

    with torch.no_grad():
        team_embeddings = model.get_team_embeddings(graph)

    round_names = {
        1: "First Round", 2: "Round of 32", 3: "Sweet 16",
        4: "Elite 8", 5: "Final Four", 6: "Championship",
    }

    current_matchups = tournament.copy()
    all_results = []

    rounds = sorted(current_matchups["round"].unique())
    for round_num in rounds:
        round_games = current_matchups[current_matchups["round"] == round_num]
        round_results = []
        winners = []

        for _, game in round_games.iterrows():
            a_id, b_id = int(game["team_a_id"]), int(game["team_b_id"])
            with torch.no_grad():
                prob = model.predict_matchup(
                    team_embeddings, torch.tensor([a_id]), torch.tensor([b_id])
                ).item()

            winner_id = a_id if prob > 0.5 else b_id
            confidence = prob if prob > 0.5 else 1 - prob

            round_results.append({
                "team_a": team_names.get(a_id, f"Team {a_id}"),
                "team_b": team_names.get(b_id, f"Team {b_id}"),
                "seed_a": team_seeds.get(a_id, "?"),
                "seed_b": team_seeds.get(b_id, "?"),
                "conf_a": team_confs.get(a_id, "?"),
                "conf_b": team_confs.get(b_id, "?"),
                "prob_a": prob,
                "winner": team_names.get(winner_id, f"Team {winner_id}"),
                "winner_id": winner_id,
                "winner_seed": team_seeds.get(winner_id, "?"),
                "confidence": confidence,
            })
            winners.append(winner_id)

        all_results.append({
            "round_num": round_num,
            "round_name": round_names.get(round_num, f"Round {round_num}"),
            "games": round_results,
        })

        # Build next round
        if len(winners) >= 2:
            next_matchups = []
            for i in range(0, len(winners), 2):
                if i + 1 < len(winners):
                    next_matchups.append({
                        "round": round_num + 1,
                        "team_a_id": winners[i],
                        "team_b_id": winners[i + 1],
                    })
            if next_matchups:
                current_matchups = pd.concat(
                    [current_matchups, pd.DataFrame(next_matchups)], ignore_index=True
                )

    return all_results


def main():
    st.set_page_config(page_title="March Madness GNN", page_icon="🏀", layout="wide")
    st.title("March Madness GNN Predictor")
    st.caption("Heterogeneous Graph Neural Network for NCAA Tournament Prediction")

    # Sidebar
    st.sidebar.header("Controls")
    page = st.sidebar.radio("Page", ["Bracket Prediction", "Head-to-Head", "Train Model"])

    # Check if model exists
    model_exists = os.path.exists(MODEL_PATH)

    if page == "Train Model":
        st.header("Train the GNN Model")
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.slider("Epochs", 10, 300, 100)
            lr = st.number_input("Learning Rate", 0.001, 0.1, 0.01, format="%.3f")
        with col2:
            hidden_dim = st.select_slider("Hidden Dimension", [32, 64, 128], value=64)

        if st.button("Train Model", type="primary"):
            progress = st.empty()
            with st.spinner("Training..."):
                model = train(epochs=epochs, lr=lr, hidden_dim=hidden_dim, save_path=MODEL_PATH)
            st.success(f"Training complete! Model saved to {MODEL_PATH}")
            st.cache_resource.clear()

    elif page == "Head-to-Head":
        st.header("Head-to-Head Prediction")

        if not model_exists:
            st.warning("No trained model found. Please train the model first.")
            return

        model, graph = load_model_and_graph()
        teams_df, _, _ = load_data()
        team_list = teams_df.sort_values("name")["name"].tolist()
        team_id_map = {r["name"]: int(r["team_id"]) for _, r in teams_df.iterrows()}
        team_seed_map = {r["name"]: int(r["seed"]) for _, r in teams_df.iterrows()}
        team_conf_map = {r["name"]: r["conference"] for _, r in teams_df.iterrows()}

        def seed_label(name):
            s = team_seed_map[name]
            return str(s) if s > 0 else "NR"

        col1, col2 = st.columns(2)
        with col1:
            team_a = st.selectbox("Team A", team_list, index=0)
        with col2:
            team_b = st.selectbox("Team B", team_list, index=min(1, len(team_list) - 1))

        if team_a == team_b:
            st.warning("Please select two different teams.")
            return

        if st.button("Predict Winner", type="primary"):
            prob = predict_matchup(model, graph, team_id_map[team_a], team_id_map[team_b])

            winner = team_a if prob > 0.5 else team_b
            confidence = prob if prob > 0.5 else 1 - prob

            st.markdown("---")

            # Display result
            col1, col2, col3 = st.columns([2, 1, 2])
            with col1:
                st.metric(
                    f"({seed_label(team_a)}) {team_a}",
                    f"{prob:.1%}",
                    delta=f"{team_conf_map[team_a]}",
                )
            with col2:
                st.markdown("<h2 style='text-align: center;'>vs</h2>", unsafe_allow_html=True)
            with col3:
                st.metric(
                    f"({seed_label(team_b)}) {team_b}",
                    f"{1-prob:.1%}",
                    delta=f"{team_conf_map[team_b]}",
                )

            st.markdown("---")
            st.subheader(f"Predicted Winner: ({seed_label(winner)}) {winner}")
            st.progress(confidence, text=f"Confidence: {confidence:.1%}")

    elif page == "Bracket Prediction":
        st.header("Full Tournament Bracket")

        if not model_exists:
            st.warning("No trained model found. Please train the model first.")
            return

        model, graph = load_model_and_graph()
        teams_df, _, tournament = load_data()

        if st.button("Simulate Bracket", type="primary"):
            results = simulate_bracket(model, graph, tournament, teams_df)

            for round_data in results:
                st.subheader(round_data["round_name"])
                games = round_data["games"]

                # Display as a table
                rows = []
                for g in games:
                    is_upset = g["winner_seed"] > min(g["seed_a"], g["seed_b"])
                    rows.append({
                        "Matchup": f"({g['seed_a']}) {g['team_a']}  vs  ({g['seed_b']}) {g['team_b']}",
                        "Winner": f"({g['winner_seed']}) {g['winner']}",
                        "Confidence": f"{g['confidence']:.1%}",
                        "Upset?": "UPSET" if is_upset else "",
                    })

                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Champion
            if results:
                final = results[-1]["games"][-1]
                st.markdown("---")
                st.markdown(
                    f"## Predicted Champion: ({final['winner_seed']}) {final['winner']}"
                )
                st.balloons()


if __name__ == "__main__":
    main()
