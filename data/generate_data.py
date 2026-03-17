"""Generate CSV data files for the March Madness GNN.

Uses the actual 2026 NCAA tournament bracket with plausible team stats.
Run this script once to create teams.csv, games.csv, and tournament.csv.
"""

import csv
import random
import os

random.seed(42)

# All 68 tournament teams with (name, conference, seed, region)
# Using the 2026 bracket - First Four teams resolved to single entries for simplicity
TEAMS = [
    # EAST REGION
    ("Duke", "ACC", 1, "East"),
    ("Siena", "MAAC", 16, "East"),
    ("Ohio State", "Big Ten", 8, "East"),
    ("TCU", "Big 12", 9, "East"),
    ("St. John's", "Big East", 5, "East"),
    ("Northern Iowa", "MVC", 12, "East"),
    ("Kansas", "Big 12", 4, "East"),
    ("Cal Baptist", "WAC", 13, "East"),
    ("Louisville", "ACC", 6, "East"),
    ("South Florida", "AAC", 11, "East"),
    ("Michigan State", "Big Ten", 3, "East"),
    ("North Dakota State", "Summit", 14, "East"),
    ("UCLA", "Big Ten", 7, "East"),
    ("UCF", "Big 12", 10, "East"),
    ("UConn", "Big East", 2, "East"),
    ("Furman", "SoCon", 15, "East"),

    # WEST REGION
    ("Arizona", "Big 12", 1, "West"),
    ("LIU", "NEC", 16, "West"),
    ("Villanova", "Big East", 8, "West"),
    ("Utah State", "MWC", 9, "West"),
    ("Wisconsin", "Big Ten", 5, "West"),
    ("High Point", "Big South", 12, "West"),
    ("Arkansas", "SEC", 4, "West"),
    ("Hawaii", "Big West", 13, "West"),
    ("BYU", "Big 12", 6, "West"),
    ("NC State", "ACC", 11, "West"),
    ("Gonzaga", "WCC", 3, "West"),
    ("Kennesaw State", "ASUN", 14, "West"),
    ("Miami", "ACC", 7, "West"),
    ("Missouri", "SEC", 10, "West"),
    ("Purdue", "Big Ten", 2, "West"),
    ("Queens", "ASUN", 15, "West"),

    # MIDWEST REGION
    ("Michigan", "Big Ten", 1, "Midwest"),
    ("Howard", "MEAC", 16, "Midwest"),
    ("Georgia", "SEC", 8, "Midwest"),
    ("Saint Louis", "A-10", 9, "Midwest"),
    ("Texas Tech", "Big 12", 5, "Midwest"),
    ("Akron", "MAC", 12, "Midwest"),
    ("Alabama", "SEC", 4, "Midwest"),
    ("Hofstra", "CAA", 13, "Midwest"),
    ("Tennessee", "SEC", 6, "Midwest"),
    ("SMU", "ACC", 11, "Midwest"),
    ("Virginia", "ACC", 3, "Midwest"),
    ("Wright State", "Horizon", 14, "Midwest"),
    ("Kentucky", "SEC", 7, "Midwest"),
    ("Santa Clara", "WCC", 10, "Midwest"),
    ("Iowa State", "Big 12", 2, "Midwest"),
    ("Tennessee State", "OVC", 15, "Midwest"),

    # SOUTH REGION
    ("Florida", "SEC", 1, "South"),
    ("Lehigh", "Patriot", 16, "South"),
    ("Clemson", "ACC", 8, "South"),
    ("Iowa", "Big Ten", 9, "South"),
    ("Vanderbilt", "SEC", 5, "South"),
    ("McNeese", "Southland", 12, "South"),
    ("Nebraska", "Big Ten", 4, "South"),
    ("Troy", "Sun Belt", 13, "South"),
    ("North Carolina", "ACC", 6, "South"),
    ("VCU", "A-10", 10, "South"),
    ("Illinois", "Big Ten", 3, "South"),
    ("Penn", "Ivy", 14, "South"),
    ("Saint Mary's", "WCC", 7, "South"),
    ("Texas A&M", "SEC", 10, "South"),
    ("Houston", "Big 12", 2, "South"),
    ("Idaho", "Big Sky", 15, "South"),
]


def generate_team_stats(name, conference, seed):
    """Generate plausible stats based on seed (lower seed = better team)."""
    # Base stats that scale with seed quality
    seed_factor = (17 - seed) / 16  # 1.0 for seed 1, ~0.06 for seed 16

    wins = int(22 + seed_factor * 12 + random.gauss(0, 2))
    losses = int(14 - seed_factor * 8 + random.gauss(0, 2))
    wins = max(15, min(35, wins))
    losses = max(2, min(18, losses))

    ppg = round(72 + seed_factor * 12 + random.gauss(0, 3), 1)
    opp_ppg = round(72 - seed_factor * 8 + random.gauss(0, 3), 1)

    off_eff = round(100 + seed_factor * 20 + random.gauss(0, 3), 1)
    def_eff = round(105 - seed_factor * 15 + random.gauss(0, 3), 1)

    sos = round(seed_factor * 12 + random.gauss(0, 2), 1)
    sos = max(0, min(15, sos))

    return wins, losses, ppg, opp_ppg, off_eff, def_eff, sos


def generate_games(teams_data):
    """Generate plausible game results between teams.

    Creates ~800 games: conference games + cross-conference matchups.
    """
    games = []
    n = len(teams_data)

    # Group teams by conference
    conf_teams = {}
    for tid, (name, conf, seed, region, *stats) in enumerate(teams_data):
        conf_teams.setdefault(conf, []).append(tid)

    # Conference games: each pair plays 1-2 times
    for conf, members in conf_teams.items():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                num_games = random.choice([1, 2])
                for _ in range(num_games):
                    a, b = members[i], members[j]
                    games.append(simulate_game(a, b, teams_data))

    # Cross-conference games: random matchups (~300 extra)
    for _ in range(300):
        a = random.randint(0, n - 1)
        b = random.randint(0, n - 1)
        if a != b:
            games.append(simulate_game(a, b, teams_data))

    return games


def simulate_game(a_id, b_id, teams_data):
    """Simulate a single game based on team strength (seed-based)."""
    a_seed = teams_data[a_id][2]
    b_seed = teams_data[b_id][2]

    # Better seed (lower number) has higher win probability
    a_strength = (17 - a_seed) / 16
    b_strength = (17 - b_seed) / 16

    a_off = teams_data[a_id][6]  # ppg
    b_off = teams_data[b_id][6]

    # Base scores
    a_score = int(a_off + random.gauss(0, 8))
    b_score = int(b_off + random.gauss(0, 8))

    # Add strength advantage
    diff = (a_strength - b_strength) * 10
    a_score += int(diff / 2)
    b_score -= int(diff / 2)

    # Ensure no ties
    if a_score == b_score:
        if random.random() < 0.5 + (a_strength - b_strength) * 0.3:
            a_score += random.randint(1, 5)
        else:
            b_score += random.randint(1, 5)

    a_score = max(45, a_score)
    b_score = max(45, b_score)

    return a_id, b_id, a_score, b_score


def main():
    outdir = os.path.dirname(__file__)

    # Generate team stats
    teams_data = []
    for name, conf, seed, region in TEAMS:
        wins, losses, ppg, opp_ppg, off_eff, def_eff, sos = generate_team_stats(name, conf, seed)
        teams_data.append((name, conf, seed, region, wins, losses, ppg, opp_ppg, off_eff, def_eff, sos))

    # Write teams.csv
    with open(os.path.join(outdir, "teams.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["team_id", "name", "conference", "seed", "region", "wins", "losses", "ppg", "opp_ppg", "off_efficiency", "def_efficiency", "sos"])
        for i, (name, conf, seed, region, wins, losses, ppg, opp_ppg, off_eff, def_eff, sos) in enumerate(teams_data):
            w.writerow([i, name, conf, seed, region, wins, losses, ppg, opp_ppg, off_eff, def_eff, sos])

    # Generate and write games.csv
    games = generate_games(teams_data)
    with open(os.path.join(outdir, "games.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["team_a_id", "team_b_id", "score_a", "score_b"])
        for a, b, sa, sb in games:
            w.writerow([a, b, sa, sb])

    # Write tournament.csv (first round matchups by region)
    with open(os.path.join(outdir, "tournament.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "team_a_id", "team_b_id"])
        # Each region has 16 teams at indices [region*16 .. region*16+15]
        # Matchups: 0v1 (1v16), 2v3 (8v9), 4v5 (5v12), 6v7 (4v13), 8v9 (6v11), 10v11 (3v14), 12v13 (7v10), 14v15 (2v15)
        for region_start in range(0, 64, 16):
            matchup_pairs = [
                (0, 1), (2, 3), (4, 5), (6, 7),
                (8, 9), (10, 11), (12, 13), (14, 15),
            ]
            for a_off, b_off in matchup_pairs:
                w.writerow([1, region_start + a_off, region_start + b_off])

    print(f"Generated {len(teams_data)} teams, {len(games)} games, 32 first-round matchups")
    print(f"Files written to {outdir}/")


if __name__ == "__main__":
    main()
