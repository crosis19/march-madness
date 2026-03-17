"""Generate CSV data files for the March Madness GNN.

Includes ALL NCAA Division I basketball teams (~362 teams) across all 32
conferences, not just the 68 tournament teams. Non-tournament teams get
seed=0 and region="". Stats are generated based on conference strength
tier and random variation to produce realistic distributions.

Run: python data/generate_data.py
"""

import csv
import random
import os

random.seed(42)

# ---------------------------------------------------------------------------
# All Division I teams organized by (conference_name, tier, [team_names])
# Tier 1 = Power conferences, Tier 5 = low-major
# ---------------------------------------------------------------------------

CONFERENCE_TEAMS = [
    # ---- TIER 1: Power conferences ----
    ("ACC", 1, [
        "Boston College", "California", "Clemson", "Duke", "Florida State",
        "Georgia Tech", "Louisville", "Miami", "NC State", "North Carolina",
        "Notre Dame", "Pittsburgh", "SMU", "Stanford", "Syracuse",
        "Virginia", "Virginia Tech", "Wake Forest",
    ]),
    ("Big 12", 1, [
        "Arizona", "Arizona State", "Baylor", "BYU", "Cincinnati",
        "Colorado", "Houston", "Iowa State", "Kansas", "Kansas State",
        "Oklahoma State", "TCU", "Texas Tech", "UCF", "Utah",
        "West Virginia",
    ]),
    ("Big East", 1, [
        "Butler", "UConn", "Creighton", "DePaul", "Georgetown",
        "Marquette", "Providence", "Seton Hall", "St. John's",
        "Villanova", "Xavier",
    ]),
    ("Big Ten", 1, [
        "Illinois", "Indiana", "Iowa", "Maryland", "Michigan",
        "Michigan State", "Minnesota", "Nebraska", "Northwestern",
        "Ohio State", "Oregon", "Penn State", "Purdue", "Rutgers",
        "UCLA", "USC", "Washington", "Wisconsin",
    ]),
    ("SEC", 1, [
        "Alabama", "Arkansas", "Auburn", "Florida", "Georgia",
        "Kentucky", "LSU", "Mississippi State", "Missouri", "Oklahoma",
        "Ole Miss", "South Carolina", "Tennessee", "Texas",
        "Texas A&M", "Vanderbilt",
    ]),

    # ---- TIER 2: Strong mid-majors ----
    ("AAC", 2, [
        "Charlotte", "East Carolina", "FAU", "Memphis", "North Texas",
        "Rice", "South Florida", "Temple", "Tulane", "Tulsa",
        "UAB", "UTSA", "Wichita State",
    ]),
    ("A-10", 2, [
        "Davidson", "Dayton", "Duquesne", "Fordham", "George Mason",
        "George Washington", "La Salle", "Loyola Chicago",
        "Massachusetts", "Rhode Island", "Richmond",
        "Saint Joseph's", "Saint Louis", "St. Bonaventure", "VCU",
    ]),
    ("MWC", 2, [
        "Air Force", "Boise State", "Colorado State", "Fresno State",
        "Nevada", "New Mexico", "San Diego State", "San Jose State",
        "UNLV", "Utah State", "Wyoming",
    ]),
    ("WCC", 2, [
        "Gonzaga", "LMU", "Oregon State", "Pacific", "Pepperdine",
        "Portland", "Saint Mary's", "San Diego", "San Francisco",
        "Santa Clara",
    ]),
    ("MVC", 2, [
        "Belmont", "Bradley", "Drake", "Evansville", "Illinois State",
        "Indiana State", "Missouri State", "Murray State",
        "Northern Iowa", "Southern Illinois", "UIC", "Valparaiso",
    ]),

    # ---- TIER 3: Mid-level conferences ----
    ("MAC", 3, [
        "Akron", "Ball State", "Bowling Green", "Buffalo",
        "Central Michigan", "Eastern Michigan", "Kent State",
        "Miami (OH)", "Northern Illinois", "Ohio", "Toledo",
        "Western Michigan",
    ]),
    ("Sun Belt", 3, [
        "Appalachian State", "Arkansas State", "Coastal Carolina",
        "Georgia Southern", "Georgia State", "James Madison",
        "Louisiana", "Louisiana-Monroe", "Marshall", "Old Dominion",
        "South Alabama", "Southern Miss", "Texas State", "Troy",
    ]),
    ("C-USA", 3, [
        "FIU", "Jacksonville State", "Kennesaw State", "Liberty",
        "Louisiana Tech", "Middle Tennessee", "New Mexico State",
        "Sam Houston", "UTEP", "Western Kentucky",
    ]),
    ("CAA", 3, [
        "Campbell", "Charleston", "Delaware", "Drexel", "Elon",
        "Hampton", "Hofstra", "Monmouth", "UNCW", "Northeastern",
        "Stony Brook", "Towson", "William & Mary",
    ]),
    ("ASUN", 3, [
        "Austin Peay", "Bellarmine", "Central Arkansas",
        "Eastern Kentucky", "Florida Gulf Coast", "Jacksonville",
        "Lipscomb", "North Alabama", "North Florida", "Queens",
        "Stetson", "West Georgia",
    ]),
    ("Horizon", 3, [
        "Cleveland State", "Detroit Mercy", "Green Bay",
        "IU Indianapolis", "Milwaukee", "Northern Kentucky",
        "Oakland", "Purdue Fort Wayne", "Robert Morris",
        "Wright State", "Youngstown State",
    ]),
    ("SoCon", 3, [
        "Chattanooga", "ETSU", "Furman", "Mercer", "Samford",
        "The Citadel", "UNC Greensboro", "VMI", "Western Carolina",
        "Wofford",
    ]),

    # ---- TIER 4: Lower mid-level ----
    ("Patriot", 4, [
        "American", "Army", "Boston University", "Bucknell", "Colgate",
        "Holy Cross", "Lafayette", "Lehigh", "Loyola Maryland", "Navy",
    ]),
    ("Ivy", 4, [
        "Brown", "Columbia", "Cornell", "Dartmouth", "Harvard",
        "Penn", "Princeton", "Yale",
    ]),
    ("MAAC", 4, [
        "Canisius", "Fairfield", "Iona", "Manhattan", "Marist",
        "Mount St. Mary's", "Niagara", "Quinnipiac", "Rider",
        "Saint Peter's", "Siena",
    ]),
    ("Big Sky", 4, [
        "Eastern Washington", "Idaho", "Idaho State", "Montana",
        "Montana State", "Northern Arizona", "Northern Colorado",
        "Portland State", "Sacramento State", "Weber State",
    ]),
    ("NEC", 4, [
        "Central Connecticut", "Fairleigh Dickinson", "LIU",
        "Le Moyne", "Merrimack", "Sacred Heart",
        "St. Francis Brooklyn", "Stonehill", "Wagner",
    ]),
    ("OVC", 4, [
        "Eastern Illinois", "Lindenwood", "Little Rock",
        "Morehead State", "SE Missouri State", "SIU Edwardsville",
        "Tennessee State", "Tennessee Tech", "UT Martin",
    ]),
    ("Big South", 4, [
        "Charleston Southern", "Gardner-Webb", "High Point",
        "Longwood", "Presbyterian", "Radford",
        "UNC Asheville", "Winthrop",
    ]),
    ("Big West", 4, [
        "Cal Poly", "Cal State Bakersfield", "Cal State Fullerton",
        "Cal State Northridge", "Hawaii", "Long Beach State",
        "UC Davis", "UC Irvine", "UC Riverside", "UC San Diego",
        "UC Santa Barbara",
    ]),
    ("Summit", 4, [
        "Denver", "Kansas City", "North Dakota",
        "North Dakota State", "Omaha", "Oral Roberts",
        "South Dakota", "South Dakota State", "St. Thomas",
    ]),
    ("WAC", 4, [
        "Abilene Christian", "Cal Baptist", "Grand Canyon",
        "Lamar", "Seattle", "Southern Utah",
        "Stephen F. Austin", "Tarleton", "UT Arlington",
        "UT Rio Grande Valley", "Utah Tech", "Utah Valley",
    ]),
    ("Southland", 4, [
        "Houston Christian", "Incarnate Word", "McNeese",
        "Nicholls", "Northwestern State", "Southeastern Louisiana",
        "Texas A&M-Corpus Christi",
    ]),

    # ---- TIER 5: Low-major conferences ----
    ("MEAC", 5, [
        "Coppin State", "Delaware State", "Howard",
        "Maryland-Eastern Shore", "Morgan State", "Norfolk State",
        "North Carolina Central", "South Carolina State",
    ]),
    ("SWAC", 5, [
        "Alabama A&M", "Alabama State", "Alcorn State",
        "Arkansas-Pine Bluff", "Bethune-Cookman", "Florida A&M",
        "Grambling", "Jackson State", "Mississippi Valley State",
        "Prairie View A&M", "Southern", "Texas Southern",
    ]),
]

# ---------------------------------------------------------------------------
# Tournament field: 64 teams with (seed, region).
# Uses projected 2025-26 bracket (Duke, Arizona, Michigan, Florida as 1-seeds).
# All other D1 teams will have seed=0, region="".
# ---------------------------------------------------------------------------

TOURNAMENT_FIELD = {
    # EAST REGION
    "Duke": (1, "East"),
    "Siena": (16, "East"),
    "Ohio State": (8, "East"),
    "TCU": (9, "East"),
    "St. John's": (5, "East"),
    "Northern Iowa": (12, "East"),
    "Kansas": (4, "East"),
    "Cal Baptist": (13, "East"),
    "Louisville": (6, "East"),
    "South Florida": (11, "East"),
    "Michigan State": (3, "East"),
    "North Dakota State": (14, "East"),
    "UCLA": (7, "East"),
    "UCF": (10, "East"),
    "UConn": (2, "East"),
    "Furman": (15, "East"),

    # WEST REGION
    "Arizona": (1, "West"),
    "LIU": (16, "West"),
    "Villanova": (8, "West"),
    "Utah State": (9, "West"),
    "Wisconsin": (5, "West"),
    "High Point": (12, "West"),
    "Arkansas": (4, "West"),
    "Hawaii": (13, "West"),
    "BYU": (6, "West"),
    "NC State": (11, "West"),
    "Gonzaga": (3, "West"),
    "Kennesaw State": (14, "West"),
    "Miami": (7, "West"),
    "Missouri": (10, "West"),
    "Purdue": (2, "West"),
    "Queens": (15, "West"),

    # MIDWEST REGION
    "Michigan": (1, "Midwest"),
    "Howard": (16, "Midwest"),
    "Georgia": (8, "Midwest"),
    "Saint Louis": (9, "Midwest"),
    "Texas Tech": (5, "Midwest"),
    "Akron": (12, "Midwest"),
    "Alabama": (4, "Midwest"),
    "Hofstra": (13, "Midwest"),
    "Tennessee": (6, "Midwest"),
    "SMU": (11, "Midwest"),
    "Virginia": (3, "Midwest"),
    "Wright State": (14, "Midwest"),
    "Kentucky": (7, "Midwest"),
    "Santa Clara": (10, "Midwest"),
    "Iowa State": (2, "Midwest"),
    "Tennessee State": (15, "Midwest"),

    # SOUTH REGION
    "Florida": (1, "South"),
    "Lehigh": (16, "South"),
    "Clemson": (8, "South"),
    "Iowa": (9, "South"),
    "Vanderbilt": (5, "South"),
    "McNeese": (12, "South"),
    "Nebraska": (4, "South"),
    "Troy": (13, "South"),
    "North Carolina": (6, "South"),
    "VCU": (11, "South"),
    "Illinois": (3, "South"),
    "Penn": (14, "South"),
    "Saint Mary's": (7, "South"),
    "Texas A&M": (10, "South"),
    "Houston": (2, "South"),
    "Idaho": (15, "South"),
}

# ---------------------------------------------------------------------------
# Strength tiers: base power range [0, 1] for each conference tier
# ---------------------------------------------------------------------------

TIER_POWER = {
    1: (0.55, 0.95),   # Power conference teams range 0.55-0.95
    2: (0.40, 0.75),   # Strong mid-majors
    3: (0.25, 0.60),   # Mid-level
    4: (0.15, 0.50),   # Lower mid
    5: (0.08, 0.35),   # Low-major
}


def assign_power_ratings(all_teams):
    """Assign a power rating [0, 1] to each team.

    Tournament teams have their power adjusted to be consistent with their seed.
    """
    power = {}
    for name, conf, tier in all_teams:
        lo, hi = TIER_POWER[tier]

        # Check if tournament team
        if name in TOURNAMENT_FIELD:
            seed, _ = TOURNAMENT_FIELD[name]
            # Seed 1 → power ~0.95, seed 16 → power ~0.35
            seed_power = 0.35 + (16 - seed) / 15 * 0.60
            # Blend with conference tier range, biased toward seed
            base = seed_power * 0.8 + (lo + hi) / 2 * 0.2
        else:
            base = random.uniform(lo, hi)

        # Add small noise
        base += random.gauss(0, 0.03)
        base = max(0.05, min(0.98, base))
        power[name] = base

    return power


def generate_team_stats(name, power):
    """Generate plausible stats from a team's power rating [0, 1]."""
    # Wins/losses: power 1.0 → ~33-4, power 0.1 → ~8-24
    wins = int(8 + power * 27 + random.gauss(0, 2))
    losses = int(24 - power * 20 + random.gauss(0, 2))
    wins = max(3, min(36, wins))
    losses = max(2, min(28, losses))

    ppg = round(64 + power * 22 + random.gauss(0, 3), 1)
    opp_ppg = round(76 - power * 16 + random.gauss(0, 3), 1)

    off_eff = round(95 + power * 30 + random.gauss(0, 3), 1)
    def_eff = round(110 - power * 22 + random.gauss(0, 3), 1)

    # Strength of schedule correlates with conference strength
    sos = round(power * 12 + random.gauss(0, 2), 1)
    sos = max(0.0, min(15.0, sos))

    # Net rating = off_eff - def_eff (KenPom-style adjusted efficiency margin)
    net_rating = round(off_eff - def_eff, 1)

    return wins, losses, ppg, opp_ppg, off_eff, def_eff, sos, net_rating


def simulate_game(a_id, b_id, teams_data, power_ratings):
    """Simulate a single game based on power ratings."""
    a_name = teams_data[a_id][0]
    b_name = teams_data[b_id][0]
    a_power = power_ratings[a_name]
    b_power = power_ratings[b_name]

    a_ppg = teams_data[a_id][6]  # ppg column (index 6 in tuple)
    b_ppg = teams_data[b_id][6]

    # Base scores from team's ppg + noise
    a_score = int(a_ppg + random.gauss(0, 8))
    b_score = int(b_ppg + random.gauss(0, 8))

    # Strength advantage
    diff = (a_power - b_power) * 12
    a_score += int(diff / 2)
    b_score -= int(diff / 2)

    # Ensure no ties
    if a_score == b_score:
        if random.random() < 0.5 + (a_power - b_power) * 0.3:
            a_score += random.randint(1, 5)
        else:
            b_score += random.randint(1, 5)

    a_score = max(40, a_score)
    b_score = max(40, b_score)

    return a_id, b_id, a_score, b_score


def generate_games(teams_data, conf_members, power_ratings):
    """Generate a full season of games.

    - Conference games: each pair plays 1-2 times depending on conference size.
    - Non-conference games: ~10 per team with bias toward comparable opponents.
    """
    games = []
    n = len(teams_data)

    # Conference games
    for conf, members in conf_members.items():
        size = len(members)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                # Large conferences (>14): play once, some rematches
                # Smaller conferences: play twice
                if size > 14:
                    games.append(simulate_game(a, b, teams_data, power_ratings))
                    if random.random() < 0.25:  # ~25% chance of rematch
                        games.append(simulate_game(a, b, teams_data, power_ratings))
                else:
                    games.append(simulate_game(a, b, teams_data, power_ratings))
                    games.append(simulate_game(b, a, teams_data, power_ratings))

    # Non-conference games: ~10 per team
    # Group teams by power tier for somewhat realistic scheduling
    team_ids = list(range(n))
    target_nonconf = 10 * n // 2  # 10 games per team, but each game involves 2

    for _ in range(target_nonconf):
        a = random.choice(team_ids)
        # 60% chance to play a team within ±0.2 power, 40% random
        if random.random() < 0.6:
            a_power = power_ratings[teams_data[a][0]]
            candidates = [
                t for t in team_ids
                if t != a and abs(power_ratings[teams_data[t][0]] - a_power) < 0.25
            ]
            if candidates:
                b = random.choice(candidates)
            else:
                b = random.choice([t for t in team_ids if t != a])
        else:
            b = random.choice([t for t in team_ids if t != a])
        games.append(simulate_game(a, b, teams_data, power_ratings))

    return games


def main():
    outdir = os.path.dirname(__file__)

    # Build flat team list: (name, conference, tier)
    all_teams = []
    for conf, tier, members in CONFERENCE_TEAMS:
        for name in members:
            all_teams.append((name, conf, tier))

    print(f"Total D1 teams: {len(all_teams)}")

    # Assign power ratings
    power_ratings = assign_power_ratings(all_teams)

    # Generate team stats and build data rows
    teams_data = []  # list of tuples: (name, conf, seed, region, wins, losses, ppg, ...)
    conf_members = {}  # conference -> list of team_ids

    for tid, (name, conf, tier) in enumerate(all_teams):
        seed, region = TOURNAMENT_FIELD.get(name, (0, ""))
        power = power_ratings[name]
        wins, losses, ppg, opp_ppg, off_eff, def_eff, sos, net_rating = generate_team_stats(name, power)
        teams_data.append((name, conf, seed, region, wins, losses, ppg, opp_ppg,
                           off_eff, def_eff, sos, net_rating))
        conf_members.setdefault(conf, []).append(tid)

    # Write teams.csv
    with open(os.path.join(outdir, "teams.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "team_id", "name", "conference", "seed", "region",
            "wins", "losses", "ppg", "opp_ppg",
            "off_efficiency", "def_efficiency", "sos", "net_rating",
        ])
        for i, row in enumerate(teams_data):
            w.writerow([i] + list(row))

    # Generate and write games.csv
    games = generate_games(teams_data, conf_members, power_ratings)
    with open(os.path.join(outdir, "games.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["team_a_id", "team_b_id", "score_a", "score_b"])
        for a, b, sa, sb in games:
            w.writerow([a, b, sa, sb])

    # Write tournament.csv
    # Need to map tournament team names to their team_ids
    name_to_id = {row[0]: i for i, row in enumerate(teams_data)}

    # Build bracket matchups by region
    regions = ["East", "West", "Midwest", "South"]
    seed_matchups = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

    with open(os.path.join(outdir, "tournament.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "team_a_id", "team_b_id"])

        for region in regions:
            # Get teams in this region, sorted by seed
            region_teams = {
                seed: name
                for name, (seed, reg) in TOURNAMENT_FIELD.items()
                if reg == region
            }
            for seed_a, seed_b in seed_matchups:
                a_name = region_teams[seed_a]
                b_name = region_teams[seed_b]
                w.writerow([1, name_to_id[a_name], name_to_id[b_name]])

    print(f"Generated {len(teams_data)} teams, {len(games)} games, 32 first-round matchups")
    print(f"Files written to {outdir}/")


if __name__ == "__main__":
    main()
