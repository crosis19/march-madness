"""Scrape real 2025-26 NCAA D1 basketball data from Sports Reference.

Fetches:
  1. All team stats/ratings from the season ratings page
  2. Game-by-game results from each team's schedule page
  3. Tournament bracket matchups

Outputs: teams.csv, games.csv, tournament.csv
"""

import csv
import os
import re
import time
import sys

import requests
from bs4 import BeautifulSoup, Comment

BASE_URL = "https://www.sports-reference.com/cbb"
SEASON = 2026  # 2025-26 season
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
OUTDIR = os.path.dirname(os.path.abspath(__file__))

# Rate limiting: seconds between requests
RATE_LIMIT = 3.5


def fetch(url, retries=3):
    """Fetch a URL with retries and rate limiting."""
    for attempt in range(retries):
        try:
            time.sleep(RATE_LIMIT)
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                return resp.text
            elif resp.status_code == 429:
                wait = 2 ** (attempt + 2)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  HTTP {resp.status_code} for {url}")
                if attempt < retries - 1:
                    time.sleep(2 ** (attempt + 1))
        except requests.RequestException as e:
            print(f"  Request error: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
    return None


def parse_ratings_table(html):
    """Parse the ratings page to get all team stats.

    Returns list of dicts with: name, slug, conference, wins, losses,
    ORtg, DRtg, NRtg, SOS, SRS
    """
    soup = BeautifulSoup(html, "lxml")

    # Sports Reference sometimes hides tables in HTML comments
    table = soup.find("table", id="ratings")
    if not table:
        # Look in comments
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            if "ratings" in comment:
                comment_soup = BeautifulSoup(comment, "lxml")
                table = comment_soup.find("table", id="ratings")
                if table:
                    break

    if not table:
        print("ERROR: Could not find ratings table")
        return []

    teams = []
    tbody = table.find("tbody")
    if not tbody:
        return []

    for row in tbody.find_all("tr"):
        if row.get("class") and "thead" in row.get("class", []):
            continue

        cells = row.find_all(["th", "td"])
        if len(cells) < 10:
            continue

        # Find the school cell (has a link)
        school_cell = row.find("td", {"data-stat": "school_name"})
        if not school_cell:
            continue

        link = school_cell.find("a")
        if not link:
            continue

        name = link.text.strip()
        href = link.get("href", "")
        # Extract slug from /cbb/schools/duke/2026.html
        slug_match = re.search(r"/schools/([^/]+)/", href)
        slug = slug_match.group(1) if slug_match else ""

        def get_stat(stat_name):
            cell = row.find("td", {"data-stat": stat_name})
            if cell and cell.text.strip():
                try:
                    return float(cell.text.strip())
                except ValueError:
                    return 0.0
            return 0.0

        conf_cell = row.find("td", {"data-stat": "conf_abbr"})
        conference = conf_cell.text.strip() if conf_cell else ""

        wins = int(get_stat("wins"))
        losses = int(get_stat("losses"))

        teams.append({
            "name": name,
            "slug": slug,
            "conference": conference,
            "wins": wins,
            "losses": losses,
            "ORtg": get_stat("off_rtg"),
            "DRtg": get_stat("def_rtg"),
            "NRtg": get_stat("off_rtg") - get_stat("def_rtg"),
            "SOS": get_stat("sos"),
            "SRS": get_stat("srs"),
        })

    return teams


def parse_basic_stats(html):
    """Parse basic school stats page for PPG and opponent PPG.

    Returns dict mapping school name -> {ppg, opp_ppg}.
    """
    soup = BeautifulSoup(html, "lxml")

    table = soup.find("table", id="basic_school_stats")
    if not table:
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            if "basic_school_stats" in comment:
                comment_soup = BeautifulSoup(comment, "lxml")
                table = comment_soup.find("table", id="basic_school_stats")
                if table:
                    break

    if not table:
        print("WARNING: Could not find basic stats table")
        return {}

    stats = {}
    tbody = table.find("tbody")
    if not tbody:
        return {}

    for row in tbody.find_all("tr"):
        if row.get("class") and "thead" in row.get("class", []):
            continue

        school_cell = row.find("td", {"data-stat": "school_name"})
        if not school_cell or not school_cell.find("a"):
            continue

        name = school_cell.find("a").text.strip()

        def get_stat(stat_name):
            cell = row.find("td", {"data-stat": stat_name})
            if cell and cell.text.strip():
                try:
                    return float(cell.text.strip())
                except ValueError:
                    return 0.0
            return 0.0

        # Total points and games to compute PPG
        g = get_stat("g")
        pts = get_stat("pts")
        opp_pts = get_stat("opp_pts")

        if g > 0:
            stats[name] = {
                "ppg": round(pts / g, 1),
                "opp_ppg": round(opp_pts / g, 1),
            }

    return stats


def parse_advanced_stats(html):
    """Parse advanced stats page for offensive efficiency per team.

    Returns dict mapping school name -> {off_efficiency, def_efficiency, sos}.
    """
    soup = BeautifulSoup(html, "lxml")

    table = soup.find("table", id="adv_school_stats")
    if not table:
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            if "adv_school_stats" in comment or "advanced" in comment:
                comment_soup = BeautifulSoup(comment, "lxml")
                table = comment_soup.find("table", id="adv_school_stats")
                if table:
                    break

    if not table:
        print("WARNING: Could not find advanced stats table")
        return {}

    stats = {}
    tbody = table.find("tbody")
    if not tbody:
        return {}

    for row in tbody.find_all("tr"):
        if row.get("class") and "thead" in row.get("class", []):
            continue

        school_cell = row.find("td", {"data-stat": "school_name"})
        if not school_cell or not school_cell.find("a"):
            continue

        name = school_cell.find("a").text.strip()

        def get_stat(stat_name):
            cell = row.find("td", {"data-stat": stat_name})
            if cell and cell.text.strip():
                try:
                    return float(cell.text.strip())
                except ValueError:
                    return 0.0
            return 0.0

        stats[name] = {
            "off_efficiency": get_stat("off_rtg"),
            "sos": get_stat("sos"),
        }

    return stats


def scrape_team_schedule(slug, team_name, name_to_id):
    """Scrape a team's schedule page for game results.

    Returns list of (team_a_id, team_b_id, score_a, score_b).
    """
    url = f"{BASE_URL}/schools/{slug}/{SEASON}-schedule.html"
    html = fetch(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="schedule")
    if not table:
        # Try in comments
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            if "schedule" in comment:
                comment_soup = BeautifulSoup(comment, "lxml")
                table = comment_soup.find("table", id="schedule")
                if table:
                    break

    if not table:
        return []

    games = []
    tbody = table.find("tbody")
    if not tbody:
        return []

    team_id = name_to_id.get(team_name)
    if team_id is None:
        return []

    for row in tbody.find_all("tr"):
        if row.get("class") and "thead" in row.get("class", []):
            continue

        # Get opponent - field is "opp_name" with a link
        opp_cell = row.find("td", {"data-stat": "opp_name"})
        if not opp_cell:
            continue

        opp_link = opp_cell.find("a")
        if opp_link:
            opp_name = opp_link.text.strip()
        else:
            # Non-D1 opponent (no link)
            continue

        opp_id = name_to_id.get(opp_name)
        if opp_id is None:
            continue

        # Get scores
        def get_stat(stat_name):
            cell = row.find("td", {"data-stat": stat_name})
            if cell and cell.text.strip():
                try:
                    return int(cell.text.strip())
                except ValueError:
                    return None
            return None

        pts = get_stat("pts")
        opp_pts = get_stat("opp_pts")

        if pts is None or opp_pts is None:
            continue

        games.append((team_id, opp_id, pts, opp_pts))

    return games


def scrape_all_teams():
    """Scrape team stats from the season ratings and stats pages."""
    print("Fetching ratings page...")
    ratings_html = fetch(f"{BASE_URL}/seasons/men/{SEASON}-ratings.html")
    if not ratings_html:
        print("FATAL: Could not fetch ratings page")
        sys.exit(1)

    teams = parse_ratings_table(ratings_html)
    print(f"  Found {len(teams)} teams from ratings page")

    print("Fetching basic stats page...")
    basic_html = fetch(f"{BASE_URL}/seasons/men/{SEASON}-school-stats.html")
    basic_stats = {}
    if basic_html:
        basic_stats = parse_basic_stats(basic_html)
        print(f"  Found basic stats for {len(basic_stats)} teams")

    print("Fetching advanced stats page...")
    adv_html = fetch(f"{BASE_URL}/seasons/men/{SEASON}-advanced-school-stats.html")
    adv_stats = {}
    if adv_html:
        adv_stats = parse_advanced_stats(adv_html)
        print(f"  Found advanced stats for {len(adv_stats)} teams")

    # Merge stats
    for team in teams:
        name = team["name"]
        if name in basic_stats:
            team["ppg"] = basic_stats[name]["ppg"]
            team["opp_ppg"] = basic_stats[name]["opp_ppg"]
        else:
            # Estimate from efficiency ratings
            team["ppg"] = round(team["ORtg"] * 0.7, 1) if team["ORtg"] else 70.0
            team["opp_ppg"] = round(team["DRtg"] * 0.7, 1) if team["DRtg"] else 70.0

        if name in adv_stats:
            team["off_efficiency"] = adv_stats[name].get("off_efficiency", team["ORtg"])
        else:
            team["off_efficiency"] = team["ORtg"]

        team["def_efficiency"] = team["DRtg"]

    return teams


def scrape_games(teams, name_to_id):
    """Scrape game results from team schedule pages.

    To avoid excessive requests, we scrape games for all teams but
    deduplicate so each game appears only once.
    """
    all_games = {}  # (min_id, max_id, score_min, score_max) -> game tuple
    total = len(teams)

    for i, team in enumerate(teams):
        slug = team["slug"]
        name = team["name"]
        if not slug:
            continue

        print(f"  [{i+1}/{total}] Scraping schedule for {name}...")
        team_games = scrape_team_schedule(slug, name, name_to_id)

        for a_id, b_id, sa, sb in team_games:
            # Deduplicate: store each game once using canonical ordering
            key = (min(a_id, b_id), max(a_id, b_id))
            if key not in all_games:
                all_games[key] = (a_id, b_id, sa, sb)

        if (i + 1) % 25 == 0:
            print(f"    ... {len(all_games)} unique games so far")

    return list(all_games.values())


def scrape_tournament_bracket(teams, name_to_id):
    """Scrape the NCAA tournament bracket.

    Falls back to the bracket from the generate_data.py if scraping fails.
    """
    print("Fetching NCAA tournament bracket...")
    url = f"{BASE_URL}/postseason/men/{SEASON}-ncaa.html"
    html = fetch(url)

    if not html:
        print("  Could not fetch bracket page, using fallback bracket")
        return None

    soup = BeautifulSoup(html, "lxml")

    # Sports Reference bracket page has a specific structure
    # Look for the bracket div or table
    bracket = soup.find("div", id="bracket")
    if not bracket:
        bracket = soup.find("div", class_="bracket")

    if not bracket:
        print("  Could not parse bracket structure, using fallback")
        return None

    # Parse first round matchups from the bracket
    matchups = []
    round_divs = bracket.find_all("div", class_=re.compile(r"round"))

    if not round_divs:
        # Try alternative parsing: look for all team links in bracket
        # and pair them up
        links = bracket.find_all("a")
        team_names = []
        for link in links:
            href = link.get("href", "")
            if "/schools/" in href:
                team_names.append(link.text.strip())

        # First round: pair consecutive teams
        for i in range(0, min(len(team_names), 64), 2):
            a_name = team_names[i]
            b_name = team_names[i + 1] if i + 1 < len(team_names) else None
            if b_name:
                a_id = name_to_id.get(a_name)
                b_id = name_to_id.get(b_name)
                if a_id is not None and b_id is not None:
                    matchups.append((1, a_id, b_id))

    if len(matchups) >= 32:
        return matchups[:32]

    print(f"  Only found {len(matchups)} matchups, using fallback bracket")
    return None


def get_fallback_tournament(name_to_id):
    """Return the projected tournament bracket as fallback."""
    # Real 2025-26 NCAA tournament bracket using Sports Reference team names.
    # First Four winners TBD as of March 17 — using one team from each pair.
    bracket = {
        "East": [
            ("Duke", 1), ("Siena", 16), ("Ohio State", 8), ("Texas Christian", 9),
            ("St. John's (NY)", 5), ("Northern Iowa", 12), ("Kansas", 4), ("California Baptist", 13),
            ("Louisville", 6), ("South Florida", 11), ("Michigan State", 3), ("North Dakota State", 14),
            ("UCLA", 7), ("UCF", 10), ("Connecticut", 2), ("Furman", 15),
        ],
        "West": [
            ("Arizona", 1), ("Long Island University", 16), ("Villanova", 8), ("Utah State", 9),
            ("Wisconsin", 5), ("High Point", 12), ("Arkansas", 4), ("Hawaii", 13),
            ("Brigham Young", 6), ("NC State", 11), ("Gonzaga", 3), ("Kennesaw State", 14),
            ("Miami (FL)", 7), ("Missouri", 10), ("Purdue", 2), ("Queens (NC)", 15),
        ],
        "Midwest": [
            ("Michigan", 1), ("Howard", 16), ("Georgia", 8), ("Saint Louis", 9),
            ("Texas Tech", 5), ("Akron", 12), ("Alabama", 4), ("Hofstra", 13),
            ("Tennessee", 6), ("Southern Methodist", 11), ("Virginia", 3), ("Wright State", 14),
            ("Kentucky", 7), ("Santa Clara", 10), ("Iowa State", 2), ("Tennessee State", 15),
        ],
        "South": [
            ("Florida", 1), ("Lehigh", 16), ("Clemson", 8), ("Iowa", 9),
            ("Vanderbilt", 5), ("McNeese", 12), ("Nebraska", 4), ("Troy", 13),
            ("North Carolina", 6), ("Virginia Commonwealth", 11), ("Illinois", 3), ("Pennsylvania", 14),
            ("Saint Mary's", 7), ("Texas A&M", 10), ("Houston", 2), ("Idaho", 15),
        ],
    }

    matchups = []
    seed_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
    tournament_seeds = {}

    for region, teams in bracket.items():
        for name, seed in teams:
            tournament_seeds[name] = (seed, region)
        for a_idx, b_idx in seed_pairs:
            a_name = teams[a_idx][0]
            b_name = teams[b_idx][0]
            a_id = name_to_id.get(a_name)
            b_id = name_to_id.get(b_name)
            if a_id is not None and b_id is not None:
                matchups.append((1, a_id, b_id))
            else:
                missing = []
                if a_id is None:
                    missing.append(a_name)
                if b_id is None:
                    missing.append(b_name)
                print(f"  WARNING: Tournament team(s) not found in scraped data: {missing}")

    return matchups, tournament_seeds


def main():
    # Step 1: Scrape team stats
    print("=" * 60)
    print("STEP 1: Scraping team stats")
    print("=" * 60)
    teams = scrape_all_teams()

    if not teams:
        print("FATAL: No teams found. Exiting.")
        sys.exit(1)

    # Assign team_ids and build lookup
    name_to_id = {}
    slug_to_name = {}
    for i, team in enumerate(teams):
        team["team_id"] = i
        name_to_id[team["name"]] = i
        if team["slug"]:
            slug_to_name[team["slug"]] = team["name"]

    print(f"\nTotal teams: {len(teams)}")

    # Step 2: Scrape game results
    print("\n" + "=" * 60)
    print("STEP 2: Scraping game results")
    print("=" * 60)
    games = scrape_games(teams, name_to_id)
    print(f"\nTotal unique games: {len(games)}")

    # Step 3: Tournament bracket
    print("\n" + "=" * 60)
    print("STEP 3: Getting tournament bracket")
    print("=" * 60)
    bracket_matchups = scrape_tournament_bracket(teams, name_to_id)
    if bracket_matchups is None:
        bracket_matchups, tournament_seeds = get_fallback_tournament(name_to_id)
    else:
        tournament_seeds = {}

    # Step 4: Write CSVs
    print("\n" + "=" * 60)
    print("STEP 4: Writing CSV files")
    print("=" * 60)

    # teams.csv
    with open(os.path.join(OUTDIR, "teams.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "team_id", "name", "conference", "seed", "region",
            "wins", "losses", "ppg", "opp_ppg",
            "off_efficiency", "def_efficiency", "sos", "net_rating",
        ])
        for team in teams:
            seed, region = tournament_seeds.get(team["name"], (0, ""))
            w.writerow([
                team["team_id"],
                team["name"],
                team["conference"],
                seed,
                region,
                team["wins"],
                team["losses"],
                team.get("ppg", 70.0),
                team.get("opp_ppg", 70.0),
                round(team.get("off_efficiency", team.get("ORtg", 100.0)), 1),
                round(team.get("def_efficiency", team.get("DRtg", 100.0)), 1),
                round(team.get("SOS", 0.0), 2),
                round(team.get("NRtg", 0.0), 1),
            ])
    print(f"  teams.csv: {len(teams)} teams")

    # games.csv
    with open(os.path.join(OUTDIR, "games.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["team_a_id", "team_b_id", "score_a", "score_b"])
        for a_id, b_id, sa, sb in games:
            w.writerow([a_id, b_id, sa, sb])
    print(f"  games.csv: {len(games)} games")

    # tournament.csv
    with open(os.path.join(OUTDIR, "tournament.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "team_a_id", "team_b_id"])
        for round_num, a_id, b_id in bracket_matchups:
            w.writerow([round_num, a_id, b_id])
    print(f"  tournament.csv: {len(bracket_matchups)} matchups")

    print("\nDone!")


if __name__ == "__main__":
    main()
