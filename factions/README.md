# Factions Points Visualization

This folder processes and visualizes weekly and cumulative points for factions, providing insights into team performance and individual contributions.

## Features

- Loads participant data from a CSV file (`factions.csv`)
- Generates three types of visualizations:

### 1. Team Weekly & Cumulative Points
- Line plot showing weekly points (solid lines) and cumulative totals (dashed lines) per team.
- Distinct color per team for clarity.
- Legends differentiate between weekly and cumulative lines.
- Output: `images/factions_weekly_progress.png`

### 2. Participant Score Distribution
- Stacked bar chart showing the distribution of participants by total score ranges.
- Highlights a specified team (`TEAM_HIGHLIGHT`) on top of the stack.
- Annotates participant names for the highlighted team (excluding 'Challenge' entries).
- Output: `images/factions_score_distribution.png`

### 3. Top 10 MVP Leaderboard
- Horizontal bar chart of the top 10 highest scoring participants (excluding 'Challenge' entries).
- Colored by team, with value labels on bars.
- Output: `factions_top_10_mvp.png`

## Usage

1. Place `factions.csv` in the project directory.
3. Run the script:
   ```bash
   ./venv/bin/python factions.py
   ```
4. Check the `images/` folder for generated plots.

