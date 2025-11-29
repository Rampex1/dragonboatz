# Fitness Score Visualization

This folder processes and visualizes fitness assessment data, producing plots of score distributions by gender.

## Features

- Loads participant data from a CSV file (`fittest.csv`) with the following columns:
  - `Name`, `Gender`, `Mobility`, `Core`, `Pullup`, `Bench`, `Pushup`, `Cardio`, `Total`
- Cleans and formats data
- Generates two main figures:
  1. **Component Score Distributions**  
     - Stacked bar charts for `Mobility`, `Core`, `Pullup`, `Bench`, `Pushup`, `Cardio`  
     - Shows male vs. female participants  
  2. **Overall Total Scores**  
     - Stacked bar chart of combined scores  
     - Includes summary statistics (mean ± standard deviation) by gender

## Output

- `images/fitness_tests.png` — component-level charts  
- `images/fitness_total.png` — overall total chart

## Usage

1. Place `fittest.csv` in the project directory.
2. Run the script:
   ```bash
   ./venv/bin/python FitTest.py
   ```
3. Check the `images/` folder for output plots.


