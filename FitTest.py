import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches

# -------------------------------
# 1. Load and prepare data
# -------------------------------
df = pd.read_csv('fittest.csv')

# Drop empty rows (rows where all key columns are missing)
df = df.dropna(subset=["Name", "Gender"], how='all')

# Define score columns
score_cols = ["Mobility", "Core", "Pullup", "Bench", "Pushup", "Cardio", "Total"]

# Convert numeric columns safely
df[score_cols] = df[score_cols].apply(pd.to_numeric, errors='coerce')

# Drop rows with all missing scores
df = df.dropna(subset=score_cols, how='all')

# Round to nearest 0.5 and keep as float
df[score_cols] = (df[score_cols] * 2).round() / 2

# Cap scores at defined maximums
max_values = {
    "Mobility": 2, "Core": 4, "Pullup": 4,
    "Bench": 4, "Pushup": 4, "Cardio": 4, "Total": 22
}

# -------------------------------
# 2. Color palette
# -------------------------------
COLORS = {
    'male': '#2E5C8A',      # Deep blue
    'female': '#D4526E',    # Rose
    'background': '#F8F9FA',
    'grid': '#E1E4E8',
    'text': '#2C3E50',
    'text_light': '#7F8C8D'
}


# -------------------------------
# 3. Helper functions
# -------------------------------
def get_gender_counts(column, max_score):
    males = df[df["Gender"] == "M"][column]
    females = df[df["Gender"] == "F"][column]

    # Allow 0.5 increments everywhere
    step = 0.5
    bins = np.arange(0, max_score + step + 0.001, step) - step / 2

    male_counts, _ = np.histogram(males, bins=bins)
    female_counts, _ = np.histogram(females, bins=bins)
    x = np.arange(0, max_score + 0.001, step)

    return x, male_counts, female_counts


def plot_stacked(ax, column, max_score, show_xlabel=True, show_ylabel=True):
    x, male_counts, female_counts = get_gender_counts(column, max_score)
    total_counts = male_counts + female_counts

    # Bar width and styling
    bar_width = 0.35 if column == "Total" else 0.5

    # Plot bars
    ax.bar(x, male_counts, width=bar_width, color=COLORS['male'],
           label="Male", edgecolor='white', linewidth=1.5, alpha=0.9)
    ax.bar(x, female_counts, bottom=male_counts, width=bar_width,
           color=COLORS['female'], label="Female",
           edgecolor='white', linewidth=1.5, alpha=0.9)

    # Add total labels on top
    for xi, m, f, t in zip(x, male_counts, female_counts, total_counts):
        if t > 0:
            ax.text(xi, t + 0.15, f'{int(t)}', ha='center', va='bottom',
                    fontsize=10, fontweight='600', color=COLORS['text'])

            if m > 0 and m / t > 0.15:
                ax.text(xi, m / 2, f'{int(m)}', ha='center', va='center',
                        fontsize=8, color='white', fontweight='500')
            if f > 0 and f / t > 0.15:
                ax.text(xi, m + f / 2, f'{int(f)}', ha='center', va='center',
                        fontsize=8, color='white', fontweight='500')

    # Axis and label formatting
    ax.set_title(f'{column}', fontsize=12, weight='600', pad=12, color=COLORS['text'])
    ax.text(0.5, 0.92, f'(max: {max_score})', transform=ax.transAxes,
            ha='center', fontsize=9, color=COLORS['text_light'], style='italic')

    # X-ticks (support 0.5 increments)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f'{v:.1f}' for v in x],
        fontsize=9, color=COLORS['text'],
        rotation=45 if len(x) > 8 else 0, ha='right' if len(x) > 8 else 'center'
    )

    # Clean y-axis
    ax.tick_params(axis='y', colors=COLORS['text'])

    # Handle case when no variation in counts
    ymax = max(total_counts) if max(total_counts) > 0 else 1
    ax.set_ylim(0, ymax * 1.25)

    # Grid styling
    ax.grid(axis='y', linestyle='--', alpha=0.3, color=COLORS['grid'], linewidth=0.8)
    ax.set_axisbelow(True)

    # Labels
    if show_xlabel:
        ax.set_xlabel('Score', fontsize=11, color=COLORS['text'],
                      fontweight='500', labelpad=8)
    if show_ylabel:
        ax.set_ylabel('Participants', fontsize=11, color=COLORS['text'],
                      fontweight='500', labelpad=8)

    # Clean spines
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS['grid'])
        spine.set_linewidth(1)

    ax.set_facecolor('white')


# -------------------------------
# 4. Figure 1 — Fitness Components
# -------------------------------
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

fig = plt.figure(figsize=(18, 10), facecolor=COLORS['background'])
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3,
                      left=0.07, right=0.95, top=0.88, bottom=0.08)

test_names = ["Mobility", "Core", "Pullup", "Bench", "Pushup", "Cardio"]

for i, col in enumerate(test_names):
    row = i // 3
    col_idx = i % 3
    ax = fig.add_subplot(gs[row, col_idx])
    show_ylabel = (col_idx == 0)
    show_xlabel = (row == 1)
    plot_stacked(ax, col, max_values[col], show_xlabel, show_ylabel)

# Legend
n_males = df[df["Gender"] == "M"].shape[0]
n_females = df[df["Gender"] == "F"].shape[0]
legend_elements = [
    mpatches.Patch(facecolor=COLORS['male'], edgecolor='white',
                   linewidth=1.5, label=f'Male (n={n_males})'),
    mpatches.Patch(facecolor=COLORS['female'], edgecolor='white',
                   linewidth=1.5, label=f'Female (n={n_females})')
]
legend = fig.legend(handles=legend_elements, loc='upper center',
                    ncol=2, frameon=True, fontsize=11,
                    bbox_to_anchor=(0.5, 0.97),
                    edgecolor=COLORS['grid'])
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.95)

fig.text(0.5, 0.955, 'Fitness Assessment Score Distribution',
         ha='center', fontsize=18, weight='700', color=COLORS['text'])
fig.text(0.5, 0.932, 'Performance breakdown by gender across six components',
         ha='center', fontsize=11, style='italic', color=COLORS['text_light'])

plt.savefig('fitness_tests_professional.png', dpi=300, bbox_inches='tight',
            facecolor=COLORS['background'])
plt.show()

# -------------------------------
# 5. Figure 2 — Overall Total
# -------------------------------
fig, ax = plt.subplots(figsize=(13, 7), facecolor=COLORS['background'])
plot_stacked(ax, "Total", max_values["Total"])

ax.set_title('Overall Fitness Score Distribution', fontsize=18,
             weight='700', pad=20, color=COLORS['text'])
ax.text(0.5, 0.96, f'Combined assessment across all fitness components (max: {max_values["Total"]})',
        transform=ax.transAxes, ha='center', fontsize=11,
        style='italic', color=COLORS['text_light'])

# Legend
legend = ax.legend(handles=legend_elements, loc='upper right',
                   frameon=True, fontsize=11, edgecolor=COLORS['grid'])
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.95)

# Summary statistics box
total_mean = df['Total'].mean()
total_std = df['Total'].std()
male_mean = df[df['Gender'] == 'M']['Total'].mean()
female_mean = df[df['Gender'] == 'F']['Total'].mean()

stats_text = f'Summary Statistics\n' \
             f'Overall: {total_mean:.1f} ± {total_std:.1f}\n' \
             f'Male: {male_mean:.1f}\n' \
             f'Female: {female_mean:.1f}'

ax.text(0.02, 0.97, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='white',
                  edgecolor=COLORS['grid'], alpha=0.95, linewidth=1),
        color=COLORS['text'], fontfamily='monospace')

plt.tight_layout()
plt.savefig('fitness_total_professional.png', dpi=300, bbox_inches='tight',
            facecolor=COLORS['background'])
plt.show()


# -------------------------------
# 6. Correlation with Total
# -------------------------------
corr = df[score_cols].corr(numeric_only=True)
total_corr = corr["Total"].sort_values(ascending=False)

print("\n=== Correlation with Total Score ===")
print(total_corr.to_string(float_format=lambda x: f"{x:.3f}"))

