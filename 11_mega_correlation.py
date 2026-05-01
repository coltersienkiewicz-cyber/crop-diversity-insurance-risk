"""Correlation matrix for ~50 variables spanning panel + summary datasets.

New variables added vs prior version:
  cause_hhi         — Herfindahl index of loss concentration across cause types
                       (high = one peril dominates; low = diverse causes)
  loss_frequency    — fraction of years the county had any indemnity payment
  liability_per_acre — total_liability / insured_acres (coverage intensity)
  indemnity_per_policy — total_indemnity / total_policies_earning (loss severity)

Variables are grouped and axis labels are colour-coded by group.
Spearman (rank) correlations used throughout — robust to right-skewed distributions.
Axes reordered by hierarchical clustering so correlated blocks appear together.

Reads:  data/county_panel.csv, data/county_summary.csv
Writes: graphics/mega_corr_matrix.png
"""
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

PROJ     = '/Users/coltms/Downloads/543_Project'
DATA_DIR = os.path.join(PROJ, 'data')

CAUSES = ['drought', 'heat', 'cold', 'wind_storm', 'precipitation', 'pest', 'disease', 'fire']

# ── Variable groups (for axis colouring) ─────────────────────────────────────

GROUPS = {
    'Loss outcomes':     ('#d62728', ['cv_indemnity', 'std_indem_acre', 'mean_indem_acre',
                                      'indemnity_per_insured_acre', 'loss_frequency']),
    'Crop diversity':    ('#2ca02c', ['mean_shannon', 'mean_simpson', 'mean_evenness',
                                      'mean_dominance', 'mean_richness', 'shannon_stability']),
    'Cause diversity':   ('#17becf', ['cause_hhi', 'n_cause_types']),
    'Exposure / scale':  ('#1f77b4', ['county_land_acres', 'insured_acres',
                                      'mean_insured_acres', 'pct_land_insured', 'mean_pct_insured']),
    'Policy features':   ('#9467bd', ['mean_coverage_level', 'pct_buyup',
                                      'total_policies_earning', 'total_liability',
                                      'liability_per_acre', 'indemnity_per_policy']),
    'Indemnity by cause':('#ff7f0e', ['total_indemnity'] + [f'indemnity_{c}' for c in CAUSES]),
    'Policies by cause': ('#8c564b', [f'policies_{c}' for c in CAUSES]),
    'Event flags':       ('#e377c2', [f'event_{c}' for c in CAUSES]),
}

# flat ordered list and colour lookup
VARS = [v for _, (_, vs) in GROUPS.items() for v in vs]
VAR_COLOR = {v: col for _, (col, vs) in GROUPS.items() for v in vs}

# ── 1. Load county_panel, engineer new variables, aggregate to county level ───

print('Loading county_panel...')
panel = pd.read_csv(os.path.join(DATA_DIR, 'county_panel.csv'))

cause_ind_cols = [f'indemnity_{c}' for c in CAUSES if f'indemnity_{c}' in panel.columns]
panel['total_cause_indemnity'] = panel[cause_ind_cols].sum(axis=1)

# cause HHI per county-year
for c in CAUSES:
    col = f'indemnity_{c}'
    if col in panel.columns:
        panel[f'_share_{c}'] = panel[col] / panel['total_cause_indemnity'].replace(0, np.nan)
panel['cause_hhi'] = sum(
    panel[f'_share_{c}'].fillna(0) ** 2
    for c in CAUSES if f'_share_{c}' in panel.columns
)

panel['liability_per_acre']    = panel['total_liability'] / panel['insured_acres'].replace(0, np.nan)
panel['indemnity_per_policy']  = panel['total_indemnity'] / panel['total_policies_earning'].replace(0, np.nan)

# loss_frequency must be computed before aggregation
loss_freq = (
    panel.groupby(['state_code', 'county_code'])
    .apply(lambda g: (g['total_indemnity'] > 0).mean(), include_groups=False)
    .rename('loss_frequency')
    .reset_index()
)

num_cols = [c for c in panel.columns
            if c not in ('commodity_year', 'state_code', 'county_code',
                         'state_name', 'state_abbreviation', 'county_name')
            and not c.startswith('_share_')]
panel_county = (
    panel.groupby(['state_code', 'county_code'])[num_cols]
    .mean()
    .reset_index()
    .merge(loss_freq, on=['state_code', 'county_code'], how='left')
)
print(f'  {len(panel_county):,} counties from panel ({len(num_cols)} numeric cols)')

# ── 2. Load county_summary ────────────────────────────────────────────────────

print('Loading county_summary...')
summary = pd.read_csv(os.path.join(DATA_DIR, 'county_summary.csv'))
summary_extra = ['state_code', 'county_code', 'mean_shannon', 'mean_simpson', 'mean_evenness',
                 'mean_dominance', 'mean_richness', 'shannon_stability', 'mean_indem_acre',
                 'std_indem_acre', 'mean_insured_acres', 'mean_pct_insured', 'cv_indemnity',
                 'mean_coverage_level', 'pct_buyup']
summary = summary[[c for c in summary_extra if c in summary.columns]]
print(f'  {len(summary):,} counties from summary')

merged = panel_county.merge(summary, on=['state_code', 'county_code'], how='inner')
print(f'  {len(merged):,} counties after join')

# ── 3. Select and clean variables ─────────────────────────────────────────────

present = [v for v in VARS if v in merged.columns]
missing = [v for v in VARS if v not in merged.columns]
if missing:
    print(f'  WARNING — missing from data: {missing}')

df = merged[present].copy().apply(pd.to_numeric, errors='coerce')

for col in present:
    cap = df[col].quantile(0.99)
    df[col] = df[col].clip(upper=cap)

df = df.dropna()
print(f'  {len(df):,} counties used  |  {len(present)} variables')

# ── 4. Spearman correlation ───────────────────────────────────────────────────

print('\nComputing Spearman correlations...')
corr_raw, _ = spearmanr(df.values)
corr_arr    = (corr_raw + corr_raw.T) / 2
np.fill_diagonal(corr_arr, 1.0)
corr_df = pd.DataFrame(corr_arr, index=present, columns=present)

# ── 5. Hierarchical clustering ────────────────────────────────────────────────

dist_arr = 1 - np.abs(corr_arr)
np.fill_diagonal(dist_arr, 0.0)
dist_arr  = np.clip(dist_arr, 0, None)
condensed = squareform(dist_arr, checks=False)
linkage   = hierarchy.linkage(condensed, method='ward')
leaf_order = hierarchy.dendrogram(linkage, no_plot=True)['leaves']
ordered   = [present[i] for i in leaf_order]
corr_ord  = corr_df.loc[ordered, ordered]

# ── 6. Plot ───────────────────────────────────────────────────────────────────

n   = len(ordered)
fig, ax = plt.subplots(figsize=(24, 21))

im = ax.imshow(corr_ord.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
cbar = plt.colorbar(im, ax=ax, shrink=0.55, pad=0.01)
cbar.set_label('Spearman r', fontsize=11)

# tick labels coloured by variable group
ax.set_xticks(range(n))
ax.set_yticks(range(n))
xlabels = ax.set_xticklabels(ordered, rotation=90, fontsize=6.5)
ylabels = ax.set_yticklabels(ordered, fontsize=6.5)
for lbl in xlabels:
    lbl.set_color(VAR_COLOR.get(lbl.get_text(), 'black'))
for lbl in ylabels:
    lbl.set_color(VAR_COLOR.get(lbl.get_text(), 'black'))

# cell annotations
for i in range(n):
    for j in range(n):
        val = corr_ord.values[i, j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=4, color='white' if abs(val) > 0.6 else 'black')

# faint grid
for k in range(n + 1):
    ax.axhline(k - 0.5, color='white', linewidth=0.25)
    ax.axvline(k - 0.5, color='white', linewidth=0.25)

# group legend
patches = [mpatches.Patch(color=col, label=grp) for grp, (col, _) in GROUPS.items()]
ax.legend(handles=patches, loc='upper left', bbox_to_anchor=(1.07, 1),
          fontsize=9, title='Variable group', title_fontsize=10, framealpha=0.9)

ax.set_title(
    f'Spearman Correlation Matrix  —  {n} variables  ({len(df):,} counties)\n'
    'Axes reordered by hierarchical clustering  |  Axis labels coloured by variable group',
    fontsize=12, fontweight='bold', pad=14
)

plt.tight_layout()
out = os.path.join(PROJ, 'graphics', 'mega_corr_matrix.png')
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.show()
print(f'\nSaved → {out}')

# ── 7. Top correlates for key outcomes ───────────────────────────────────────

for target in ['cv_indemnity', 'mean_indem_acre', 'mean_shannon', 'cause_hhi']:
    if target not in corr_df.columns:
        continue
    top = corr_df[target].drop(target).abs().sort_values(ascending=False).head(10)
    print(f'\n── Top 10 correlates of {target} ──')
    print(corr_df[target].loc[top.index].round(3).to_string())
