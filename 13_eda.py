"""Step 4 — Exploratory Analysis: diversity vs. loss ratio.

Produces five figures (PNG) in eda_plots/:
  1. scatter_diversity_lossratio.png   — Shannon diversity vs. loss ratio, colored by state
  2. boxplot_diversity_quartile.png    — Loss ratio distributions by diversity quartile
  3. heatmap_feature_correlations.png  — Feature correlation matrix
  4. map_diversity_lossratio.png       — Side-by-side choropleth: avg diversity vs. avg loss ratio
  5. timeseries_diversity_risk.png     — County-median diversity and loss ratio by year

Reads:  data/feature_matrix.csv,
        Farm Resource Regions/tl_2025_us_county/tl_2025_us_county.shp
Writes: eda_plots/*.png
"""
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

PROJ    = '/Users/coltms/Downloads/543_Project'
DATA    = os.path.join(PROJ, 'data', 'feature_matrix.csv')
SHP     = os.path.join(PROJ, 'Farm Resource Regions', 'tl_2025_us_county', 'tl_2025_us_county.shp')
OUT_DIR = os.path.join(PROJ, 'eda_plots')
os.makedirs(OUT_DIR, exist_ok=True)

EXCLUDE_TERRITORIES = ['02', '15', '60', '66', '69', '72', '78']

sns.set_theme(style='whitegrid', font_scale=1.1)
PALETTE = 'tab20'

# ── Load ──────────────────────────────────────────────────────────────────────

fm = pd.read_csv(DATA, dtype={'state_code': str, 'county_code': str, 'commodity_year': str})
fm['state_code']  = fm['state_code'].str.zfill(2)
fm['county_code'] = fm['county_code'].str.zfill(3)

# working subset: rows with both diversity and loss ratio
clean = fm.dropna(subset=['shannon_diversity', 'loss_ratio']).copy()
# cap loss_ratio at 99th percentile for plotting (extreme audit outliers)
lr_cap = clean['loss_ratio'].quantile(0.99)
clean['loss_ratio_plot'] = clean['loss_ratio'].clip(upper=lr_cap)

print(f'Rows with shannon + loss_ratio: {len(clean):,} / {len(fm):,}')

# ── 1. Scatter: Shannon diversity vs. loss ratio ──────────────────────────────

print('Plot 1: scatter...')
top_states = clean['state_abbreviation'].value_counts().head(20).index
plot1 = clean[clean['state_abbreviation'].isin(top_states)].copy()

fig, ax = plt.subplots(figsize=(11, 7))
states = sorted(plot1['state_abbreviation'].unique())
cmap   = plt.colormaps[PALETTE]
colors = {s: cmap(i / len(states)) for i, s in enumerate(states)}

for state, grp in plot1.groupby('state_abbreviation'):
    ax.scatter(
        grp['shannon_diversity'], grp['loss_ratio_plot'],
        c=[colors[state]], label=state, alpha=0.45, s=18, linewidths=0,
    )

# OLS trend line across all points
m, b, r, p, _ = stats.linregress(plot1['shannon_diversity'], plot1['loss_ratio_plot'])
x_line = np.linspace(plot1['shannon_diversity'].min(), plot1['shannon_diversity'].max(), 200)
ax.plot(x_line, m * x_line + b, color='black', linewidth=1.8, linestyle='--',
        label=f'OLS  r={r:.2f}  p={p:.3f}')

ax.set_xlabel("Shannon Diversity (H')", fontsize=12)
ax.set_ylabel(f'Loss Ratio (capped at {lr_cap:.2f})', fontsize=12)
ax.set_title("Crop Diversity vs. Loss Ratio by County-Year\n(top 20 states by observation count)",
             fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=7, ncol=3, framealpha=0.8,
          markerscale=1.5, title='State')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'scatter_diversity_lossratio.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  saved')

# ── 2. Box plot: loss ratio by diversity quartile ─────────────────────────────

print('Plot 2: boxplot...')
clean['diversity_quartile'] = pd.qcut(
    clean['shannon_diversity'], q=4,
    labels=['Q1\n(least diverse)', 'Q2', 'Q3', 'Q4\n(most diverse)'],
)

fig, ax = plt.subplots(figsize=(9, 6))
quartile_palette = sns.color_palette('YlOrRd_r', 4)
sns.boxplot(
    data=clean, x='diversity_quartile', y='loss_ratio_plot',
    hue='diversity_quartile', palette=quartile_palette, width=0.55,
    fliersize=2.5, linewidth=1.2, ax=ax, legend=False,
)
# median labels
medians = clean.groupby('diversity_quartile', observed=True)['loss_ratio_plot'].median()
for i, (q, med) in enumerate(medians.items()):
    ax.text(i, med + 0.01, f'{med:.3f}', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='black')

ax.set_xlabel('Shannon Diversity Quartile', fontsize=12)
ax.set_ylabel(f'Loss Ratio (capped at {lr_cap:.2f})', fontsize=12)
ax.set_title('Loss Ratio Distributions by Crop Diversity Quartile\n'
             'Q1 = least diverse counties,  Q4 = most diverse',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'boxplot_diversity_quartile.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  saved')

# ── 3. Correlation heatmap ────────────────────────────────────────────────────

print('Plot 3: correlation heatmap...')
FEAT_COLS = [
    'shannon_diversity', 'num_unique_crops', 'top_crop_share', 'evenness',
    'total_liability', 'avg_coverage_level',
    'num_loss_causes', 'weather_loss_share',
    'loss_ratio',
]
corr_data = clean[FEAT_COLS].dropna()
corr = corr_data.corr()

labels = {
    'shannon_diversity':  "Shannon H'",
    'num_unique_crops':   '# Crops',
    'top_crop_share':     'Top Crop Share',
    'evenness':           'Evenness (J)',
    'total_liability':    'Total Liability',
    'avg_coverage_level': 'Avg Coverage',
    'num_loss_causes':    '# Loss Causes',
    'weather_loss_share': 'Weather Share',
    'loss_ratio':         'Loss Ratio *',
}
corr.index   = [labels.get(c, c) for c in corr.index]
corr.columns = [labels.get(c, c) for c in corr.columns]

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
    vmin=-1, vmax=1, linewidths=0.4, ax=ax, annot_kws={'size': 9},
    cbar_kws={'shrink': 0.75, 'label': 'Pearson r'},
)
ax.set_title('Feature Correlation Matrix  (* = target)',
             fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'heatmap_feature_correlations.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  saved')

# ── 4. Side-by-side choropleth: avg diversity vs. avg loss ratio ──────────────

print('Plot 4: choropleth map...')
county_avg = (
    clean
    .groupby(['state_code', 'county_code'])
    .agg(avg_shannon=('shannon_diversity', 'mean'),
         avg_loss_ratio=('loss_ratio', 'mean'))
    .reset_index()
)

gdf_base = gpd.read_file(SHP)
conus    = gdf_base[~gdf_base['STATEFP'].isin(EXCLUDE_TERRITORIES)].copy()
gdf      = conus.merge(county_avg,
                       left_on=['STATEFP', 'COUNTYFP'],
                       right_on=['state_code', 'county_code'],
                       how='left')

fig, axes = plt.subplots(1, 2, figsize=(20, 7))

# panel A: diversity
gdf.plot(column='avg_shannon', ax=axes[0], cmap='YlGn',
         linewidth=0.08, edgecolor='white', legend=True,
         missing_kwds={'color': 'lightgrey'},
         legend_kwds={'label': "Mean Shannon H'", 'shrink': 0.6, 'orientation': 'horizontal'})
axes[0].set_axis_off()
axes[0].set_title("Avg Crop Diversity (Shannon H')", fontsize=13, fontweight='bold')

# panel B: loss ratio
gdf.plot(column='avg_loss_ratio', ax=axes[1], cmap='RdYlGn_r',
         linewidth=0.08, edgecolor='white', legend=True,
         missing_kwds={'color': 'lightgrey'},
         legend_kwds={'label': 'Mean Loss Ratio', 'shrink': 0.6, 'orientation': 'horizontal'})
axes[1].set_axis_off()
axes[1].set_title('Avg Loss Ratio', fontsize=13, fontweight='bold')

fig.suptitle('Crop Diversity vs. Insurance Loss Ratio by County (1999–2004 avg)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'map_diversity_lossratio.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  saved')

# ── 5. Time series: county-median diversity and loss ratio by year ────────────

print('Plot 5: time series...')
ts = (
    clean
    .groupby('commodity_year')
    .agg(
        med_shannon   = ('shannon_diversity', 'median'),
        med_loss_ratio= ('loss_ratio',        'median'),
        p25_shannon   = ('shannon_diversity',  lambda x: x.quantile(0.25)),
        p75_shannon   = ('shannon_diversity',  lambda x: x.quantile(0.75)),
        p25_loss      = ('loss_ratio',         lambda x: x.quantile(0.25)),
        p75_loss      = ('loss_ratio',         lambda x: x.quantile(0.75)),
    )
    .reset_index()
    .sort_values('commodity_year')
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

years = ts['commodity_year'].astype(str)
x     = range(len(years))

ax1.plot(x, ts['med_shannon'], marker='o', color='#2ca02c', linewidth=2, label='Median Shannon H\'')
ax1.fill_between(x, ts['p25_shannon'], ts['p75_shannon'], color='#2ca02c', alpha=0.18, label='IQR')
ax1.set_ylabel("Shannon H' (diversity)", fontsize=11)
ax1.set_title('County-Level Crop Diversity Over Time', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

ax2.plot(x, ts['med_loss_ratio'], marker='s', color='#d62728', linewidth=2, label='Median Loss Ratio')
ax2.fill_between(x, ts['p25_loss'], ts['p75_loss'], color='#d62728', alpha=0.18, label='IQR')
ax2.set_ylabel('Loss Ratio', fontsize=11)
ax2.set_title('County-Level Loss Ratio Over Time', fontsize=12, fontweight='bold')
ax2.set_xticks(list(x))
ax2.set_xticklabels(list(years))
ax2.legend(fontsize=9)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

fig.suptitle('Do Diversity and Risk Move Together Across Years?',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'timeseries_diversity_risk.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  saved')

print(f'\nAll plots saved to {OUT_DIR}')

# ── Quick diagnostic numbers for the narrative ────────────────────────────────

print('\n── Key numbers ──────────────────────────────────────────────────────────')
_slope, _intercept, _r, _p, _ = stats.linregress(clean['shannon_diversity'], clean['loss_ratio_plot'])
print(f"Shannon vs loss_ratio (capped):  r={_r:.3f}  slope={_slope:.4f}  p={_p:.4f}")

q_medians = clean.groupby('diversity_quartile', observed=True)['loss_ratio'].median()
print(f"\nMedian loss ratio by quartile:\n{q_medians.to_string()}")

print(f"\nCorrelation of shannon_diversity with loss_ratio: "
      f"{clean[['shannon_diversity','loss_ratio']].corr().iloc[0,1]:.3f}")
