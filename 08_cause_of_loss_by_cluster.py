"""Analyse how diversity clusters weather different types of loss events.

For each cause-of-loss group (drought, wind/storm, cold, precipitation,
pest, disease, fire, heat) and each diversity cluster, computes:
  - mean indemnity per acre  (loss intensity)
  - CV of indemnity per acre (loss volatility across years)

Reads:  data/colsommonth.csv, data/county_summary.csv (must include
        diversity_category column written by 04_cluster_analysis.py)
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJ     = '/Users/coltms/Downloads/543_Project'
DATA_DIR = os.path.join(PROJ, 'data')

# ── Cause-of-loss code → group mapping ───────────────────────────────────────
# codes from RMA cause-of-loss key; grouped into interpretable weather categories

CAUSE_GROUPS = {
    '11': 'Drought', '13': 'Drought', '14': 'Drought',
    '12': 'Heat',    '22': 'Heat',    '45': 'Heat',
    '41': 'Cold',    '42': 'Cold',    '43': 'Cold',    '44': 'Cold', '74': 'Cold',
    '21': 'Wind/Storm', '61': 'Wind/Storm', '62': 'Wind/Storm',
    '63': 'Wind/Storm', '64': 'Wind/Storm',
    '31': 'Precipitation', '51': 'Precipitation', '65': 'Precipitation',
    '67': 'Precipitation', '92': 'Precipitation',
    '71': 'Pest',    '73': 'Pest',    '93': 'Pest',
    '80': 'Disease', '81': 'Disease', '82': 'Disease', '76': 'Disease',
    '91': 'Fire',    '95': 'Fire',
}

# ── Load colsommonth ──────────────────────────────────────────────────────────

print('Loading colsommonth...')
col_path = os.path.join(DATA_DIR, 'colsommonth.csv')
if not os.path.exists(col_path):
    raise FileNotFoundError(
        f'{col_path} not found — run 01_load_data.py first.'
    )

col_df = pd.read_csv(
    col_path,
    usecols=['commodity_year', 'state_code', 'county_code',
             'cause_of_loss_code', 'net_endorsed_acres', 'indemnity_amount'],
    dtype=str,
)
col_df['cause_of_loss_code'] = col_df['cause_of_loss_code'].str.strip().str.zfill(2)
col_df['state_code']         = col_df['state_code'].str.strip().str.zfill(2)
col_df['county_code']        = col_df['county_code'].str.strip().str.zfill(3)
col_df['net_endorsed_acres'] = pd.to_numeric(col_df['net_endorsed_acres'], errors='coerce')
col_df['indemnity_amount']   = pd.to_numeric(col_df['indemnity_amount'],   errors='coerce')

# keep only known cause groups
col_df['cause_group'] = col_df['cause_of_loss_code'].map(CAUSE_GROUPS)
col_df = col_df.dropna(subset=['cause_group'])
print(f'  {len(col_df):,} rows across {col_df["cause_group"].nunique()} cause groups')

# ── Load cluster assignments (re-cluster if 04 hasn't been run yet) ──────────

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

county_summary = pd.read_csv(os.path.join(DATA_DIR, 'county_summary.csv'))

if 'diversity_category' not in county_summary.columns:
    print('  diversity_category not found — running K-Means now...')
    CLUSTER_FEATURES = ['mean_shannon', 'mean_evenness', 'mean_dominance',
                        'mean_richness', 'shannon_stability', 'mean_acres']
    for col in CLUSTER_FEATURES:
        county_summary[col] = pd.to_numeric(county_summary[col], errors='coerce')

    feat_df  = county_summary[CLUSTER_FEATURES].dropna()
    X_scaled = StandardScaler().fit_transform(feat_df.values)

    # pick best k by silhouette
    best_k, best_score = 3, -1
    for k in range(2, 9):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
        s = silhouette_score(X_scaled, labels)
        if s > best_score:
            best_k, best_score = k, s

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    county_summary.loc[feat_df.index, 'cluster'] = km.fit_predict(X_scaled)

    order = (county_summary.groupby('cluster')['mean_shannon']
             .mean().sort_values().index.tolist())
    label_map = {c: f'Cluster {i+1}' for i, c in enumerate(order)}
    county_summary['diversity_category'] = county_summary['cluster'].map(label_map)
    print(f'  clustered into k={best_k} groups')

clusters = (
    county_summary[['state_code', 'county_code', 'diversity_category']]
    .dropna(subset=['diversity_category'])
    .astype(str)
)
clusters['state_code']  = clusters['state_code'].str.strip().str.zfill(2)
clusters['county_code'] = clusters['county_code'].str.strip().str.zfill(3)
print(f'  {len(clusters):,} counties with cluster assignments')

# ── Aggregate to county-year-cause_group ─────────────────────────────────────

KEY = ['commodity_year', 'state_code', 'county_code', 'cause_group']
county_cause = (
    col_df
    .groupby(KEY, as_index=False)
    .agg(total_indemnity=('indemnity_amount',   'sum'),
         total_acres    =('net_endorsed_acres',  'sum'))
)
county_cause = county_cause[county_cause['total_acres'] > 0].copy()
county_cause['indemnity_per_acre'] = (
    county_cause['total_indemnity'] / county_cause['total_acres']
)

# join cluster assignments
county_cause = county_cause.merge(
    clusters, on=['state_code', 'county_code'], how='inner'
)
print(f'  {len(county_cause):,} county-year-cause_group observations after cluster join')

# ── Per-county stats by cause group ──────────────────────────────────────────
# For each county × cause group: mean and CV across years

county_stats = (
    county_cause
    .groupby(['state_code', 'county_code', 'diversity_category', 'cause_group'])
    ['indemnity_per_acre']
    .agg(
        mean_ipa=('mean'),
        cv_ipa  =(lambda x: x.std() / x.mean() if x.mean() > 0 and len(x) > 1 else np.nan),
    )
    .reset_index()
)

# ── Cluster × cause_group summary ────────────────────────────────────────────

cluster_cause = (
    county_stats
    .groupby(['diversity_category', 'cause_group'])
    .agg(
        mean_intensity  =('mean_ipa', 'median'),
        mean_volatility =('cv_ipa',   'median'),
        n_counties      =('mean_ipa', 'count'),
    )
    .reset_index()
)

# ── Pivot for heatmaps ────────────────────────────────────────────────────────

cat_order   = sorted(cluster_cause['diversity_category'].unique())
cause_order = ['Drought', 'Heat', 'Cold', 'Wind/Storm', 'Precipitation',
               'Pest', 'Disease', 'Fire']
cause_order = [c for c in cause_order if c in cluster_cause['cause_group'].unique()]

intensity_pivot = (
    cluster_cause
    .pivot(index='cause_group', columns='diversity_category', values='mean_intensity')
    .reindex(index=cause_order, columns=cat_order)
)
volatility_pivot = (
    cluster_cause
    .pivot(index='cause_group', columns='diversity_category', values='mean_volatility')
    .reindex(index=cause_order, columns=cat_order)
)

print('\nMedian indemnity/acre by (cause group × cluster):')
print(intensity_pivot.round(1).to_string())
print('\nMedian CV by (cause group × cluster):')
print(volatility_pivot.round(3).to_string())

# ── Heatmaps ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, pivot, title, cmap in [
    (axes[0], intensity_pivot,  'Median Loss Intensity\n(indemnity / acre, $)', 'YlOrRd'),
    (axes[1], volatility_pivot, 'Median Claim Volatility\n(CV across years)',   'YlOrBr'),
]:
    im = ax.imshow(pivot.values.astype(float), cmap=cmap, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(cat_order)))
    ax.set_xticklabels(cat_order, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(cause_order)))
    ax.set_yticklabels(cause_order, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')

    # annotate cells
    for i in range(len(cause_order)):
        for j in range(len(cat_order)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        fontsize=8, color='black')

plt.suptitle('Crop Diversity Clusters vs. Loss Type: Intensity and Volatility',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PROJ, 'graphics', 'cause_of_loss_heatmaps.png'),
            dpi=150, bbox_inches='tight')
plt.show()

# ── Bar charts: top 3 most damaging cause groups ─────────────────────────────

top_causes = (
    cluster_cause.groupby('cause_group')['mean_intensity']
    .median().nlargest(4).index.tolist()
)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

palette = plt.cm.tab10.colors

for ax, cause in zip(axes, top_causes):
    sub = cluster_cause[cluster_cause['cause_group'] == cause].sort_values('diversity_category')
    bars = ax.bar(sub['diversity_category'], sub['mean_intensity'],
                  color=palette[:len(sub)], alpha=0.75, edgecolor='white')
    ax2 = ax.twinx()
    ax2.plot(range(len(sub)), sub['mean_volatility'], 'k--o',
             linewidth=1.5, markersize=5, label='CV (volatility)')
    ax2.set_ylabel('Median CV', fontsize=9, color='black')
    ax2.tick_params(axis='y', labelsize=8)

    ax.set_title(f'{cause} losses', fontsize=11, fontweight='bold')
    ax.set_ylabel('Median indemnity / acre ($)', fontsize=9)
    ax.tick_params(axis='x', rotation=25, labelsize=8)
    ax2.legend(loc='upper right', fontsize=8)

plt.suptitle('Loss Intensity (bars) and Volatility (line) by Cluster\nTop 4 Cause Groups',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PROJ, 'graphics', 'cause_of_loss_bar_charts.png'),
            dpi=150, bbox_inches='tight')
plt.show()

# ── Statistical test: does cluster predict CV within each cause group? ────────
# Kruskal-Wallis H-test (non-parametric ANOVA) — tests whether any cluster
# has significantly different volatility than the others

print('\n' + '='*60)
print('Kruskal-Wallis test: does cluster predict volatility per cause?')
print('='*60)
for cause in cause_order:
    groups = [
        county_stats.loc[
            (county_stats['cause_group'] == cause) &
            (county_stats['diversity_category'] == cat), 'cv_ipa'
        ].dropna().values
        for cat in cat_order
    ]
    groups = [g for g in groups if len(g) >= 3]
    if len(groups) < 2:
        continue
    h, p = stats.kruskal(*groups)
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    print(f'  {cause:16s}  H={h:6.2f}  p={p:.4f}  {sig}')
