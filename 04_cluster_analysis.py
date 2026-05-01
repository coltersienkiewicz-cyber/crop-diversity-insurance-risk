"""K-Means clustering across all diversity metrics, volatility analysis, and Random Forest.

Reads:  data/county_summary.csv
Output: prints stats, shows plots
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJ     = '/Users/coltms/Downloads/543_Project'
DATA_DIR = os.path.join(PROJ, 'data')

county_summary = pd.read_csv(os.path.join(DATA_DIR, 'county_summary.csv'))
numeric_cols = [
    'mean_shannon', 'mean_simpson', 'mean_evenness', 'mean_dominance',
    'mean_richness', 'shannon_stability', 'mean_indem_acre', 'std_indem_acre',
    'mean_acres', 'cv_indemnity', 'mean_coverage_level', 'pct_buyup',
]
for col in numeric_cols:
    if col in county_summary.columns:
        county_summary[col] = pd.to_numeric(county_summary[col], errors='coerce')

# ── Features used for clustering ──────────────────────────────────────────────
# Include all diversity dimensions; standardise so scale doesn't dominate

CLUSTER_FEATURES = [
    'mean_shannon',       # combined richness + evenness
    'mean_evenness',      # Pielou's J: how uniformly distributed
    'mean_dominance',     # Berger-Parker: share of top crop
    'mean_richness',      # raw species count
    'shannon_stability',  # how much crop mix shifts year to year
    'mean_acres',         # county exposure size
]

cluster_df = county_summary[CLUSTER_FEATURES].dropna()
valid_idx   = cluster_df.index
X_raw       = cluster_df.values
X_scaled    = StandardScaler().fit_transform(X_raw)

# ── Optimal k: elbow + silhouette ─────────────────────────────────────────────

k_range   = range(2, 10)
inertias  = []
silhouettes = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(k_range), inertias, 'o-', color='steelblue')
axes[0].set_xlabel('Number of clusters (k)')
axes[0].set_ylabel('Inertia (within-cluster SSE)')
axes[0].set_title('Elbow Method')
axes[0].grid(alpha=0.3)

axes[1].plot(list(k_range), silhouettes, 'o-', color='firebrick')
axes[1].set_xlabel('Number of clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score (higher = better separation)')
best_k = list(k_range)[int(np.argmax(silhouettes))]
axes[1].axvline(best_k, color='firebrick', linestyle='--', alpha=0.6,
                label=f'best k={best_k}')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('Choosing the Number of Clusters', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print(f'\nBest k by silhouette: {best_k}')
print('Silhouette scores:', {k: round(s, 3) for k, s in zip(k_range, silhouettes)})

# ── Final clustering at best_k ────────────────────────────────────────────────

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
county_summary.loc[valid_idx, 'cluster'] = labels

# label clusters by mean Shannon (low → high diversity)
order = (county_summary.groupby('cluster')['mean_shannon']
         .mean().sort_values().index.tolist())

palette = ['#d62728', '#ff7f0e', '#bcbd22', '#2ca02c',
           '#17becf', '#9467bd', '#8c564b', '#e377c2']
label_map  = {c: f'Cluster {i+1}' for i, c in enumerate(order)}
color_map  = {f'Cluster {i+1}': palette[i] for i in range(best_k)}
county_summary['diversity_category'] = county_summary['cluster'].map(label_map)

print('\nCluster sizes:')
print(county_summary['diversity_category'].value_counts().sort_index())

# ── Cluster profiles ──────────────────────────────────────────────────────────

profile_cols = CLUSTER_FEATURES + ['cv_indemnity', 'mean_indem_acre']
profile = (county_summary.groupby('diversity_category')[profile_cols]
           .median().round(3))
print('\nCluster medians:')
print(profile.to_string())

# ── PCA scatter (2D projection of all clustering features) ────────────────────

pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
county_summary.loc[valid_idx, 'pca1'] = X_pca[:, 0]
county_summary.loc[valid_idx, 'pca2'] = X_pca[:, 1]

fig, ax = plt.subplots(figsize=(9, 7))
for cat in sorted(county_summary['diversity_category'].dropna().unique()):
    grp = county_summary[county_summary['diversity_category'] == cat]
    ax.scatter(grp['pca1'], grp['pca2'],
               label=cat, alpha=0.4, s=18, color=color_map[cat])

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=11)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=11)
ax.set_title(f'K-Means Clusters (k={best_k}) — PCA Projection of Diversity Features', fontsize=12)
ax.legend(markerscale=2)
plt.tight_layout()
plt.show()

# ── Volatility boxplots ───────────────────────────────────────────────────────

cat_order   = sorted(county_summary['diversity_category'].dropna().unique())
plot_colors = [color_map[c] for c in cat_order]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

data_mean = [county_summary[county_summary['diversity_category'] == c]['mean_indem_acre'].dropna().values
             for c in cat_order]
bp1 = axes[0].boxplot(data_mean, patch_artist=True, labels=cat_order)
for patch, color in zip(bp1['boxes'], plot_colors):
    patch.set_facecolor(color); patch.set_alpha(0.6)
axes[0].set_title('Mean Indemnity per Acre by Cluster', fontsize=11)
axes[0].set_ylabel('Mean Indemnity / Acre ($)')
axes[0].tick_params(axis='x', rotation=20)

data_cv = [county_summary[county_summary['diversity_category'] == c]['cv_indemnity'].dropna().values
           for c in cat_order]
bp2 = axes[1].boxplot(data_cv, patch_artist=True, labels=cat_order)
for patch, color in zip(bp2['boxes'], plot_colors):
    patch.set_facecolor(color); patch.set_alpha(0.6)
axes[1].set_title('Claim Volatility (CV) by Cluster', fontsize=11)
axes[1].set_ylabel('Coefficient of Variation (std / mean)')
axes[1].tick_params(axis='x', rotation=20)

plt.suptitle('Diversity Clusters vs. Loss Level and Volatility', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# ── Random Forest ─────────────────────────────────────────────────────────────

RF_FEATURES = [
    'mean_shannon', 'mean_evenness', 'mean_dominance', 'mean_richness',
    'shannon_stability', 'mean_coverage_level', 'pct_buyup', 'mean_acres',
    'diversity_category',
]
rf_df = county_summary[RF_FEATURES + ['mean_indem_acre']].dropna().copy()

le = LabelEncoder()
rf_df['diversity_category'] = le.fit_transform(rf_df['diversity_category'])

X = rf_df[RF_FEATURES]
y = rf_df['mean_indem_acre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print(f'\nRandom Forest test R²: {r2_score(y_test, rf.predict(X_test)):.3f}')

importance = pd.Series(rf.feature_importances_, index=RF_FEATURES).sort_values()

diversity_feats = {'mean_shannon', 'mean_evenness', 'mean_dominance',
                   'mean_richness', 'shannon_stability', 'diversity_category'}
bar_colors = ['#2ca02c' if f in diversity_feats else '#1f77b4' for f in importance.index]

fig, ax = plt.subplots(figsize=(9, 5))
importance.plot.barh(ax=ax, color=bar_colors)
ax.set_xlabel('Feature Importance (Mean Decrease in Impurity)', fontsize=11)
ax.set_title('Random Forest: What Predicts Indemnity per Acre?', fontsize=12)
ax.legend(handles=[Patch(color='#2ca02c', label='Diversity features'),
                   Patch(color='#1f77b4', label='Policy type / exposure')],
          loc='lower right')
ax.axvline(0, color='k', linewidth=0.5)
plt.tight_layout()
plt.show()

# ── Save cluster assignments back to county_summary.csv ───────────────────────
# (required by 08_cause_of_loss_by_cluster.py)

county_summary.to_csv(os.path.join(DATA_DIR, 'county_summary.csv'), index=False)
print(f'\nSaved clustered county_summary (with diversity_category) to {DATA_DIR}')
