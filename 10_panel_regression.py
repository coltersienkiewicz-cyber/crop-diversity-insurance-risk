"""Two OLS regressions with correlation matrix and VIF diagnostics.

Regression 1 (county-year level, n ≈ 10k+):
    indemnity_per_acre ~ total_indemnity + insured_acres + n_crops +
                         shannon + simpson + dominance + evenness +
                         county_land_acres + pct_land_insured +
                         [select interaction terms]

Regression 2 (county level, n ≈ 2.6k):
    cv_indemnity ~ mean of all above predictors
    (cv_indemnity = std / mean of indemnity_per_acre across years per county)

Both use a 75/25 train/test split.

Reads:  data/county_year_df.csv, data/county_summary.csv
"""
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

PROJ     = '/Users/coltms/Downloads/543_Project'
DATA_DIR = os.path.join(PROJ, 'data')

# ── Shared helpers ────────────────────────────────────────────────────────────

def vif_table(df, features):
    X = df[features].assign(const=1).dropna()
    return pd.DataFrame({
        'VIF': [variance_inflation_factor(X.values, i) for i in range(len(features))]
    }, index=features).round(1)


def coef_plot(model, title, ax):
    terms = [t for t in model.params.index if t != 'Intercept']
    coefs = model.params[terms]
    ci_lo = model.conf_int()[0][terms]
    ci_hi = model.conf_int()[1][terms]
    pvals = model.pvalues[terms]

    colors = ['#2ca02c' if p < 0.05 else '#aec7e8' for p in pvals]
    y_pos  = range(len(terms))
    ax.barh(list(y_pos), coefs, color=colors, alpha=0.75)
    ax.errorbar(coefs, list(y_pos),
                xerr=[coefs - ci_lo, ci_hi - coefs],
                fmt='none', color='black', linewidth=1, capsize=3)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(terms, fontsize=8)
    ax.set_xlabel('Coefficient (95% CI)', fontsize=9)
    ax.set_title(title + '\n(green = p < 0.05)', fontsize=10)


def eval_split(model, train_df, test_df, target):
    train_pred = model.predict(train_df)
    test_pred  = model.predict(test_df)
    print(f"  Train  R² = {r2_score(train_df[target], train_pred):.3f}  "
          f"RMSE = {np.sqrt(mean_squared_error(train_df[target], train_pred)):.3f}")
    print(f"  Test   R² = {r2_score(test_df[target],  test_pred):.3f}  "
          f"RMSE = {np.sqrt(mean_squared_error(test_df[target],  test_pred)):.3f}")
    gap = r2_score(train_df[target], train_pred) - r2_score(test_df[target], test_pred)
    if gap > 0.05:
        print(f"  ⚠ Train–test gap = {gap:.3f} — possible overfit")
    return test_pred


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Correlation matrix (county-year level)
# ═══════════════════════════════════════════════════════════════════════════════

print('Loading county_year_df...')
cy = pd.read_csv(os.path.join(DATA_DIR, 'county_year_df.csv'))

PRED_COLS = ['total_indemnity', 'insured_acres', 'n_crops', 'shannon',
             'simpson', 'dominance', 'evenness', 'county_land_acres', 'pct_land_insured']
TARGET1   = 'indemnity_per_acre'

for col in PRED_COLS + [TARGET1]:
    cy[col] = pd.to_numeric(cy[col], errors='coerce')

corr_cols = PRED_COLS + [TARGET1]
cy_clean  = cy[corr_cols].replace([np.inf, -np.inf], np.nan).dropna()

# clip extreme indemnity/acre values (top 1%) to reduce outlier distortion
ipa_cap = cy_clean[TARGET1].quantile(0.99)
cy_clean = cy_clean[cy_clean[TARGET1] <= ipa_cap].copy()
print(f'  {len(cy_clean):,} county-year observations (after dropping nulls, top-1% clip)')

corr = cy_clean.corr()

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')
ax.set_xticks(range(len(corr_cols)))
ax.set_yticks(range(len(corr_cols)))
ax.set_xticklabels(corr_cols, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(corr_cols, fontsize=9)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax.text(j, i, f'{corr.values[i, j]:.2f}',
                ha='center', va='center', fontsize=7,
                color='white' if abs(corr.values[i, j]) > 0.5 else 'black')
ax.set_title('Correlation Matrix — County-Year Predictors', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PROJ, 'graphics', 'corr_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()

print('\nTop correlations with indemnity_per_acre:')
print(corr[TARGET1].drop(TARGET1).sort_values(key=abs, ascending=False).round(3).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Regression 1: indemnity_per_acre (county-year)
# ═══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*65)
print('REGRESSION 1: indemnity_per_acre ~ diversity + exposure features')
print('  (county-year level, 75/25 train/test)')
print('='*65)

# NOTE: total_indemnity / insured_acres = indemnity_per_acre by construction,
# so both appearing as predictors will produce near-perfect multicollinearity
# (VIF will flag this clearly). They are included as the user requested.

# Interaction terms — theoretically motivated pairings:
#   shannon × pct_land_insured  : does diversity help more in ag-intensive counties?
#   dominance × insured_acres   : does monoculture scale amplify losses?
#   n_crops × county_land_acres : more crops in larger counties → lower risk?
FORMULA1 = (
    f'{TARGET1} ~ '
    + ' + '.join(PRED_COLS)
    + ' + shannon:pct_land_insured'
    + ' + dominance:insured_acres'
    + ' + n_crops:county_land_acres'
)

train1, test1 = train_test_split(cy_clean, test_size=0.25, random_state=42)
print(f'\nTrain: {len(train1):,}  |  Test: {len(test1):,}')

model1 = smf.ols(FORMULA1, data=train1).fit()
print(model1.summary())

print('\nHoldout performance:')
test1_pred = eval_split(model1, train1, test1, TARGET1)

print('\nVIF (main effects only — interaction VIF not independently meaningful):')
print(vif_table(cy_clean, PRED_COLS).to_string())

# coefficient + predicted-vs-actual plots
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

coef_plot(model1, 'Reg 1: indemnity_per_acre\nCoefficients', axes[0])

axes[1].scatter(test1[TARGET1], test1_pred, alpha=0.25, s=10, color='steelblue')
lo = min(test1[TARGET1].min(), test1_pred.min())
hi = max(test1[TARGET1].max(), test1_pred.max())
axes[1].plot([lo, hi], [lo, hi], 'k--', linewidth=1)
r2_test = r2_score(test1[TARGET1], test1_pred)
axes[1].set_xlabel('Actual indemnity per acre ($)', fontsize=10)
axes[1].set_ylabel('Predicted', fontsize=10)
axes[1].set_title(f'Reg 1 — Test Set Predicted vs Actual\n(R² = {r2_test:.3f})', fontsize=10)

plt.suptitle('Regression 1: indemnity_per_acre', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PROJ, 'graphics', 'reg1_indemnity_per_acre.png'),
            dpi=150, bbox_inches='tight')
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Regression 2: cv_indemnity / volatility (county level)
# ═══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*65)
print('REGRESSION 2: cv_indemnity ~ diversity + exposure features')
print('  (county level — volatility across years, 75/25 train/test)')
print('='*65)

cs = pd.read_csv(os.path.join(DATA_DIR, 'county_summary.csv'))

# county_summary uses mean_ prefixed columns; map to match user's requested names
rename_map = {
    'mean_richness':      'n_crops',
    'mean_shannon':       'shannon',
    'mean_simpson':       'simpson',
    'mean_dominance':     'dominance',
    'mean_evenness':      'evenness',
    'mean_insured_acres': 'insured_acres',
    'mean_pct_insured':   'pct_land_insured',
    'mean_indem_acre':    'mean_indem_acre',
    'std_indem_acre':     'std_indem_acre',
}
cs = cs.rename(columns=rename_map)

# total_indemnity not in summary — proxy with mean_indem_acre × insured_acres
if 'total_indemnity' not in cs.columns:
    cs['total_indemnity'] = cs['mean_indem_acre'] * cs['insured_acres']

TARGET2 = 'cv_indemnity'
for col in PRED_COLS + [TARGET2]:
    if col in cs.columns:
        cs[col] = pd.to_numeric(cs[col], errors='coerce')

cs_clean = (
    cs[PRED_COLS + [TARGET2]]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)
# clip top 1% of CV
cv_cap = cs_clean[TARGET2].quantile(0.99)
cs_clean = cs_clean[cs_clean[TARGET2] <= cv_cap].copy()
print(f'\n{len(cs_clean):,} counties (after dropping nulls, top-1% clip)')

FORMULA2 = f'{TARGET2} ~ ' + ' + '.join(PRED_COLS)

train2, test2 = train_test_split(cs_clean, test_size=0.25, random_state=42)
print(f'Train: {len(train2):,}  |  Test: {len(test2):,}')

model2 = smf.ols(FORMULA2, data=train2).fit()
print(model2.summary())

print('\nHoldout performance:')
test2_pred = eval_split(model2, train2, test2, TARGET2)

print('\nVIF:')
print(vif_table(cs_clean, PRED_COLS).to_string())

# coefficient + predicted-vs-actual plots
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

coef_plot(model2, 'Reg 2: cv_indemnity (volatility)\nCoefficients', axes[0])

axes[1].scatter(test2[TARGET2], test2_pred, alpha=0.35, s=15, color='firebrick')
lo = min(test2[TARGET2].min(), test2_pred.min())
hi = max(test2[TARGET2].max(), test2_pred.max())
axes[1].plot([lo, hi], [lo, hi], 'k--', linewidth=1)
r2_test2 = r2_score(test2[TARGET2], test2_pred)
axes[1].set_xlabel('Actual CV (volatility)', fontsize=10)
axes[1].set_ylabel('Predicted', fontsize=10)
axes[1].set_title(f'Reg 2 — Test Set Predicted vs Actual\n(R² = {r2_test2:.3f})', fontsize=10)

plt.suptitle('Regression 2: claim volatility (cv_indemnity)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PROJ, 'graphics', 'reg2_cv_indemnity.png'),
            dpi=150, bbox_inches='tight')
plt.show()

# ── Side-by-side coefficient comparison ───────────────────────────────────────

shared = [t for t in model1.params.index
          if t in model2.params.index and t != 'Intercept']

if shared:
    fig, ax = plt.subplots(figsize=(10, 6))
    x   = np.arange(len(shared))
    w   = 0.35
    b1  = ax.bar(x - w/2, model1.params[shared], w, label='Reg 1: indemnity/acre',
                 color='steelblue', alpha=0.75)
    b2  = ax.bar(x + w/2, model2.params[shared], w, label='Reg 2: volatility (CV)',
                 color='firebrick', alpha=0.75)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(shared, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Coefficient', fontsize=10)
    ax.set_title('Shared Predictors: Effect on Loss Intensity vs Volatility',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PROJ, 'graphics', 'reg_coef_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
