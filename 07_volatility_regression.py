"""Multiple regression of crop diversity metrics on claim volatility (cv_indemnity).

Train/test split: 75% / 25%
Reports: OLS summary, VIF, train vs test R², RMSE, predicted vs actual plot.

Reads:  data/county_summary.csv
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

PROJ     = '/Users/coltms/Downloads/543_Project'
DATA_DIR = os.path.join(PROJ, 'data')

# ── Load data ─────────────────────────────────────────────────────────────────

county_summary = pd.read_csv(os.path.join(DATA_DIR, 'county_summary.csv'))

FEATURES = [
    'mean_richness',      # number of distinct crops
    'mean_evenness',      # Pielou's J — how uniformly distributed
    'mean_dominance',     # Berger-Parker — share of top crop
    'mean_shannon',       # H' — combined richness + evenness
    'shannon_stability',  # std(H') across years — temporal consistency
]
TARGET = 'cv_indemnity'

for col in FEATURES + [TARGET]:
    county_summary[col] = pd.to_numeric(county_summary[col], errors='coerce')

reg_df = county_summary[FEATURES + [TARGET]].dropna()
reg_df = reg_df[np.isfinite(reg_df[TARGET]) & (reg_df[TARGET] < reg_df[TARGET].quantile(0.99))]
print(f'Observations after cleaning: {len(reg_df):,}')

# ── Variance Inflation Factors ────────────────────────────────────────────────
# Checks multicollinearity before fitting; VIF > 5 is worth noting

X_mat = reg_df[FEATURES].assign(const=1)
vif = pd.DataFrame({
    'feature': FEATURES,
    'VIF': [variance_inflation_factor(X_mat.values, i) for i in range(len(FEATURES))]
}).set_index('feature').round(2)
print('\nVariance Inflation Factors (VIF > 5 = multicollinearity concern):')
print(vif.to_string())

# ── Train / test split (75 / 25) ──────────────────────────────────────────────

train_df, test_df = train_test_split(reg_df, test_size=0.25, random_state=42)
print(f'\nTrain: {len(train_df):,}  |  Test: {len(test_df):,}')

# ── OLS on training set ───────────────────────────────────────────────────────

formula = f'{TARGET} ~ ' + ' + '.join(FEATURES)
model = smf.ols(formula, data=train_df).fit()
print('\n' + '='*60)
print('OLS REGRESSION SUMMARY (trained on 75% of data)')
print('='*60)
print(model.summary())

# ── Evaluate on held-out test set ─────────────────────────────────────────────

train_pred = model.predict(train_df)
test_pred  = model.predict(test_df)

train_r2   = r2_score(train_df[TARGET], train_pred)
test_r2    = r2_score(test_df[TARGET],  test_pred)
train_rmse = np.sqrt(mean_squared_error(train_df[TARGET], train_pred))
test_rmse  = np.sqrt(mean_squared_error(test_df[TARGET],  test_pred))

print('\n' + '='*60)
print('HOLDOUT PERFORMANCE')
print('='*60)
print(f'  Train R²:   {train_r2:.3f}   RMSE: {train_rmse:.3f}')
print(f'  Test  R²:   {test_r2:.3f}   RMSE: {test_rmse:.3f}')
gap = train_r2 - test_r2
print(f'  Train–Test gap: {gap:+.3f}{"  (possible overfit)" if gap > 0.05 else ""}')

# coefficient table with significance stars
coef_df = pd.DataFrame({
    'coef':    model.params,
    'std_err': model.bse,
    't':       model.tvalues,
    'p':       model.pvalues,
}).drop('Intercept')
coef_df['sig'] = coef_df['p'].apply(
    lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
)
print('\nCoefficients (target = claim volatility CV):')
print(coef_df[['coef', 'std_err', 't', 'p', 'sig']].round(4).to_string())

# ── Plots ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A — predicted vs actual on test set
axes[0].scatter(test_df[TARGET], test_pred, alpha=0.3, s=15, color='steelblue')
lo = min(test_df[TARGET].min(), test_pred.min())
hi = max(test_df[TARGET].max(), test_pred.max())
axes[0].plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='Perfect fit')
axes[0].set_xlabel('Actual CV (claim volatility)', fontsize=11)
axes[0].set_ylabel('Predicted CV', fontsize=11)
axes[0].set_title(f'Test Set: Predicted vs Actual\n(R² = {test_r2:.3f}, RMSE = {test_rmse:.3f})',
                  fontsize=11)
axes[0].legend(fontsize=9)

# Panel B — coefficient plot (effect on volatility)
sig_colors = [
    '#2ca02c' if p < 0.05 else '#aec7e8'
    for p in model.pvalues[FEATURES]
]
ci_lo = model.conf_int()[0][FEATURES]
ci_hi = model.conf_int()[1][FEATURES]
y_pos = range(len(FEATURES))

axes[1].barh(list(y_pos), model.params[FEATURES], color=sig_colors, alpha=0.7)
axes[1].errorbar(
    model.params[FEATURES],
    list(y_pos),
    xerr=[model.params[FEATURES] - ci_lo, ci_hi - model.params[FEATURES]],
    fmt='none', color='black', linewidth=1.2, capsize=4
)
axes[1].axvline(0, color='black', linewidth=0.8)
axes[1].set_yticks(list(y_pos))
axes[1].set_yticklabels(FEATURES, fontsize=10)
axes[1].set_xlabel('Coefficient (effect on claim volatility CV)', fontsize=10)
axes[1].set_title('Regression Coefficients with 95% CI\n(green = p < 0.05)', fontsize=11)

plt.suptitle('Crop Diversity → Claim Volatility: Multiple Regression', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PROJ, 'graphics', 'volatility_regression.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f'\nPlot saved to {PROJ}/graphics/volatility_regression.png')
