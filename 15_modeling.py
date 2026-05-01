"""Step 6 — Regression & ML Modeling: predicting loss_ratio from diversity + features.

Stage A — OLS (interpretable)
  Three nested models, each adding controls, to isolate the diversity coefficient:
    A1: log1p(loss_ratio) ~ diversity metrics only
    A2: + liability, coverage, loss causes, weather share
    A3: + year fixed effects + state fixed effects
         (clustered standard errors at county level)

  Target is log1p(loss_ratio) to handle the heavy right skew and zero values.
  Coefficients are directly comparable (a 1-unit change in H' → β * 100% approx change
  in loss_ratio).

Stage B — Predictive (GradientBoosting + RandomForest)
  Temporal split: train 1999–2001, test 2004
  Reports R², MAE, RMSE on held-out 2004 data
  Feature importance (gain) from both models
  SHAP values from GBM: beeswarm summary + diversity dependence plots

Reads:  data/feature_matrix.csv
Writes: modeling/ols_results.txt
        modeling/fig_ols_coef.png
        modeling/fig_ml_importance.png
        modeling/fig_shap_summary.png
        modeling/fig_shap_dependence.png
        modeling/fig_pdp.png
"""
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import statsmodels.formula.api as smf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

PROJ    = '/Users/coltms/Downloads/543_Project'
OUT_DIR = os.path.join(PROJ, 'modeling')
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', font_scale=1.05)

DIVERSITY_FEATURES = ['shannon_diversity', 'num_unique_crops', 'top_crop_share', 'evenness']
OTHER_FEATURES     = ['total_liability', 'avg_coverage_level',
                      'num_loss_causes', 'weather_loss_share']
ML_FEATURES        = DIVERSITY_FEATURES + OTHER_FEATURES + ['dominant_insurance_plan_enc']

TRAIN_YEARS = ['1999', '2000', '2001']
TEST_YEAR   = '2004'

# ── Load & prepare ────────────────────────────────────────────────────────────

print('Loading feature matrix...')
fm = pd.read_csv(
    os.path.join(PROJ, 'data', 'feature_matrix.csv'),
    dtype={'state_code': str, 'county_code': str, 'commodity_year': str},
)
fm = fm.dropna(subset=['shannon_diversity', 'loss_ratio'])
fm['weather_loss_share'] = fm['weather_loss_share'].fillna(0)
fm['county_fips'] = fm['state_code'].str.zfill(2) + fm['county_code'].str.zfill(3)
fm['log_liability'] = np.log1p(fm['total_liability'])
fm['log_lr'] = np.log1p(fm['loss_ratio'])

# cap loss_ratio at 99th pct for ML (extreme audit events, not underwriting predictable)
lr_cap = fm['loss_ratio'].quantile(0.99)
fm['loss_ratio_cap'] = fm['loss_ratio'].clip(upper=lr_cap)
fm['log_lr_cap'] = np.log1p(fm['loss_ratio_cap'])

print(f'  {len(fm):,} county-years')
print(f'  Train (1999-2001): {fm.commodity_year.isin(TRAIN_YEARS).sum():,}  '
      f'Test (2004): {(fm.commodity_year == TEST_YEAR).sum():,}')
print(f'  loss_ratio capped at {lr_cap:.3f} (99th pct)')

# ══════════════════════════════════════════════════════════════════════════════
# STAGE A — OLS with progressively more controls
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('STAGE A — OLS REGRESSION')
print('='*70)

train_ols = fm[fm['commodity_year'].isin(TRAIN_YEARS)].copy()
test_ols  = fm[fm['commodity_year'] == TEST_YEAR].copy()

FORMULAS = {
    'A1 – Diversity only': (
        'log_lr ~ shannon_diversity + num_unique_crops + top_crop_share + evenness'
    ),
    'A2 – + Controls': (
        'log_lr ~ shannon_diversity + num_unique_crops + top_crop_share + evenness'
        ' + log_liability + avg_coverage_level + num_loss_causes + weather_loss_share'
    ),
    'A3 – + Year + State FE (clustered SE)': (
        'log_lr ~ shannon_diversity + num_unique_crops + top_crop_share + evenness'
        ' + log_liability + avg_coverage_level + num_loss_causes + weather_loss_share'
        ' + C(commodity_year) + C(state_abbreviation)'
    ),
}

ols_results   = {}
coef_rows     = []

for name, formula in FORMULAS.items():
    use_cluster = 'clustered' in name.lower()
    use_year_fe = 'C(commodity_year)' in formula
    if use_cluster:
        model = smf.ols(formula, data=train_ols).fit(
            cov_type='cluster',
            cov_kwds={'groups': train_ols['county_fips']},
        )
    else:
        model = smf.ols(formula, data=train_ols).fit()

    # holdout R² (test 2004)
    # Year FE can't extrapolate to an unseen 2004 level; substitute '2001'
    # (most recent training year) so the FE term uses the nearest known intercept
    test_for_pred = test_ols.copy()
    if use_year_fe:
        test_for_pred['commodity_year'] = '2001'
    test_pred = model.predict(test_for_pred)
    test_r2   = r2_score(test_ols['log_lr'], test_pred)

    ols_results[name] = (model, test_r2)
    print(f'\n{name}')
    print(f'  Train R² = {model.rsquared:.4f}   adj-R² = {model.rsquared_adj:.4f}   '
          f'Test R² (2004) = {test_r2:.4f}   n = {int(model.nobs):,}')

    for div_feat in DIVERSITY_FEATURES:
        if div_feat in model.params:
            beta = model.params[div_feat]
            pval = model.pvalues[div_feat]
            ci_lo, ci_hi = model.conf_int().loc[div_feat]
            sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'ns'))
            coef_rows.append({'model': name, 'feature': div_feat,
                               'coef': beta, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
                               'p': pval, 'sig': sig})
            print(f'  {div_feat:22s}  β={beta:+.5f}  [{ci_lo:+.4f}, {ci_hi:+.4f}]  '
                  f'p={pval:.3e}  {sig}')

# save full A3 summary to file
with open(os.path.join(OUT_DIR, 'ols_results.txt'), 'w') as f:
    f.write('OLS STAGE A — FULL MODELS\n\n')
    for name, (model, test_r2) in ols_results.items():
        f.write(f'\n{"="*60}\n{name}\nTest R² (2004) = {test_r2:.4f}\n{"="*60}\n')
        f.write(model.summary().as_text())
        f.write('\n')
print('\nOLS summaries saved → modeling/ols_results.txt')

# ── Figure: diversity coefficients across three models ───────────────────────

coef_df = pd.DataFrame(coef_rows)
model_labels  = list(FORMULAS.keys())
model_short   = ['A1: Diversity only', 'A2: + Controls', 'A3: + Year/State FE\n(clustered SE)']

fig, axes = plt.subplots(1, len(DIVERSITY_FEATURES), figsize=(16, 5), sharey=False)

for ax, feat in zip(axes, DIVERSITY_FEATURES):
    sub = coef_df[coef_df['feature'] == feat].copy()
    sub['model_short'] = model_short[:len(sub)]
    colors = ['#aec7e8' if s == 'ns' else
              ('#1f77b4' if s in ('*', '**') else '#d62728') for s in sub['sig']]

    x = range(len(sub))
    ax.barh(list(x), sub['coef'], color=colors, alpha=0.85, height=0.5)
    ax.errorbar(sub['coef'], list(x),
                xerr=[sub['coef'] - sub['ci_lo'], sub['ci_hi'] - sub['coef']],
                fmt='none', color='black', linewidth=1.2, capsize=4)
    ax.axvline(0, color='black', linewidth=0.9, linestyle='--')
    ax.set_yticks(list(x))
    ax.set_yticklabels(sub['model_short'], fontsize=8)
    ax.set_xlabel('Coefficient', fontsize=9)
    ax.set_title(feat.replace('_', '\n'), fontsize=9, fontweight='bold')

    for i, (_, row) in enumerate(sub.iterrows()):
        ax.text(row['coef'], i + 0.27, row['sig'], ha='center', fontsize=8, color='black')

fig.suptitle("OLS Diversity Coefficients: Stability Across Model Specifications\n"
             "(target = log(1+loss_ratio), training years 1999-2001)\n"
             "red = p<0.001  blue = p<0.05  grey = ns",
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_ols_coef.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig_ols_coef.png')

# ══════════════════════════════════════════════════════════════════════════════
# STAGE B — ML: GradientBoosting + RandomForest
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('STAGE B — MACHINE LEARNING  (train 1999-2001 → test 2004)')
print('='*70)

# encode dominant_insurance_plan (15 categories → integer)
le = LabelEncoder()
fm['dominant_insurance_plan_enc'] = le.fit_transform(fm['dominant_insurance_plan'])
plan_name_map = dict(zip(le.transform(le.classes_), le.classes_))

FEATURE_LABELS = {
    'shannon_diversity':          "Shannon H'",
    'num_unique_crops':           '# Crops',
    'top_crop_share':             'Top Crop Share',
    'evenness':                   'Evenness (J)',
    'total_liability':            'Total Liability',
    'avg_coverage_level':         'Avg Coverage Level',
    'num_loss_causes':            '# Loss Causes',
    'weather_loss_share':         'Weather Loss Share',
    'dominant_insurance_plan_enc': 'Dominant Plan',
}

X_train = fm.loc[fm.commodity_year.isin(TRAIN_YEARS), ML_FEATURES]
y_train = fm.loc[fm.commodity_year.isin(TRAIN_YEARS), 'loss_ratio_cap']
X_test  = fm.loc[fm.commodity_year == TEST_YEAR,       ML_FEATURES]
y_test  = fm.loc[fm.commodity_year == TEST_YEAR,       'loss_ratio_cap']

print(f'Train: {len(X_train):,}   Test: {len(X_test):,}   Features: {len(ML_FEATURES)}')

models_ml = {
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        min_samples_leaf=20, subsample=0.8, random_state=42,
    ),
    'RandomForest': RandomForestRegressor(
        n_estimators=400, max_depth=8, min_samples_leaf=20,
        max_features=0.6, random_state=42, n_jobs=-1,
    ),
}

ml_metrics = {}
for model_name, clf in models_ml.items():
    clf.fit(X_train, y_train)
    pred_train = clf.predict(X_train)
    pred_test  = clf.predict(X_test)

    metrics = {
        'train_r2':  r2_score(y_train, pred_train),
        'test_r2':   r2_score(y_test,  pred_test),
        'test_mae':  mean_absolute_error(y_test, pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, pred_test)),
    }
    ml_metrics[model_name] = metrics
    print(f'\n{model_name}')
    print(f'  Train R² = {metrics["train_r2"]:.4f}   '
          f'Test R² = {metrics["test_r2"]:.4f}   '
          f'MAE = {metrics["test_mae"]:.4f}   '
          f'RMSE = {metrics["test_rmse"]:.4f}')

    imp = pd.Series(clf.feature_importances_, index=ML_FEATURES).sort_values(ascending=False)
    print('  Feature importances:')
    for feat, val in imp.items():
        label = FEATURE_LABELS.get(feat, feat)
        rank  = ' <-- diversity' if feat in DIVERSITY_FEATURES else ''
        print(f'    {label:30s}  {val:.4f}{rank}')

gbm = models_ml['GradientBoosting']
rf  = models_ml['RandomForest']

# ── Feature importance comparison plot ───────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (name, clf) in zip(axes, models_ml.items()):
    imp = (
        pd.Series(clf.feature_importances_, index=ML_FEATURES)
        .rename(index=FEATURE_LABELS)
        .sort_values()
    )
    colors = ['#d62728' if any(f in k for f in ['Shannon', 'Crops', 'Top Crop', 'Evenness'])
              else '#aec7e8'
              for k in imp.index]
    imp.plot.barh(ax=ax, color=colors, edgecolor='none')
    ax.set_xlabel('Feature Importance (mean decrease impurity)', fontsize=9)
    ax.set_title(f'{name}', fontsize=10, fontweight='bold')
    r2 = ml_metrics[name]['test_r2']
    ax.annotate(f'Test R² = {r2:.3f}', xy=(0.97, 0.04), xycoords='axes fraction',
                ha='right', fontsize=9, bbox=dict(boxstyle='round', fc='white', alpha=0.7))

fig.suptitle('Feature Importance (red = diversity metrics)\n'
             'Train: 1999-2001   Test: 2004', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_ml_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print('\nSaved fig_ml_importance.png')

# ── SHAP values (GBM) ─────────────────────────────────────────────────────────

print('\nComputing SHAP values (GradientBoosting, test set)...')
explainer  = shap.TreeExplainer(gbm)
shap_vals  = explainer(X_test)   # Explanation object (shap 0.40+)
shap_array = shap_vals.values    # shape: (n_test, n_features)

# SHAP summary: beeswarm
fig, ax = plt.subplots(figsize=(9, 6))
shap.summary_plot(
    shap_array, X_test,
    feature_names=[FEATURE_LABELS.get(f, f) for f in ML_FEATURES],
    show=False, plot_type='dot', color_bar=True,
    max_display=len(ML_FEATURES),
)
plt.title('SHAP Values — GradientBoosting (test set, 2004)\n'
          'Each dot = one county; color = feature value (blue=low, red=high)',
          fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_shap_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig_shap_summary.png')

# SHAP dependence plots for all four diversity metrics
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes_flat = axes.flatten()

for ax, feat in zip(axes_flat, DIVERSITY_FEATURES):
    idx = ML_FEATURES.index(feat)
    feat_vals  = X_test[feat].values
    shap_feat  = shap_array[:, idx]

    # color by weather_loss_share (key confounder)
    color_feat_idx = ML_FEATURES.index('weather_loss_share')
    color_vals     = X_test['weather_loss_share'].values

    sc = ax.scatter(feat_vals, shap_feat, c=color_vals, cmap='RdYlGn_r',
                    alpha=0.4, s=12, linewidths=0, vmin=0, vmax=1)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    # lowess trend
    from statsmodels.nonparametric.smoothers_lowess import lowess
    order = np.argsort(feat_vals)
    sm    = lowess(shap_feat[order], feat_vals[order], frac=0.4)
    ax.plot(sm[:, 0], sm[:, 1], color='black', linewidth=2)

    ax.set_xlabel(FEATURE_LABELS.get(feat, feat), fontsize=10)
    ax.set_ylabel('SHAP value\n(impact on log loss_ratio)', fontsize=9)

    # annotate direction
    slope_text = 'Higher diversity → higher loss risk' if np.corrcoef(feat_vals, shap_feat)[0,1] > 0 \
                 else 'Higher diversity → lower loss risk'
    ax.set_title(FEATURE_LABELS.get(feat, feat), fontsize=10, fontweight='bold')
    ax.annotate(slope_text, xy=(0.5, 0.92), xycoords='axes fraction',
                ha='center', fontsize=8, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8, ec='none'))

plt.colorbar(sc, ax=axes_flat[-1], label='Weather Loss Share', shrink=0.8)
fig.suptitle('SHAP Dependence Plots — Diversity Features\n'
             'Color = weather loss share (red=high weather losses)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_shap_dependence.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig_shap_dependence.png')

# ── Partial dependence plot (shannon_diversity) ────────────────────────────────

print('Computing partial dependence plot...')
fig, ax = plt.subplots(figsize=(8, 5))
shannon_idx = ML_FEATURES.index('shannon_diversity')
PartialDependenceDisplay.from_estimator(
    gbm, X_train, features=[shannon_idx],
    feature_names=[FEATURE_LABELS.get(f, f) for f in ML_FEATURES],
    kind='both', subsample=500, random_state=42,
    ax=ax, line_kw={'color': 'firebrick', 'linewidth': 2},
    ice_lines_kw={'alpha': 0.04, 'color': 'steelblue'},
)
ax.set_xlabel("Shannon Diversity (H')", fontsize=11)
ax.set_ylabel('Predicted Loss Ratio (marginal effect)', fontsize=11)
ax.set_title("Partial Dependence: Shannon Diversity → Loss Ratio\n"
             "(red = average PD, blue = individual county ICE curves)",
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_pdp.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig_pdp.png')

# ══════════════════════════════════════════════════════════════════════════════
# Final summary
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('MODELING SUMMARY')
print('='*70)

a3_model, a3_test_r2 = ols_results['A3 – + Year + State FE (clustered SE)']
shannon_beta = a3_model.params['shannon_diversity']
shannon_p    = a3_model.pvalues['shannon_diversity']
shannon_sig  = '***' if shannon_p < 0.001 else ('**' if shannon_p < 0.01 else ('*' if shannon_p < 0.05 else 'ns'))

gbm_imp     = pd.Series(gbm.feature_importances_, index=ML_FEATURES).sort_values(ascending=False)
div_imp_rank = {f: int(gbm_imp.index.get_loc(f)) + 1 for f in DIVERSITY_FEATURES}
total_div_imp = sum(gbm.feature_importances_[ML_FEATURES.index(f)] for f in DIVERSITY_FEATURES)

print(f'\nOLS (Model A3, clustered SE, state+year FE):')
print(f'  shannon_diversity  β={shannon_beta:+.5f}  p={shannon_p:.3e}  {shannon_sig}')
print(f'  Train R² = {a3_model.rsquared:.4f}   Test R² (2004) = {a3_test_r2:.4f}')

print(f'\nGradientBoosting (temporal split 1999-2001 → 2004):')
print(f'  Test R²={ml_metrics["GradientBoosting"]["test_r2"]:.4f}  '
      f'MAE={ml_metrics["GradientBoosting"]["test_mae"]:.4f}  '
      f'RMSE={ml_metrics["GradientBoosting"]["test_rmse"]:.4f}')
print(f'  Diversity feature ranks (out of {len(ML_FEATURES)}):')
for feat, rank in sorted(div_imp_rank.items(), key=lambda x: x[1]):
    label = FEATURE_LABELS.get(feat, feat)
    imp_val = gbm.feature_importances_[ML_FEATURES.index(feat)]
    print(f'    #{rank}  {label:25s}  importance={imp_val:.4f}')
print(f'  Total diversity importance: {total_div_imp:.4f} '
      f'({total_div_imp / gbm.feature_importances_.sum() * 100:.1f}% of total)')
