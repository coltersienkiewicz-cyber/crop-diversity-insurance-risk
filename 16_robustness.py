"""Step 7 — Robustness Checks.

Four confounds tested systematically. For each, the benchmark is Model A3 from
Step 6 (OLS with state + year FE, clustered SE at county level), which already
controls for state-level geography and year shocks. Each check adds or replaces
one set of controls, and we track whether the shannon_diversity coefficient
changes sign, loses significance, or shrinks materially.

Check 1 — Geography confound
    Replace 50 state FEs with 9 ERS Farm Resource Region FEs.
    If the diversity effect is just a Corn Belt / Great Plains proxy, it should
    disappear once we control for agro-climatic zones rather than political states.
    Also: run within-region Spearman to verify the result holds inside each zone.

Check 2 — Scale confound
    num_unique_crops grows mechanically with county size. Shannon H' and Evenness
    are proportion-based and scale-invariant. Run OLS using only scale-invariant
    metrics, then run separately within insured-acres quartiles to confirm the
    effect is not driven by county size.

Check 3 — Commodity confound
    Crop mix composition (shares of corn/soy/wheat/cotton) predicts expected loss
    ratios independently of diversity. Add crop-share controls to A3 and test
    whether diversity is merely a proxy for low-risk crop selection. Also compute
    a "crop-mix-adjusted" residual and check whether Shannon predicts it.

Check 4 — Coverage selection bias
    High-risk counties may purchase more insurance, inflating the loss ratio
    denominator. Add pct_land_insured as a control. If selection drives results,
    diversity should lose significance after this addition.

Summary output: coefficient stability forest plot across all 9 specifications.

Reads:  data/feature_matrix.csv, data/county_year_df.csv, data/sobtpu.csv,
        data/colsommonth.csv, Farm Resource Regions/reglink.xls
Writes: robustness/fig_check1_geography.png
        robustness/fig_check2_scale.png
        robustness/fig_check3_commodity.png
        robustness/fig_check4_selection.png
        robustness/fig_stability.png
        robustness/stability_table.csv
"""
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf

PROJ    = '/Users/coltms/Downloads/543_Project'
OUT_DIR = os.path.join(PROJ, 'robustness')
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', font_scale=1.05)
KEY   = ['commodity_year', 'state_code', 'county_code']
ALPHA = 0.05

ERS_NAMES = {
    1: 'Heartland', 2: 'N. Crescent', 3: 'N. Great Plains',
    4: 'Prairie Gateway', 5: 'E. Uplands', 6: 'S. Seaboard',
    7: 'Fruitful Rim', 8: 'Basin & Range', 9: 'MS. Portal',
}

# ── Load base feature matrix ──────────────────────────────────────────────────

print('Loading feature matrix...')
fm = pd.read_csv(
    os.path.join(PROJ, 'data', 'feature_matrix.csv'),
    dtype={'state_code': str, 'county_code': str, 'commodity_year': str},
)
fm = fm.dropna(subset=['shannon_diversity', 'loss_ratio'])
fm['weather_loss_share'] = fm['weather_loss_share'].fillna(0)
fm['county_fips']   = fm['state_code'].str.zfill(2) + fm['county_code'].str.zfill(3)
fm['log_lr']        = np.log1p(fm['loss_ratio'])
fm['log_liability'] = np.log1p(fm['total_liability'])

# ── ERS Farm Resource Regions ─────────────────────────────────────────────────

print('Loading ERS regions...')
reglink = pd.read_excel(
    os.path.join(PROJ, 'Farm Resource Regions', 'reglink.xls'),
    header=2, usecols=['Fips', 'ERS resource region'],
).dropna(subset=['Fips', 'ERS resource region'])
reglink['state_code']  = reglink['Fips'].astype(int).astype(str).str.zfill(5).str[:2]
reglink['county_code'] = reglink['Fips'].astype(int).astype(str).str.zfill(5).str[2:]
reglink['ers_region']  = reglink['ERS resource region'].astype(int).map(ERS_NAMES)
reglink = reglink[['state_code', 'county_code', 'ers_region']]
fm = fm.merge(reglink, on=['state_code', 'county_code'], how='left')
print(f'  ERS region matched: {fm.ers_region.notna().sum():,} / {len(fm):,} rows')

# ── pct_land_insured (Check 4) ────────────────────────────────────────────────

cyd = pd.read_csv(
    os.path.join(PROJ, 'data', 'county_year_df.csv'),
    dtype={'state_code': str, 'county_code': str, 'commodity_year': str},
)
cyd['state_code']  = cyd['state_code'].str.zfill(2)
cyd['county_code'] = cyd['county_code'].str.zfill(3)
cyd['pct_land_insured'] = pd.to_numeric(cyd['pct_land_insured'], errors='coerce')
cyd['insured_acres']    = pd.to_numeric(cyd['insured_acres'],    errors='coerce')
fm = fm.merge(cyd[KEY + ['pct_land_insured', 'insured_acres']], on=KEY, how='left')

# ── Commodity shares (Check 3) ────────────────────────────────────────────────

print('Computing commodity shares from sobtpu...')
tpu = pd.read_csv(os.path.join(PROJ, 'data', 'sobtpu.csv'), dtype=str)
tpu['state_code']  = tpu['state_code'].str.strip().str.zfill(2)
tpu['county_code'] = tpu['county_code'].str.strip().str.zfill(3)
tpu['commodity_name_clean'] = tpu['commodity_name'].str.strip().str.title()
tpu['net_reporting_level_amount'] = pd.to_numeric(
    tpu['net_reporting_level_amount'], errors='coerce')

acres_tpu = tpu[
    (tpu['reporting_level_type'].str.strip() == 'Acres') &
    (tpu['net_reporting_level_amount'] > 0)
].copy()
total_ac = acres_tpu.groupby(KEY)['net_reporting_level_amount'].sum().rename('_total')

crop_shares = total_ac.reset_index()
for crop, col_name in [('Corn', 'share_corn'), ('Soybeans', 'share_soy'),
                        ('Wheat', 'share_wheat'), ('Cotton', 'share_cotton')]:
    crop_ac = (
        acres_tpu[acres_tpu['commodity_name_clean'] == crop]
        .groupby(KEY)['net_reporting_level_amount'].sum()
        .rename(col_name)
    )
    crop_shares = crop_shares.merge(crop_ac, on=KEY, how='left')
    crop_shares[col_name] = (crop_shares[col_name].fillna(0) / crop_shares['_total']).clip(0, 1)

fm = fm.merge(crop_shares.drop(columns='_total'), on=KEY, how='left')
for c in ['share_corn', 'share_soy', 'share_wheat', 'share_cotton']:
    fm[c] = fm[c].fillna(0)

# ── National commodity loss ratios for expected-LR baseline ──────────────────

print('Computing national commodity loss ratios...')
col_df = pd.read_csv(os.path.join(PROJ, 'data', 'colsommonth.csv'), dtype=str)
col_df['commodity_name_clean'] = col_df['commodity_name'].str.strip().str.title()
col_df['indemnity_amount'] = pd.to_numeric(col_df['indemnity_amount'], errors='coerce')
col_df['total_premium']    = pd.to_numeric(col_df['total_premium'],    errors='coerce')

# use only the 4-year window matching our panel (1999-2001, 2004)
col_df = col_df[col_df['commodity_year'].isin(['1999','2000','2001','2004'])]
nat_lr = (col_df.groupby('commodity_name_clean')[['indemnity_amount','total_premium']]
          .sum())
nat_lr['national_lr'] = (nat_lr['indemnity_amount'] / nat_lr['total_premium']).clip(0, 10)
nat_lr = nat_lr[nat_lr['total_premium'] > 1e5]

nat_lr_map = nat_lr['national_lr'].to_dict()
CROPS_FOR_EXPECTED = {
    'Corn': 'share_corn', 'Soybeans': 'share_soy',
    'Wheat': 'share_wheat', 'Cotton': 'share_cotton',
}
fm['expected_lr'] = sum(
    fm[share_col] * nat_lr_map.get(crop, nat_lr['national_lr'].median())
    for crop, share_col in CROPS_FOR_EXPECTED.items()
)
fm['log_expected_lr'] = np.log1p(fm['expected_lr'])
fm['lr_residual']     = fm['log_lr'] - fm['log_expected_lr']

print(f'  Expected LR computed for all {len(fm):,} rows')
print(f'  Correlation expected_lr with actual loss_ratio: '
      f'{fm[["expected_lr","loss_ratio"]].corr().iloc[0,1]:.3f}')

# ══════════════════════════════════════════════════════════════════════════════
# Baseline: Model A3 from Step 6
# ══════════════════════════════════════════════════════════════════════════════

TRAIN = fm[fm.commodity_year.isin(['1999','2000','2001'])].copy()

BASE_CONTROLS = (
    'log_liability + avg_coverage_level + num_loss_causes + weather_loss_share'
    ' + C(commodity_year) + C(state_abbreviation)'
)
BASE_FORMULA = (
    'log_lr ~ shannon_diversity + num_unique_crops + top_crop_share + evenness'
    f' + {BASE_CONTROLS}'
)

def fit_ols(formula, data=TRAIN, cluster_col='county_fips'):
    return smf.ols(formula, data=data).fit(
        cov_type='cluster', cov_kwds={'groups': data[cluster_col]}
    )

def shannon_coef(model):
    b  = model.params['shannon_diversity']
    lo = model.conf_int().loc['shannon_diversity', 0]
    hi = model.conf_int().loc['shannon_diversity', 1]
    p  = model.pvalues['shannon_diversity']
    return b, lo, hi, p

stability_rows = []

print('\nFitting baseline A3...')
m_base = fit_ols(BASE_FORMULA)
b, lo, hi, p = shannon_coef(m_base)
sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
stability_rows.append({'spec': 'A3 baseline\n(state+year FE)', 'group': 'Baseline',
                        'coef': b, 'ci_lo': lo, 'ci_hi': hi, 'p': p, 'sig': sig,
                        'n': int(m_base.nobs), 'r2': m_base.rsquared})
print(f'  shannon β={b:+.4f}  [{lo:+.4f}, {hi:+.4f}]  p={p:.3e}  {sig}')

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1 — Geography: ERS region FE instead of state FE
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*65)
print('CHECK 1 — GEOGRAPHY: ERS Region FE vs. State FE')
print('='*65)

train_ers = TRAIN.dropna(subset=['ers_region'])

ERS_FORMULA = (
    'log_lr ~ shannon_diversity + num_unique_crops + top_crop_share + evenness'
    ' + log_liability + avg_coverage_level + num_loss_causes + weather_loss_share'
    ' + C(commodity_year) + C(ers_region)'
)
m_ers = fit_ols(ERS_FORMULA, data=train_ers)
b, lo, hi, p = shannon_coef(m_ers)
sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
stability_rows.append({'spec': 'ERS region FE\n(9 agro-climate zones)', 'group': 'Check 1: Geography',
                        'coef': b, 'ci_lo': lo, 'ci_hi': hi, 'p': p, 'sig': sig,
                        'n': int(m_ers.nobs), 'r2': m_ers.rsquared})
print(f'  ERS FE: shannon β={b:+.4f}  p={p:.3e}  {sig}')

# within-ERS-region Spearman
region_spearman = []
county_avg = (
    fm.dropna(subset=['ers_region', 'shannon_diversity', 'loss_ratio'])
    .groupby(['state_code', 'county_code', 'ers_region'])[['shannon_diversity', 'loss_ratio']]
    .mean().reset_index()
)
for region in sorted(county_avg['ers_region'].dropna().unique()):
    sub = county_avg[county_avg['ers_region'] == region]
    if len(sub) >= 10:
        r_s, p_s = stats.spearmanr(sub['shannon_diversity'], sub['loss_ratio'])
        region_spearman.append({'region': region, 'n': len(sub), 'r_s': r_s, 'p': p_s})
        sig_r = '***' if p_s < 0.001 else ('**' if p_s < 0.01 else ('*' if p_s < 0.05 else 'ns'))
        print(f'  Within {region:20s}: n={len(sub):4d}  r_s={r_s:+.3f}  p={p_s:.3e}  {sig_r}')

region_sp_df = pd.DataFrame(region_spearman)

# Figure 1: within-region Spearman bars
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#d62728' if r > 0 else '#1f77b4' for r in region_sp_df['r_s']]
bars = ax.barh(region_sp_df['region'], region_sp_df['r_s'], color=colors, alpha=0.8)
for bar, row in zip(bars, region_sp_df.itertuples()):
    sig_r = '***' if row.p < 0.001 else ('**' if row.p < 0.01 else ('*' if row.p < 0.05 else ''))
    x_offset = 0.005 if row.r_s >= 0 else -0.005
    ha = 'left' if row.r_s >= 0 else 'right'
    ax.text(row.r_s + x_offset, bar.get_y() + bar.get_height()/2,
            f'{sig_r}  n={row.n}', va='center', ha=ha, fontsize=8)
ax.axvline(0, color='black', linewidth=0.9)
ax.set_xlabel("Within-Region Spearman r_s  (Shannon diversity vs. loss ratio)", fontsize=10)
ax.set_title("Check 1: Does Shannon → Loss Ratio Hold Within Each Agro-Climate Region?\n"
             "(county-average level; red = positive, blue = negative)", fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_check1_geography.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  Saved fig_check1_geography.png')

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 2 — Scale: scale-invariant metrics only + by-size-quartile analysis
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*65)
print('CHECK 2 — SCALE: Scale-Invariant Metrics + Size Quartiles')
print('='*65)

# 2a: OLS with only Shannon + evenness (both are proportion-based, scale-invariant)
SCALE_INV_FORMULA = (
    'log_lr ~ shannon_diversity + evenness'
    f' + {BASE_CONTROLS}'
)
m_scale = fit_ols(SCALE_INV_FORMULA)
b, lo, hi, p = shannon_coef(m_scale)
sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
stability_rows.append({'spec': 'Scale-invariant only\n(Shannon+Evenness)', 'group': 'Check 2: Scale',
                        'coef': b, 'ci_lo': lo, 'ci_hi': hi, 'p': p, 'sig': sig,
                        'n': int(m_scale.nobs), 'r2': m_scale.rsquared})
print(f'  Scale-invariant: shannon β={b:+.4f}  p={p:.3e}  {sig}')

# 2b: within-size-quartile OLS
size_quartile_rows = []
TRAIN_SQ = TRAIN.dropna(subset=['insured_acres']).copy()
TRAIN_SQ['size_q'] = pd.qcut(TRAIN_SQ['insured_acres'], q=4,
                               labels=['Q1 small', 'Q2', 'Q3', 'Q4 large'])
for q_label in ['Q1 small', 'Q2', 'Q3', 'Q4 large']:
    sub_q = TRAIN_SQ[TRAIN_SQ['size_q'] == q_label]
    if sub_q['state_abbreviation'].nunique() < 3 or len(sub_q) < 100:
        continue
    try:
        m_q = fit_ols(BASE_FORMULA, data=sub_q)
        b_q, lo_q, hi_q, p_q = shannon_coef(m_q)
        sig_q = '***' if p_q < 0.001 else ('**' if p_q < 0.01 else ('*' if p_q < 0.05 else 'ns'))
        size_quartile_rows.append({'q': q_label, 'coef': b_q, 'ci_lo': lo_q,
                                    'ci_hi': hi_q, 'p': p_q, 'sig': sig_q, 'n': len(sub_q)})
        stability_rows.append({'spec': f'Size {q_label}\n(insured acres)', 'group': 'Check 2: Scale',
                                'coef': b_q, 'ci_lo': lo_q, 'ci_hi': hi_q, 'p': p_q, 'sig': sig_q,
                                'n': len(sub_q), 'r2': m_q.rsquared})
        print(f'  {q_label}: n={len(sub_q):,}  shannon β={b_q:+.4f}  p={p_q:.3e}  {sig_q}')
    except Exception as e:
        print(f'  {q_label}: failed ({e})')

sq_df = pd.DataFrame(size_quartile_rows)

# Figure 2: scatter num_unique_crops vs county_land_acres + by-size-quartile coef
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# left: n_crops vs insured_acres scatter (scale confound diagnostic)
TRAIN_SC = TRAIN.dropna(subset=['insured_acres']).copy()
axes[0].scatter(np.log1p(TRAIN_SC['insured_acres']), TRAIN_SC['num_unique_crops'],
                alpha=0.15, s=10, color='steelblue', linewidths=0)
r_scale, p_scale = stats.pearsonr(
    np.log1p(TRAIN_SC['insured_acres'].dropna()),
    TRAIN_SC.loc[TRAIN_SC['insured_acres'].notna(), 'num_unique_crops']
)
axes[0].set_xlabel('log(Insured Acres)', fontsize=10)
axes[0].set_ylabel('# Unique Crops', fontsize=10)
axes[0].set_title(f'# Crops vs. County Size\nr={r_scale:.3f}  p={p_scale:.2e}\n'
                  f'(Shannon H\' is scale-invariant by design)', fontsize=10)

# right: shannon β by size quartile
if sq_df is not None and len(sq_df) > 0:
    x = range(len(sq_df))
    axes[1].barh(list(x), sq_df['coef'], color='steelblue', alpha=0.8, height=0.5)
    axes[1].errorbar(sq_df['coef'], list(x),
                     xerr=[sq_df['coef'] - sq_df['ci_lo'], sq_df['ci_hi'] - sq_df['coef']],
                     fmt='none', color='black', linewidth=1.2, capsize=4)
    axes[1].axvline(0, color='black', linewidth=0.9, linestyle='--')
    axes[1].set_yticks(list(x))
    axes[1].set_yticklabels([f"{r['q']}  (n={r['n']:,})" for _, r in sq_df.iterrows()], fontsize=9)
    for i, (_, row) in enumerate(sq_df.iterrows()):
        axes[1].text(row['coef'], i + 0.27, row['sig'], ha='center', fontsize=9)
    axes[1].set_xlabel("Shannon β  (95% CI)", fontsize=10)
    axes[1].set_title("Shannon Coefficient by County-Size Quartile\n"
                      "(controls: liability, coverage, year+state FE)", fontsize=10)

fig.suptitle('Check 2: Scale Confound', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_check2_scale.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  Saved fig_check2_scale.png')

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 3 — Commodity: add crop-share controls; test residuals
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*65)
print('CHECK 3 — COMMODITY: Crop-Share Controls + Residual Analysis')
print('='*65)

COMM_FORMULA = (
    'log_lr ~ shannon_diversity + num_unique_crops + top_crop_share + evenness'
    ' + log_liability + avg_coverage_level + num_loss_causes + weather_loss_share'
    ' + share_corn + share_soy + share_wheat + share_cotton'
    ' + C(commodity_year) + C(state_abbreviation)'
)
m_comm = fit_ols(COMM_FORMULA)
b, lo, hi, p = shannon_coef(m_comm)
sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
stability_rows.append({'spec': '+ Crop shares\n(corn/soy/wheat/cotton)', 'group': 'Check 3: Commodity',
                        'coef': b, 'ci_lo': lo, 'ci_hi': hi, 'p': p, 'sig': sig,
                        'n': int(m_comm.nobs), 'r2': m_comm.rsquared})
print(f'  + Crop shares: shannon β={b:+.4f}  p={p:.3e}  {sig}')

# Crop-mix-adjusted residual: does Shannon predict (actual - expected) LR?
resid_clean = TRAIN.dropna(subset=['lr_residual', 'shannon_diversity'])
RESID_FORMULA = (
    'lr_residual ~ shannon_diversity + num_unique_crops + top_crop_share + evenness'
    ' + log_liability + avg_coverage_level + num_loss_causes + weather_loss_share'
    ' + C(commodity_year) + C(state_abbreviation)'
)
m_resid = fit_ols(RESID_FORMULA, data=resid_clean)
b_r, lo_r, hi_r, p_r = shannon_coef(m_resid)
sig_r2 = '***' if p_r < 0.001 else ('**' if p_r < 0.01 else ('*' if p_r < 0.05 else 'ns'))
stability_rows.append({'spec': 'Crop-mix-adjusted\nresidual as target', 'group': 'Check 3: Commodity',
                        'coef': b_r, 'ci_lo': lo_r, 'ci_hi': hi_r, 'p': p_r, 'sig': sig_r2,
                        'n': int(m_resid.nobs), 'r2': m_resid.rsquared})
print(f'  Residual target: shannon β={b_r:+.4f}  p={p_r:.3e}  {sig_r2}')

# Figure 3: crop shares per ERS region + residual scatter
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# left: crop composition by ERS region (shows WHY commodity confound matters)
TRAIN_ERS = TRAIN.dropna(subset=['ers_region'])
crop_region = (
    TRAIN_ERS.groupby('ers_region')[['share_corn', 'share_soy', 'share_wheat', 'share_cotton']]
    .mean()
    .rename(columns={'share_corn':'Corn','share_soy':'Soybeans',
                     'share_wheat':'Wheat','share_cotton':'Cotton'})
)
crop_region.plot.barh(stacked=True, ax=axes[0], colormap='tab10', alpha=0.85)
axes[0].set_xlabel('Mean Crop Share of Insured Acres', fontsize=9)
axes[0].set_title('Crop Composition by ERS Region\n'
                  '(motivates commodity confound concern)', fontsize=10)
axes[0].legend(fontsize=8, loc='lower right')

# right: Shannon vs crop-mix-adjusted residual
r_s_resid, p_s_resid = stats.spearmanr(
    resid_clean['shannon_diversity'], resid_clean['lr_residual'])
axes[1].scatter(resid_clean['shannon_diversity'],
                resid_clean['lr_residual'].clip(-2, 2),
                alpha=0.12, s=10, color='steelblue', linewidths=0)
axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[1].set_xlabel("Shannon H'", fontsize=10)
axes[1].set_ylabel('Crop-Mix-Adjusted LR Residual (clipped ±2)', fontsize=9)
axes[1].set_title(f'Shannon vs. Residual After Removing Expected Commodity Risk\n'
                  f'Spearman r_s={r_s_resid:+.3f}  p={p_s_resid:.2e}', fontsize=10)

fig.suptitle('Check 3: Commodity Confound', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_check3_commodity.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  Saved fig_check3_commodity.png')

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 4 — Coverage selection bias: add pct_land_insured
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*65)
print('CHECK 4 — COVERAGE SELECTION BIAS: Add pct_land_insured')
print('='*65)

train_sel = TRAIN.dropna(subset=['pct_land_insured'])

SEL_FORMULA = (
    'log_lr ~ shannon_diversity + num_unique_crops + top_crop_share + evenness'
    ' + log_liability + avg_coverage_level + num_loss_causes + weather_loss_share'
    ' + pct_land_insured'
    ' + C(commodity_year) + C(state_abbreviation)'
)
m_sel = fit_ols(SEL_FORMULA, data=train_sel)
b, lo, hi, p = shannon_coef(m_sel)
sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
stability_rows.append({'spec': '+ pct_land_insured\n(selection control)', 'group': 'Check 4: Selection',
                        'coef': b, 'ci_lo': lo, 'ci_hi': hi, 'p': p, 'sig': sig,
                        'n': int(m_sel.nobs), 'r2': m_sel.rsquared})
print(f'  + pct_land_insured: shannon β={b:+.4f}  p={p:.3e}  {sig}')
print(f'    pct_land_insured β={m_sel.params.get("pct_land_insured", np.nan):+.4f}'
      f'  p={m_sel.pvalues.get("pct_land_insured", np.nan):.3e}')

# diagnostic correlations
corr_table = fm[['shannon_diversity', 'pct_land_insured', 'avg_coverage_level',
                  'loss_ratio', 'log_lr']].dropna().corr()
print('\n  Correlations (selection-bias diagnostic):')
for feat in ['pct_land_insured', 'avg_coverage_level']:
    r1 = corr_table.loc[feat, 'shannon_diversity']
    r2 = corr_table.loc[feat, 'loss_ratio']
    print(f'    {feat:25s}  r(shannon)={r1:+.3f}  r(loss_ratio)={r2:+.3f}')

# Figure 4: selection bias diagnostics
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# left: pct_land_insured vs shannon (are diverse counties buying more insurance?)
axes[0].scatter(fm['shannon_diversity'].dropna(),
                fm['pct_land_insured'].dropna().reindex(fm['shannon_diversity'].dropna().index),
                alpha=0.15, s=10, color='steelblue', linewidths=0)
r_diag = corr_table.loc['pct_land_insured', 'shannon_diversity']
axes[0].set_xlabel("Shannon H'", fontsize=10)
axes[0].set_ylabel('% of County Land Insured', fontsize=10)
axes[0].set_title(f'Selection Diagnostic: Diverse Counties Buy More Insurance?\n'
                  f'Pearson r = {r_diag:.3f}', fontsize=10)

# right: pct_land_insured vs loss_ratio
lr_capped = fm['loss_ratio'].clip(upper=fm['loss_ratio'].quantile(0.99))
axes[1].scatter(fm['pct_land_insured'].dropna(),
                lr_capped.reindex(fm['pct_land_insured'].dropna().index),
                alpha=0.15, s=10, color='firebrick', linewidths=0)
r_diag2 = corr_table.loc['pct_land_insured', 'loss_ratio']
axes[1].set_xlabel('% of County Land Insured', fontsize=10)
axes[1].set_ylabel('Loss Ratio (capped)', fontsize=10)
axes[1].set_title(f'Selection Diagnostic: More Insured → Higher Loss Ratio?\n'
                  f'Pearson r = {r_diag2:.3f}', fontsize=10)

fig.suptitle('Check 4: Coverage Selection Bias', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_check4_selection.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  Saved fig_check4_selection.png')

# ══════════════════════════════════════════════════════════════════════════════
# STABILITY FOREST PLOT — all 9 specs on one chart
# ══════════════════════════════════════════════════════════════════════════════

print('\nBuilding coefficient stability plot...')
stab_df = pd.DataFrame(stability_rows)
stab_df.to_csv(os.path.join(OUT_DIR, 'stability_table.csv'), index=False)

GROUP_COLORS = {
    'Baseline':           '#2c7bb6',
    'Check 1: Geography': '#1a9641',
    'Check 2: Scale':     '#d7191c',
    'Check 3: Commodity': '#fdae61',
    'Check 4: Selection': '#7b2d8b',
}

fig, ax = plt.subplots(figsize=(10, max(6, len(stab_df) * 0.75)))
y_pos = range(len(stab_df))

for i, (_, row) in enumerate(stab_df.iterrows()):
    color = GROUP_COLORS.get(row['group'], 'grey')
    ax.plot(row['coef'], i, 'o', color=color, markersize=8, zorder=3)
    ax.plot([row['ci_lo'], row['ci_hi']], [i, i], '-', color=color, linewidth=2.2, zorder=2)
    ax.text(row['ci_hi'] + 0.01, i, f" {row['sig']}  n={row['n']:,}",
            va='center', fontsize=8, color='black')

ax.axvline(0, color='black', linewidth=1, linestyle='--', zorder=1)
ax.axvline(stab_df.iloc[0]['coef'], color=GROUP_COLORS['Baseline'],
           linewidth=0.8, linestyle=':', alpha=0.5, zorder=0)

ax.set_yticks(list(y_pos))
ax.set_yticklabels(stab_df['spec'], fontsize=8.5)
ax.set_xlabel("Shannon Diversity Coefficient β  (95% CI)\n"
              "target = log(1 + loss_ratio); clustered SE by county", fontsize=10)
ax.set_title("Robustness: Shannon Diversity Coefficient Across All Specifications\n"
             "(dotted vertical = baseline estimate; dashed = zero)", fontsize=11, fontweight='bold')

from matplotlib.patches import Patch
legend_handles = [Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
                  if g in stab_df['group'].values]
ax.legend(handles=legend_handles, loc='lower right', fontsize=8.5, framealpha=0.9)

ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_stability.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig_stability.png')

# ── Print stability summary ───────────────────────────────────────────────────

print('\n' + '='*65)
print('STABILITY SUMMARY — shannon_diversity coefficient')
print('='*65)
print(f"{'Specification':42s}  {'β':>8}  {'p-value':>10}  {'sig':>5}  n")
print('-'*65)
for _, row in stab_df.iterrows():
    spec_flat = row['spec'].replace('\n', ' ')
    print(f"{spec_flat:42s}  {row['coef']:+8.4f}  {row['p']:10.3e}  {row['sig']:>5}  {row['n']:,}")

n_sig   = (stab_df['sig'] != 'ns').sum()
n_pos   = (stab_df['coef'] > 0).sum()
print(f'\nSignificant in {n_sig}/{len(stab_df)} specs  |  Positive in {n_pos}/{len(stab_df)} specs')
