"""Step 5 — Statistical Hypothesis Testing: diversity vs. loss ratio.

Null hypothesis (H₀): Crop diversity has no effect on insurance loss ratio.

Three test families, each run at two levels of analysis:
  (A) county × year — full panel (n ≈ 10 k), note: same county repeats across years
  (B) county average — one point per county (n ≈ 2.7 k), observations are independent

Tests performed:
  1. Spearman correlation  — rank-based, robust to the heavy right-skew of loss_ratio
  2. Kruskal-Wallis ANOVA  — non-parametric F-test across diversity quartiles
     + pairwise Mann-Whitney with Bonferroni correction
  3. Mann-Whitney U test   — top-50% vs. bottom-50% diversity split

Diversity metrics tested: shannon_diversity, num_unique_crops, top_crop_share, evenness

Effect sizes reported:
  Spearman:    r_s (already an effect size)
  KW:          eta² = (H - k + 1) / (n - k)      [0 = no effect, 1 = perfect]
  Mann-Whitney: rank-biserial r = 1 - 2U / (n₁·n₂)

Reads:  data/feature_matrix.csv
Writes: hypothesis_tests/results_summary.txt
        hypothesis_tests/fig_distributions.png
        hypothesis_tests/fig_spearman.png
        hypothesis_tests/fig_kruskal_wallis.png
        hypothesis_tests/fig_mannwhitney.png
"""
import os
import textwrap

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

PROJ    = '/Users/coltms/Downloads/543_Project'
OUT_DIR = os.path.join(PROJ, 'hypothesis_tests')
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', font_scale=1.05)
ALPHA = 0.05

# ── Load & prepare ────────────────────────────────────────────────────────────

fm = pd.read_csv(
    os.path.join(PROJ, 'data', 'feature_matrix.csv'),
    dtype={'state_code': str, 'county_code': str, 'commodity_year': str},
)
clean_cy = fm.dropna(subset=['shannon_diversity', 'loss_ratio']).copy()

# county-level averages (one row per county — truly independent observations)
DIV_METRICS = ['shannon_diversity', 'num_unique_crops', 'top_crop_share', 'evenness']
clean_c = (
    clean_cy
    .dropna(subset=DIV_METRICS + ['loss_ratio'])
    .groupby(['state_code', 'county_code'])[DIV_METRICS + ['loss_ratio']]
    .mean()
    .reset_index()
)

print(f'County-year observations : {len(clean_cy):,}')
print(f'Unique counties          : {len(clean_c):,}')
print(f'\nloss_ratio  skewness={clean_cy.loss_ratio.skew():.2f}  '
      f'kurtosis={clean_cy.loss_ratio.kurtosis():.2f}  '
      f'→ strongly right-skewed, non-parametric tests required')

# ── Figure 0: loss_ratio distribution (justifies test choice) ────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
lr_cap = clean_cy['loss_ratio'].quantile(0.99)

for ax, data, title in [
    (axes[0], clean_cy['loss_ratio'].clip(upper=lr_cap), 'County-Year (raw, capped at 99th pct)'),
    (axes[1], np.log1p(clean_cy['loss_ratio']),           'County-Year (log(1+x) transformed)'),
]:
    ax.hist(data, bins=60, color='steelblue', edgecolor='white', linewidth=0.3)
    ax.set_xlabel('Loss Ratio', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(title, fontsize=10)

fig.suptitle('Loss Ratio Distribution — Justification for Non-Parametric Tests',
             fontsize=12, fontweight='bold')
sw_stat, sw_p = stats.shapiro(clean_cy['loss_ratio'].sample(5000, random_state=42))
fig.text(0.5, -0.02,
         f'Shapiro-Wilk (n=5000): W={sw_stat:.4f},  p={sw_p:.1e}  →  normality strongly rejected',
         ha='center', fontsize=10, style='italic')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: Spearman correlation
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('TEST 1 — SPEARMAN CORRELATION')
print('='*70)

spearman_rows = []

for metric in DIV_METRICS:
    for level, df, n_label in [
        ('county-year', clean_cy, f'n={len(clean_cy):,}'),
        ('county-avg',  clean_c,  f'n={len(clean_c):,}'),
    ]:
        sub = df[[metric, 'loss_ratio']].dropna()
        r_s, p_s = stats.spearmanr(sub[metric], sub['loss_ratio'])
        sig = '***' if p_s < 0.001 else ('**' if p_s < 0.01 else ('*' if p_s < 0.05 else 'ns'))
        spearman_rows.append({
            'metric': metric, 'level': level, 'n': len(sub),
            'r_s': r_s, 'p': p_s, 'sig': sig,
        })
        print(f'  {metric:22s}  [{level}  {n_label}]  '
              f'r_s={r_s:+.4f}  p={p_s:.4e}  {sig}')

spearman_df = pd.DataFrame(spearman_rows)

# Spearman scatter for shannon (county-avg level — independent observations)
fig, axes = plt.subplots(1, len(DIV_METRICS), figsize=(16, 4))
for ax, metric in zip(axes, DIV_METRICS):
    sub = clean_c[[metric, 'loss_ratio']].dropna()
    row = spearman_df[(spearman_df.metric == metric) & (spearman_df.level == 'county-avg')].iloc[0]

    ax.scatter(sub[metric], sub['loss_ratio'].clip(upper=sub['loss_ratio'].quantile(0.99)),
               alpha=0.3, s=12, color='steelblue', linewidths=0)

    # lowess smoother
    from statsmodels.nonparametric.smoothers_lowess import lowess
    lr_clip = sub['loss_ratio'].clip(upper=sub['loss_ratio'].quantile(0.99))
    xv = sub[metric].values
    yv = lr_clip.values
    order = np.argsort(xv)
    smooth = lowess(yv[order], xv[order], frac=0.35)
    ax.plot(smooth[:, 0], smooth[:, 1], color='firebrick', linewidth=2)

    ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=9)
    ax.set_ylabel('Loss Ratio (capped 99th)' if ax == axes[0] else '', fontsize=9)
    color = '#d62728' if abs(row['r_s']) > 0.1 else '#aec7e8'
    ax.set_title(f"r_s = {row['r_s']:+.3f}   p = {row['p']:.3e}   {row['sig']}",
                 fontsize=9, color=color)

fig.suptitle('Spearman Correlation: Diversity Metrics vs. Loss Ratio  (county averages)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_spearman.png'), dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: Kruskal-Wallis across diversity quartiles + pairwise post-hoc
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('TEST 2 — KRUSKAL-WALLIS ANOVA ACROSS DIVERSITY QUARTILES')
print('='*70)

QUARTILE_LABELS = ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']

kw_rows    = []
posthoc_all = {}

for metric in DIV_METRICS:
    for level, df, n_label in [
        ('county-year', clean_cy, f'n={len(clean_cy):,}'),
        ('county-avg',  clean_c,  f'n={len(clean_c):,}'),
    ]:
        sub = df[[metric, 'loss_ratio']].dropna().copy()
        sub['q'] = pd.qcut(sub[metric], q=4, labels=QUARTILE_LABELS)
        groups = [grp['loss_ratio'].values for _, grp in sub.groupby('q', observed=True)]

        H, p_kw = stats.kruskal(*groups)
        n = len(sub)
        k = len(groups)
        eta2 = max(0, (H - k + 1) / (n - k))   # eta-squared effect size

        sig = '***' if p_kw < 0.001 else ('**' if p_kw < 0.01 else ('*' if p_kw < 0.05 else 'ns'))
        kw_rows.append({
            'metric': metric, 'level': level, 'n': n,
            'H': H, 'p': p_kw, 'eta2': eta2, 'sig': sig,
        })
        print(f'  {metric:22s}  [{level}  {n_label}]  '
              f'H={H:.2f}  p={p_kw:.4e}  eta²={eta2:.4f}  {sig}')

        # pairwise Mann-Whitney with Bonferroni for the shannon county-avg case
        if metric == 'shannon_diversity' and level == 'county-avg':
            pairs   = [(QUARTILE_LABELS[i], QUARTILE_LABELS[j])
                       for i in range(4) for j in range(i+1, 4)]
            raw_p   = []
            u_stats = []
            for q1_lbl, q2_lbl in pairs:
                g1 = sub.loc[sub['q'] == q1_lbl, 'loss_ratio'].values
                g2 = sub.loc[sub['q'] == q2_lbl, 'loss_ratio'].values
                u, p_mw = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                raw_p.append(p_mw)
                u_stats.append(u)

            reject, p_corr, _, _ = multipletests(raw_p, method='bonferroni')
            print(f'\n    Pairwise post-hoc (Bonferroni-corrected):')
            for (q1_lbl, q2_lbl), u, p_raw, p_adj, rej in zip(
                    pairs, u_stats, raw_p, p_corr, reject):
                flag = '*** REJECT H₀' if rej else 'ns'
                print(f'      {q1_lbl} vs {q2_lbl:10s}  '
                      f'U={u:.0f}  p_raw={p_raw:.4e}  p_bonf={p_adj:.4e}  {flag}')
            posthoc_all[metric] = (pairs, u_stats, raw_p, p_corr, reject)
            posthoc_sub = sub.copy()

kw_df = pd.DataFrame(kw_rows)

# KW box plot (shannon, county-avg)
sub_plot = clean_c[['shannon_diversity', 'loss_ratio']].dropna().copy()
sub_plot['Diversity Quartile'] = pd.qcut(
    sub_plot['shannon_diversity'], q=4, labels=QUARTILE_LABELS)
lr_cap_c = sub_plot['loss_ratio'].quantile(0.99)
sub_plot['loss_ratio_plot'] = sub_plot['loss_ratio'].clip(upper=lr_cap_c)

kw_row = kw_df[(kw_df.metric == 'shannon_diversity') & (kw_df.level == 'county-avg')].iloc[0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# left: boxplot
palette = sns.color_palette('YlOrRd_r', 4)
sns.boxplot(data=sub_plot, x='Diversity Quartile', y='loss_ratio_plot',
            hue='Diversity Quartile', palette=palette, width=0.55,
            fliersize=2, linewidth=1.2, ax=axes[0], legend=False)
axes[0].set_xlabel("Shannon Diversity Quartile", fontsize=10)
axes[0].set_ylabel(f'Loss Ratio (capped at {lr_cap_c:.2f})', fontsize=10)
axes[0].set_title(
    f"KW: H={kw_row['H']:.2f},  p={kw_row['p']:.2e},  eta²={kw_row['eta2']:.4f}  {kw_row['sig']}",
    fontsize=10,
)

# annotate group medians
grp_meds = sub_plot.groupby('Diversity Quartile', observed=True)['loss_ratio_plot'].median()
for i, (q, med) in enumerate(grp_meds.items()):
    axes[0].text(i, med + 0.01, f'{med:.3f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')

# right: mean ± 95% CI per quartile
grp_stats = (
    sub_plot.groupby('Diversity Quartile', observed=True)['loss_ratio']
    .agg(mean='mean', std='std', n='count')
    .reset_index()
)
grp_stats['se']   = grp_stats['std'] / np.sqrt(grp_stats['n'])
grp_stats['ci95'] = 1.96 * grp_stats['se']

x_pos = range(4)
axes[1].bar(x_pos, grp_stats['mean'], color=palette, alpha=0.8, width=0.55)
axes[1].errorbar(x_pos, grp_stats['mean'], yerr=grp_stats['ci95'],
                 fmt='none', color='black', linewidth=1.5, capsize=5)
axes[1].set_xticks(list(x_pos))
axes[1].set_xticklabels(QUARTILE_LABELS, fontsize=9)
axes[1].set_ylabel('Mean Loss Ratio', fontsize=10)
axes[1].set_title('Mean Loss Ratio ± 95% CI by Quartile', fontsize=10)

fig.suptitle('Kruskal-Wallis Test: Loss Ratio Across Diversity Quartiles  (county averages)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_kruskal_wallis.png'), dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: Mann-Whitney U — high vs. low diversity (median split)
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('TEST 3 — MANN-WHITNEY U: HIGH vs. LOW DIVERSITY (MEDIAN SPLIT)')
print('='*70)

mw_rows = []

for metric in DIV_METRICS:
    for level, df, n_label in [
        ('county-year', clean_cy, f'n={len(clean_cy):,}'),
        ('county-avg',  clean_c,  f'n={len(clean_c):,}'),
    ]:
        sub = df[[metric, 'loss_ratio']].dropna().copy()
        med = sub[metric].median()
        low  = sub.loc[sub[metric] <= med, 'loss_ratio'].values
        high = sub.loc[sub[metric] >  med, 'loss_ratio'].values

        u, p_mw = stats.mannwhitneyu(high, low, alternative='two-sided')
        # rank-biserial r = 1 - 2U / (n1 * n2)   [−1 to +1, sign = direction]
        rb_r = 1 - 2 * u / (len(high) * len(low))

        sig = '***' if p_mw < 0.001 else ('**' if p_mw < 0.01 else ('*' if p_mw < 0.05 else 'ns'))
        mw_rows.append({
            'metric': metric, 'level': level,
            'n_low': len(low), 'n_high': len(high),
            'med_low': np.median(low), 'med_high': np.median(high),
            'U': u, 'p': p_mw, 'rank_biserial_r': rb_r, 'sig': sig,
        })
        direction = 'HIGH > LOW' if np.median(high) > np.median(low) else 'HIGH < LOW'
        print(f'  {metric:22s}  [{level}  {n_label}]  '
              f'U={u:.0f}  p={p_mw:.4e}  rb_r={rb_r:+.4f}  {sig}  ({direction})')

mw_df = pd.DataFrame(mw_rows)

# Mann-Whitney violin plot (all metrics, county-avg)
fig, axes = plt.subplots(1, len(DIV_METRICS), figsize=(16, 5))
lr_cap_c2 = clean_c['loss_ratio'].quantile(0.99)

for ax, metric in zip(axes, DIV_METRICS):
    sub = clean_c[[metric, 'loss_ratio']].dropna().copy()
    med_val = sub[metric].median()
    sub['Group'] = np.where(sub[metric] > med_val, 'High diversity', 'Low diversity')
    sub['lr_plot'] = sub['loss_ratio'].clip(upper=lr_cap_c2)

    row = mw_df[(mw_df.metric == metric) & (mw_df.level == 'county-avg')].iloc[0]

    sns.violinplot(data=sub, x='Group', y='lr_plot', hue='Group',
                   palette=['#2ca02c', '#d62728'], inner='quartile',
                   linewidth=1, ax=ax, legend=False)
    ax.set_xlabel('')
    ax.set_ylabel('Loss Ratio (capped)' if ax == axes[0] else '', fontsize=9)
    ax.set_title(
        f"{metric.replace('_', ' ').title()}\n"
        f"p = {row['p']:.3e}  {row['sig']}\n"
        f"rb_r = {row['rank_biserial_r']:+.3f}",
        fontsize=8.5,
    )
    # median annotations
    for i, grp in enumerate(['High diversity', 'Low diversity']):
        med_lr = sub.loc[sub['Group'] == grp, 'lr_plot'].median()
        ax.text(i, med_lr, f'{med_lr:.3f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.2', fc='grey', alpha=0.6, ec='none'))

fig.suptitle('Mann-Whitney U Test: High vs. Low Diversity  (county averages, median split)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_mannwhitney.png'), dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Summary table + narrative
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('RESULTS SUMMARY')
print('='*70)

lines = []

lines.append('HYPOTHESIS TESTING RESULTS')
lines.append('=' * 70)
lines.append(f'Null hypothesis (H₀): Crop diversity has no effect on insurance loss ratio')
lines.append(f'Data: {len(clean_cy):,} county-year observations, {len(clean_c):,} unique counties')
lines.append(f'Years: {sorted(clean_cy.commodity_year.unique())}')
lines.append(f'Alpha: {ALPHA}')
lines.append('')
lines.append('Loss ratio distribution: skewness=3.25, kurtosis=20.9')
lines.append('Shapiro-Wilk (n=5000): p=1.38e-68  → normality rejected → non-parametric tests used')
lines.append('')

lines.append('TEST 1: SPEARMAN CORRELATION  (primary metric: shannon_diversity)')
lines.append('-' * 70)
for _, row in spearman_df[spearman_df.metric == 'shannon_diversity'].iterrows():
    lines.append(
        f"  [{row['level']:12s}]  r_s = {row['r_s']:+.4f}   "
        f"p = {row['p']:.4e}   {row['sig']}"
    )
lines.append('')
lines.append('  All four diversity metrics (shannon, n_crops, top_crop_share, evenness):')
for _, row in spearman_df[spearman_df.level == 'county-avg'].iterrows():
    lines.append(
        f"    {row['metric']:22s}  r_s = {row['r_s']:+.4f}   "
        f"p = {row['p']:.4e}   {row['sig']}"
    )
lines.append('')

lines.append('TEST 2: KRUSKAL-WALLIS ANOVA ACROSS QUARTILES  (shannon_diversity, county-avg)')
lines.append('-' * 70)
kw_row = kw_df[(kw_df.metric == 'shannon_diversity') & (kw_df.level == 'county-avg')].iloc[0]
lines.append(
    f"  H = {kw_row['H']:.4f}   p = {kw_row['p']:.4e}   "
    f"eta² = {kw_row['eta2']:.4f}   {kw_row['sig']}"
)
if 'shannon_diversity' in posthoc_all:
    pairs, u_stats, raw_p, p_corr, reject = posthoc_all['shannon_diversity']
    lines.append('  Bonferroni-corrected pairwise comparisons:')
    for (q1, q2), p_adj, rej in zip(pairs, p_corr, reject):
        lines.append(
            f"    {q1} vs {q2:10s}  p_bonf = {p_adj:.4e}  "
            f"{'REJECT H₀' if rej else 'fail to reject'}"
        )
lines.append('')

lines.append('TEST 3: MANN-WHITNEY U — HIGH vs. LOW DIVERSITY  (median split, county-avg)')
lines.append('-' * 70)
for _, row in mw_df[mw_df.level == 'county-avg'].iterrows():
    direction = 'HIGH > LOW' if row['med_high'] > row['med_low'] else 'HIGH < LOW'
    lines.append(
        f"  {row['metric']:22s}  U = {row['U']:.0f}   p = {row['p']:.4e}   "
        f"rb_r = {row['rank_biserial_r']:+.4f}   {row['sig']}   ({direction})"
    )
lines.append('')

lines.append('INTERPRETATION')
lines.append('-' * 70)
shannon_c = spearman_df[(spearman_df.metric == 'shannon_diversity') &
                        (spearman_df.level == 'county-avg')].iloc[0]
mw_shannon = mw_df[(mw_df.metric == 'shannon_diversity') & (mw_df.level == 'county-avg')].iloc[0]

if shannon_c['p'] < ALPHA:
    direction_word = 'POSITIVE' if shannon_c['r_s'] > 0 else 'NEGATIVE'
    lines.append(
        f"  H₀ REJECTED at alpha={ALPHA}. Shannon diversity has a statistically "
        f"significant {direction_word} association with loss ratio "
        f"(r_s = {shannon_c['r_s']:+.4f}, p = {shannon_c['p']:.2e})."
    )
    lines.append(
        f"  Effect size is {'small' if abs(shannon_c['r_s']) < 0.1 else 'moderate' if abs(shannon_c['r_s']) < 0.3 else 'large'} "
        f"(|r_s| = {abs(shannon_c['r_s']):.4f})."
    )
else:
    lines.append(
        f"  H₀ CANNOT BE REJECTED at alpha={ALPHA}. "
        f"Shannon diversity shows no statistically significant association with "
        f"loss ratio at the county-average level "
        f"(r_s = {shannon_c['r_s']:+.4f}, p = {shannon_c['p']:.2e})."
    )

if mw_shannon['sig'] != 'ns':
    lines.append(
        f"  Mann-Whitney confirms: HIGH-diversity counties have a "
        f"{'HIGHER' if mw_shannon['med_high'] > mw_shannon['med_low'] else 'LOWER'} "
        f"median loss ratio ({mw_shannon['med_high']:.3f} vs {mw_shannon['med_low']:.3f}, "
        f"p = {mw_shannon['p']:.2e}, rb_r = {mw_shannon['rank_biserial_r']:+.4f})."
    )

lines.append('')
lines.append(
    '  NOTE: County-year tests include ~4 repeated observations per county '
    '(years 1999–2001, 2004).'
)
lines.append(
    '  County-average tests (independent observations) are the primary inference basis.'
)

summary_text = '\n'.join(lines)
print('\n' + summary_text)

summary_path = os.path.join(OUT_DIR, 'results_summary.txt')
with open(summary_path, 'w') as f:
    f.write(summary_text + '\n')

print(f'\nSaved results → {summary_path}')
print(f'Saved figures → {OUT_DIR}/')
