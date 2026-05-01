"""Step 3 — Build the county × year feature matrix.

One row per (commodity_year, state_code, county_code) with:

  shannon_diversity       Shannon entropy H' of insured-acre crop mix
  num_unique_crops        Count of distinct commodities with positive acreage
  top_crop_share          Berger-Parker dominance = max crop-share (= 1 means monoculture)
  total_liability         Sum of total insured liability across all crops ($)
  avg_coverage_level      Liability-weighted mean coverage-level fraction [0–1]
  dominant_insurance_plan Insurance plan with the most liability in that county-year
  num_loss_causes         # distinct cause-type groups that triggered claims
  weather_loss_share      Fraction of total colsommonth indemnity from weather causes
  loss_ratio              TARGET: total indemnity / total premium (from sobsccc)

Reads:  data/county_panel.csv, data/county_year_df.csv, data/sobsccc.csv
Writes: data/feature_matrix.csv
"""
import os

import numpy as np
import pandas as pd

PROJ     = '/Users/coltms/Downloads/543_Project'
DATA_DIR = os.path.join(PROJ, 'data')
KEY      = ['commodity_year', 'state_code', 'county_code']

# ── 1. Panel spine (cause-of-loss features + loss_ratio) ─────────────────────

print('Loading county_panel...')
panel = pd.read_csv(os.path.join(DATA_DIR, 'county_panel.csv'), dtype=str)
panel['commodity_year'] = panel['commodity_year'].str.strip()
panel['state_code']     = panel['state_code'].str.strip().str.zfill(2)
panel['county_code']    = panel['county_code'].str.strip().str.zfill(3)

MONEY_COLS = [
    'total_indemnity', 'total_liability', 'loss_ratio',
    'indemnity_drought', 'indemnity_heat', 'indemnity_cold',
    'indemnity_wind_storm', 'indemnity_precipitation',
    'indemnity_disease', 'indemnity_pest', 'indemnity_fire',
    'n_cause_types',
]
for c in MONEY_COLS:
    panel[c] = pd.to_numeric(panel[c], errors='coerce')

# weather_loss_share: numerator and denominator both from colsommonth so the
# ratio stays in [0, 1] and isn't contaminated by sobtpu's different grain
WEATHER_COLS    = ['indemnity_drought', 'indemnity_heat', 'indemnity_cold',
                   'indemnity_wind_storm', 'indemnity_precipitation']
ALL_CAUSE_COLS  = WEATHER_COLS + ['indemnity_disease', 'indemnity_pest', 'indemnity_fire']

panel['_weather']   = panel[WEATHER_COLS].sum(axis=1)
panel['_all_cause'] = panel[ALL_CAUSE_COLS].sum(axis=1)
panel['weather_loss_share'] = (
    panel['_weather'] / panel['_all_cause']
).where(panel['_all_cause'] > 0)

print(f'  {len(panel):,} county-years')

# ── 2. Diversity metrics (from county_year_df) ────────────────────────────────

print('Loading county_year_df...')
cyd = pd.read_csv(os.path.join(DATA_DIR, 'county_year_df.csv'), dtype=str)
cyd['commodity_year'] = cyd['commodity_year'].str.strip()
cyd['state_code']     = cyd['state_code'].str.strip().str.zfill(2)
cyd['county_code']    = cyd['county_code'].str.strip().str.zfill(3)
for c in ['shannon', 'n_crops', 'dominance', 'evenness']:
    cyd[c] = pd.to_numeric(cyd[c], errors='coerce')

diversity = cyd[KEY + ['shannon', 'n_crops', 'dominance', 'evenness']].copy()
print(f'  {len(diversity):,} county-years')

# ── 3. Coverage features from sobsccc (liability-weighted) ───────────────────

print('Loading sobsccc for coverage features...')
scc = pd.read_csv(
    os.path.join(DATA_DIR, 'sobsccc.csv'),
    usecols=['commodity_year', 'location_state_code', 'location_county_code',
             'coverage_level', 'insurance_plan_abbr', 'liability_amount'],
    dtype=str,
)
scc = scc.rename(columns={
    'location_state_code':  'state_code',
    'location_county_code': 'county_code',
})
scc['commodity_year'] = scc['commodity_year'].str.strip()
scc['state_code']     = scc['state_code'].str.strip().str.zfill(2)
scc['county_code']    = scc['county_code'].str.strip().str.zfill(3)
scc['coverage_level']   = pd.to_numeric(scc['coverage_level'],   errors='coerce')
scc['liability_amount'] = pd.to_numeric(scc['liability_amount'],  errors='coerce').clip(lower=0)

# liability-weighted mean coverage level per county-year
scc['_cov_x_liab'] = scc['coverage_level'] * scc['liability_amount']
cov_num = scc.groupby(KEY)['_cov_x_liab'].sum()
cov_den = scc.groupby(KEY)['liability_amount'].sum()
avg_cov = (cov_num / cov_den).where(cov_den > 0).rename('avg_coverage_level').reset_index()

# dominant plan = plan with highest total liability per county-year
plan_liab = (
    scc.groupby(KEY + ['insurance_plan_abbr'])['liability_amount']
    .sum()
    .reset_index()
)
dominant_plan = (
    plan_liab
    .sort_values('liability_amount', ascending=False)
    .groupby(KEY, as_index=False)
    .first()[KEY + ['insurance_plan_abbr']]
    .rename(columns={'insurance_plan_abbr': 'dominant_insurance_plan'})
)

cov_features = avg_cov.merge(dominant_plan, on=KEY, how='outer')
print(f'  {len(cov_features):,} county-years from sobsccc')

# ── 4. Assemble feature matrix ────────────────────────────────────────────────

print('\nAssembling feature matrix...')
fm = (
    panel[KEY + ['state_name', 'state_abbreviation', 'county_name',
                 'total_liability', 'n_cause_types', 'weather_loss_share', 'loss_ratio']]
    .merge(diversity, on=KEY, how='left')
    .merge(cov_features, on=KEY, how='left')
)

fm = fm.rename(columns={
    'shannon':       'shannon_diversity',
    'n_crops':       'num_unique_crops',
    'dominance':     'top_crop_share',
    'n_cause_types': 'num_loss_causes',
})

FINAL_COLS = [
    'commodity_year', 'state_code', 'county_code',
    'state_name', 'state_abbreviation', 'county_name',
    # features
    'shannon_diversity', 'num_unique_crops', 'top_crop_share', 'evenness',
    'total_liability', 'avg_coverage_level', 'dominant_insurance_plan',
    'num_loss_causes', 'weather_loss_share',
    # target
    'loss_ratio',
]
fm = fm[FINAL_COLS]

print(f'Feature matrix shape: {fm.shape[0]:,} rows × {fm.shape[1]} columns')
print(f'\nNull counts:')
print(fm.isnull().sum().to_string())

# ── 5. Save ───────────────────────────────────────────────────────────────────

out = os.path.join(DATA_DIR, 'feature_matrix.csv')
fm.to_csv(out, index=False)
print(f'\nSaved → {out}')
