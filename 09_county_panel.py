"""Build a comprehensive county-year panel combining all three RMA datasets.

One row per county-year containing:
  From sobtpu  : insured_acres, total_indemnity, indemnity_per_insured_acre
  From landmass: county_land_acres, pct_land_insured (ag-use proxy)
  From colsom  : policies_indemnified and total_indemnity broken out by
                 cause-of-loss group (drought, heat, cold, wind/storm,
                 precipitation, pest, disease, fire)
  From sobsccc : total policies earning premium, total liability

Reads:  data/colsommonth.csv, data/sobsccc.csv, data/sobtpu.csv,
        county_landmass.csv
Writes: data/county_panel.csv
"""
import os

import numpy as np
import pandas as pd

PROJ     = '/Users/coltms/Downloads/543_Project'
DATA_DIR = os.path.join(PROJ, 'data')

CAUSE_GROUPS = {
    '11': 'drought',       '13': 'drought',       '14': 'drought',
    '12': 'heat',          '22': 'heat',           '45': 'heat',
    '41': 'cold',          '42': 'cold',           '43': 'cold',
    '44': 'cold',          '74': 'cold',
    '21': 'wind_storm',    '61': 'wind_storm',     '62': 'wind_storm',
    '63': 'wind_storm',    '64': 'wind_storm',
    '31': 'precipitation', '51': 'precipitation',  '65': 'precipitation',
    '67': 'precipitation', '92': 'precipitation',
    '71': 'pest',          '73': 'pest',           '93': 'pest',
    '80': 'disease',       '81': 'disease',        '82': 'disease',
    '76': 'disease',
    '91': 'fire',          '95': 'fire',
}
CAUSE_COLS = ['drought', 'heat', 'cold', 'wind_storm',
              'precipitation', 'pest', 'disease', 'fire']

KEY = ['commodity_year', 'state_code', 'county_code']

# ── 1. County landmass ────────────────────────────────────────────────────────

print('Loading county landmass...')
landmass = pd.read_csv(os.path.join(PROJ, 'county_landmass.csv'), dtype=str)
landmass['state_code']        = landmass['FIPS_state'].str.zfill(2)
landmass['county_code']       = landmass['FIPS_county'].str.zfill(3)
landmass['county_land_acres'] = pd.to_numeric(landmass['land_sq_mi'], errors='coerce') * 640
landmass = landmass[['state_code', 'county_code', 'county_land_acres']]
print(f'  {len(landmass):,} counties')

# ── 2. sobtpu → insured acres + total indemnity ───────────────────────────────

print('Loading sobtpu...')
sobtpu = pd.read_csv(
    os.path.join(DATA_DIR, 'sobtpu.csv'),
    usecols=['commodity_year', 'state_code', 'county_code', 'state_name',
             'state_abbreviation', 'county_name',
             'net_reporting_level_amount', 'reporting_level_type', 'indemnity_amount'],
    dtype=str,
)
sobtpu['state_code']   = sobtpu['state_code'].str.strip().str.zfill(2)
sobtpu['county_code']  = sobtpu['county_code'].str.strip().str.zfill(3)
sobtpu['net_reporting_level_amount'] = pd.to_numeric(
    sobtpu['net_reporting_level_amount'], errors='coerce')
sobtpu['indemnity_amount'] = pd.to_numeric(sobtpu['indemnity_amount'], errors='coerce')

insured_acres = (
    sobtpu[
        (sobtpu['reporting_level_type'].str.strip() == 'Acres') &
        (sobtpu['net_reporting_level_amount'] > 0)
    ]
    .groupby(KEY)['net_reporting_level_amount']
    .sum()
    .rename('insured_acres')
    .reset_index()
)

sobtpu_agg = (
    sobtpu
    .groupby(KEY)
    .agg(
        state_name         = ('state_name',         'first'),
        state_abbreviation = ('state_abbreviation', 'first'),
        county_name        = ('county_name',        'first'),
        total_indemnity    = ('indemnity_amount',    'sum'),
    )
    .reset_index()
    .merge(insured_acres, on=KEY, how='left')
)
print(f'  {len(sobtpu_agg):,} county-years')

# ── 3. colsommonth → cause-of-loss counts + indemnity by cause ───────────────

print('Loading colsommonth...')
col_df = pd.read_csv(
    os.path.join(DATA_DIR, 'colsommonth.csv'),
    usecols=['commodity_year', 'state_code', 'county_code',
             'cause_of_loss_code', 'policies_indemnified', 'indemnity_amount'],
    dtype=str,
)
col_df['state_code']          = col_df['state_code'].str.strip().str.zfill(2)
col_df['county_code']         = col_df['county_code'].str.strip().str.zfill(3)
col_df['cause_of_loss_code']  = col_df['cause_of_loss_code'].str.strip().str.zfill(2)
col_df['policies_indemnified'] = pd.to_numeric(col_df['policies_indemnified'], errors='coerce')
col_df['indemnity_amount']     = pd.to_numeric(col_df['indemnity_amount'],     errors='coerce')

col_df['cause_group'] = col_df['cause_of_loss_code'].map(CAUSE_GROUPS)
col_known = col_df.dropna(subset=['cause_group'])

# policies indemnified per cause group per county-year
cause_policies = (
    col_known
    .groupby(KEY + ['cause_group'])['policies_indemnified']
    .sum()
    .unstack('cause_group', fill_value=0)
    .rename(columns=lambda c: f'policies_{c}')
    .reset_index()
)

# indemnity per cause group per county-year
cause_indemnity = (
    col_known
    .groupby(KEY + ['cause_group'])['indemnity_amount']
    .sum()
    .unstack('cause_group', fill_value=0)
    .rename(columns=lambda c: f'indemnity_{c}')
    .reset_index()
)

# any-event flag per cause group (1 if any claims that year)
cause_occurred = cause_policies.copy()
for c in CAUSE_COLS:
    col = f'policies_{c}'
    if col in cause_occurred.columns:
        cause_occurred[f'event_{c}'] = (cause_occurred[col] > 0).astype(int)

cause_agg = cause_policies.merge(cause_indemnity, on=KEY, how='outer')
for c in CAUSE_COLS:
    col = f'policies_{c}'
    if col in cause_occurred.columns:
        cause_agg[f'event_{c}'] = (cause_agg[col] > 0).astype(int)

print(f'  {len(cause_agg):,} county-years with cause-of-loss data')

# ── 4. sobsccc → policy counts + liability ────────────────────────────────────

print('Loading sobsccc...')
sobsccc = pd.read_csv(
    os.path.join(DATA_DIR, 'sobsccc.csv'),
    usecols=['commodity_year', 'location_state_code', 'location_county_code',
             'policies_earning_premium_count', 'liability_amount',
             'indemnity_amount', 'total_premium_amount'],
    dtype=str,
)
sobsccc = sobsccc.rename(columns={
    'location_state_code':  'state_code',
    'location_county_code': 'county_code',
})
sobsccc['state_code']  = sobsccc['state_code'].str.strip().str.zfill(2)
sobsccc['county_code'] = sobsccc['county_code'].str.strip().str.zfill(3)
sobsccc['policies_earning_premium_count'] = pd.to_numeric(
    sobsccc['policies_earning_premium_count'], errors='coerce')
sobsccc['liability_amount']    = pd.to_numeric(sobsccc['liability_amount'],    errors='coerce')
sobsccc['indemnity_amount']    = pd.to_numeric(sobsccc['indemnity_amount'],    errors='coerce')
sobsccc['total_premium_amount'] = pd.to_numeric(sobsccc['total_premium_amount'], errors='coerce')

sobsccc_agg = (
    sobsccc
    .groupby(KEY)
    .agg(
        total_policies_earning  = ('policies_earning_premium_count', 'sum'),
        total_liability         = ('liability_amount',               'sum'),
        sobsccc_indemnity       = ('indemnity_amount',               'sum'),
        sobsccc_premium         = ('total_premium_amount',           'sum'),
    )
    .reset_index()
)
# Compute loss_ratio bottom-up from components (sobsccc is the RMA policy-summary
# table with the cleanest premium denominator). Never average the pre-computed
# row-level loss_ratio fields — they are unit-weighted, not premium-weighted.
sobsccc_agg['loss_ratio'] = (
    sobsccc_agg['sobsccc_indemnity'] / sobsccc_agg['sobsccc_premium']
).where(sobsccc_agg['sobsccc_premium'] > 0)
print(f'  {len(sobsccc_agg):,} county-years')

# ── 5. Assemble panel ─────────────────────────────────────────────────────────

print('\nAssembling panel...')
panel = (
    sobtpu_agg
    .merge(cause_agg,   on=KEY, how='left')
    .merge(sobsccc_agg, on=KEY, how='left')
    .merge(landmass,    on=['state_code', 'county_code'], how='left')
)

# fill cause columns with 0 where no claims of that type occurred
cause_num_cols = (
    [f'policies_{c}' for c in CAUSE_COLS] +
    [f'indemnity_{c}' for c in CAUSE_COLS] +
    [f'event_{c}'    for c in CAUSE_COLS]
)
for col in cause_num_cols:
    if col in panel.columns:
        panel[col] = panel[col].fillna(0)

# derived columns
panel['indemnity_per_insured_acre'] = (
    panel['total_indemnity'] / panel['insured_acres']
)
panel['pct_land_insured'] = (
    panel['insured_acres'] / panel['county_land_acres']
)
# total events (distinct cause types that hit that county that year)
event_cols = [f'event_{c}' for c in CAUSE_COLS if f'event_{c}' in panel.columns]
panel['n_cause_types'] = panel[event_cols].sum(axis=1)

# drop helper columns used only for loss_ratio construction
panel = panel.drop(columns=['sobsccc_indemnity', 'sobsccc_premium'], errors='ignore')

print(f'Panel shape: {panel.shape[0]:,} rows × {panel.shape[1]} columns')
lr = panel['loss_ratio']
print(f"loss_ratio (target): {lr.notna().sum():,} non-null, "
      f"median={lr.median():.3f}, 99th={lr.quantile(0.99):.3f}, "
      f"negative={( lr < 0).sum()} (audit refunds / carryover)")
print(f"pct_land_insured range: "
      f"{panel['pct_land_insured'].min():.3f} – {panel['pct_land_insured'].quantile(0.99):.3f} "
      f"(99th pct)")

# ── 6. Save ───────────────────────────────────────────────────────────────────

out = os.path.join(DATA_DIR, 'county_panel.csv')
panel.to_csv(out, index=False)
print(f'\nSaved {len(panel):,} rows → {out}')
print('\nColumn list:')
for col in panel.columns:
    print(f'  {col}')
