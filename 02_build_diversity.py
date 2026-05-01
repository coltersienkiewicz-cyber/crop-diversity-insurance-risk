"""Compute multi-metric crop diversity indices and build county-level DataFrames.

Metrics computed per county-year:
  - n_crops      : species richness (# distinct crops with positive acreage)
  - shannon      : Shannon entropy H' = -Σ p_i ln(p_i)
  - simpson      : Gini-Simpson D = 1 - Σ p_i²
  - evenness     : Pielou's J = H' / ln(n_crops)  [0=uneven, 1=perfectly even]
  - dominance    : Berger-Parker d = max(p_i)      [0=diverse, 1=monoculture]

County-summary adds:
  - shannon_stability : std of H' across years (how much the crop mix changes)

Reads:  data/sobtpu.csv, data/sobsccc.csv, county_landmass.csv
Writes: data/county_year_df.csv, data/county_summary.csv
"""
import os

import numpy as np
import pandas as pd

PROJ     = '/Users/coltms/Downloads/543_Project'
DATA_DIR = os.path.join(PROJ, 'data')

# ── Load raw datasets ─────────────────────────────────────────────────────────

print('Loading sobtpu...')
sobtpu_df = pd.read_csv(os.path.join(DATA_DIR, 'sobtpu.csv'), dtype=str)
sobtpu_df['net_reporting_level_amount'] = pd.to_numeric(
    sobtpu_df['net_reporting_level_amount'], errors='coerce'
)
sobtpu_df['indemnity_amount'] = pd.to_numeric(sobtpu_df['indemnity_amount'], errors='coerce')
print(f'  {sobtpu_df.shape[0]:,} rows')

# diversity and insured-acreage calcs must use only Acres-unit rows
# (sobtpu also tracks Trees, Head, Tons, etc. which can't be summed as acres)
sobtpu_acres = sobtpu_df[sobtpu_df['reporting_level_type'].str.strip() == 'Acres'].copy()
print(f'  {sobtpu_acres.shape[0]:,} rows with reporting_level_type == Acres')

print('Loading county landmass...')
landmass = pd.read_csv(os.path.join(PROJ, 'county_landmass.csv'), dtype=str)
landmass['state_code']        = landmass['FIPS_state'].str.zfill(2)
landmass['county_code']       = landmass['FIPS_county'].str.zfill(3)
landmass['county_land_acres'] = pd.to_numeric(landmass['land_sq_mi'], errors='coerce') * 640
landmass = landmass[['state_code', 'county_code', 'county_land_acres']]
print(f'  {len(landmass):,} counties')

print('Loading sobsccc...')
sobsccc_df = pd.read_csv(os.path.join(DATA_DIR, 'sobsccc.csv'), dtype=str)
sobsccc_df['liability_amount'] = pd.to_numeric(sobsccc_df['liability_amount'], errors='coerce')
sobsccc_df['coverage_level']   = pd.to_numeric(sobsccc_df['coverage_level'],   errors='coerce')
print(f'  {sobsccc_df.shape[0]:,} rows')

# ── Build county-crop proportions ─────────────────────────────────────────────

KEY = ['commodity_year', 'state_code', 'county_code']

# sum acreage per county-year-crop (Acres rows, positive values only)
county_crop = (
    sobtpu_acres[sobtpu_acres['net_reporting_level_amount'] > 0]
    .groupby(KEY + ['commodity_name'], as_index=False)['net_reporting_level_amount']
    .sum()
)

county_totals = (
    county_crop
    .groupby(KEY, as_index=False)['net_reporting_level_amount']
    .sum()
    .rename(columns={'net_reporting_level_amount': 'total_quantity'})
)

county_crop = county_crop.merge(county_totals, on=KEY)
county_crop['p_i'] = county_crop['net_reporting_level_amount'] / county_crop['total_quantity']

# ── Diversity metrics per county-year ─────────────────────────────────────────

# richness: number of crops
richness = (
    county_crop.groupby(KEY)['commodity_name']
    .nunique()
    .rename('n_crops')
    .reset_index()
)

# Shannon H'
county_crop['p_ln_p'] = county_crop['p_i'].apply(lambda p: p * np.log(p) if p > 0 else 0)
shannon = (
    county_crop.groupby(KEY)['p_ln_p']
    .sum().mul(-1)
    .rename('shannon')
    .reset_index()
)

# Simpson D = 1 - Σp²
county_crop['p_sq'] = county_crop['p_i'] ** 2
simpson = (
    county_crop.groupby(KEY)['p_sq']
    .sum().rsub(1)
    .rename('simpson')
    .reset_index()
)

# Berger-Parker dominance = max(p_i)
dominance = (
    county_crop.groupby(KEY)['p_i']
    .max()
    .rename('dominance')
    .reset_index()
)

# Combine and derive evenness = H' / ln(S)
diversity = (
    richness
    .merge(shannon,   on=KEY)
    .merge(simpson,   on=KEY)
    .merge(dominance, on=KEY)
)
diversity['evenness'] = np.where(
    diversity['n_crops'] > 1,
    diversity['shannon'] / np.log(diversity['n_crops']),
    0.0
)

print(f"\nDiversity metric ranges (county-year level):")
for col in ['n_crops', 'shannon', 'simpson', 'evenness', 'dominance']:
    print(f"  {col:12s}: {diversity[col].min():.3f} – {diversity[col].max():.3f}")

# ── county_year_df ────────────────────────────────────────────────────────────

# total insured acres from Acres-only rows; indemnity from all rows
insured_acres = (
    sobtpu_acres[sobtpu_acres['net_reporting_level_amount'] > 0]
    .groupby(KEY)['net_reporting_level_amount']
    .sum()
    .rename('insured_acres')
    .reset_index()
)

county_year_df = (
    sobtpu_df
    .groupby(KEY)
    .agg(
        state_name         = ('state_name',         'first'),
        state_abbreviation = ('state_abbreviation', 'first'),
        county_name        = ('county_name',        'first'),
        total_indemnity    = ('indemnity_amount',    'sum'),
    )
    .reset_index()
)
county_year_df = (
    county_year_df
    .merge(insured_acres, on=KEY, how='left')
    .merge(diversity,     on=KEY, how='left')
    .merge(landmass,      on=['state_code', 'county_code'], how='left')
)
county_year_df['indemnity_per_acre'] = (
    county_year_df['total_indemnity'] / county_year_df['insured_acres']
)
county_year_df['pct_land_insured'] = (
    county_year_df['insured_acres'] / county_year_df['county_land_acres']
)

print(f'\ncounty_year_df: {county_year_df.shape[0]:,} rows × {county_year_df.shape[1]} columns')

# ── Policy features from sobsccc ─────────────────────────────────────────────

policy_features = (
    sobsccc_df
    .groupby(['location_state_code', 'location_county_code'])
    .apply(lambda g: pd.Series({
        'mean_coverage_level': (
            np.average(g['coverage_level'].fillna(0),
                       weights=g['liability_amount'].fillna(0).clip(lower=0))
            if g['liability_amount'].fillna(0).sum() > 0 else np.nan
        ),
        'pct_buyup': (
            g.loc[g['coverage_category'] == 'A', 'liability_amount'].sum() /
            g['liability_amount'].sum()
            if g['liability_amount'].sum() > 0 else np.nan
        ),
    }), include_groups=False)
    .reset_index()
    .rename(columns={
        'location_state_code':  'state_code',
        'location_county_code': 'county_code',
    })
)

# ── county_summary ────────────────────────────────────────────────────────────

county_summary = (
    county_year_df
    .groupby(['state_code', 'county_code', 'county_name', 'state_abbreviation'])
    .agg(
        mean_shannon      = ('shannon',            'mean'),
        mean_simpson      = ('simpson',            'mean'),
        mean_evenness     = ('evenness',           'mean'),
        mean_dominance    = ('dominance',          'mean'),
        mean_richness     = ('n_crops',            'mean'),
        shannon_stability = ('shannon',            'std'),
        mean_indem_acre   = ('indemnity_per_acre', 'mean'),
        std_indem_acre    = ('indemnity_per_acre', 'std'),
        mean_insured_acres = ('insured_acres',     'mean'),
        mean_pct_insured  = ('pct_land_insured',   'mean'),
        county_land_acres = ('county_land_acres',  'first'),
    )
    .reset_index()
    .dropna(subset=['mean_shannon', 'mean_indem_acre'])
)

county_summary['cv_indemnity'] = (
    county_summary['std_indem_acre'] / county_summary['mean_indem_acre']
)

# drop Dixie outlier + degenerate rows
county_summary = county_summary[
    (county_summary['county_name'] != 'Dixie') &
    np.isfinite(county_summary['cv_indemnity']) &
    (county_summary['mean_indem_acre'] > 0)
].reset_index(drop=True)

county_summary = county_summary.merge(policy_features, on=['state_code', 'county_code'], how='left')
print(f'county_summary: {len(county_summary):,} counties × {county_summary.shape[1]} columns')
print(f"  county_land_acres coverage: {county_summary['county_land_acres'].notna().sum()} / {len(county_summary)} counties matched")

# ── Save ──────────────────────────────────────────────────────────────────────

county_year_df.to_csv(os.path.join(DATA_DIR, 'county_year_df.csv'), index=False)
county_summary.to_csv(os.path.join(DATA_DIR, 'county_summary.csv'), index=False)
print(f'\nSaved to {DATA_DIR}')
