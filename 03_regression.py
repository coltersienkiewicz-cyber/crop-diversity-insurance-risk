"""OLS regression of indemnity per acre on Shannon diversity index.

Reads:  data/county_year_df.csv
Output: prints regression summaries, shows scatter plots
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PROJ     = '/Users/coltms/Downloads/543_Project'
DATA_DIR = os.path.join(PROJ, 'data')

county_year_df = pd.read_csv(os.path.join(DATA_DIR, 'county_year_df.csv'), dtype=str)
for col in ['shannon', 'indemnity_per_acre']:
    county_year_df[col] = pd.to_numeric(county_year_df[col], errors='coerce')

# collapse to one point per county (mean across years)
county_df = (
    county_year_df
    .groupby(['state_code', 'county_code', 'county_name', 'state_abbreviation'])[
        ['shannon', 'indemnity_per_acre']
    ]
    .mean()
    .reset_index()
    .dropna()
)
county_df = county_df[np.isfinite(county_df['indemnity_per_acre'])]

# top outliers
print('Top 5 by indemnity/acre:')
print(county_df.nlargest(5, 'indemnity_per_acre')[
    ['county_name', 'state_abbreviation', 'shannon', 'indemnity_per_acre']
])

# ── OLS (all counties) ────────────────────────────────────────────────────────

model_all = smf.ols('indemnity_per_acre ~ shannon', data=county_df).fit()
print('\n--- OLS (all counties) ---')
print(model_all.summary())

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(county_df['shannon'], county_df['indemnity_per_acre'],
           alpha=0.3, s=15, color='steelblue', label='County avg')
x_rng = np.linspace(county_df['shannon'].min(), county_df['shannon'].max(), 200)
ax.plot(x_rng,
        model_all.params['Intercept'] + model_all.params['shannon'] * x_rng,
        color='firebrick', linewidth=2, label='OLS fit')
ax.set_xlabel("Shannon Diversity Index (H')", fontsize=12)
ax.set_ylabel('Indemnity per Acre ($)', fontsize=12)
ax.set_title('Crop Diversity vs. Insurance Loss Intensity by County', fontsize=13)
ax.legend()
ax.annotate(f"R² = {model_all.rsquared:.3f}   p = {model_all.pvalues['shannon']:.3e}",
            xy=(0.05, 0.92), xycoords='axes fraction', fontsize=11)
plt.tight_layout()
plt.show()

# ── OLS (Dixie County excluded) ───────────────────────────────────────────────

county_df_clean = county_df[county_df['county_name'] != 'Dixie']
model_clean = smf.ols('indemnity_per_acre ~ shannon', data=county_df_clean).fit()
print('\n--- OLS (Dixie excluded) ---')
print(model_clean.summary())

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(county_df_clean['shannon'], county_df_clean['indemnity_per_acre'],
           alpha=0.3, s=15, color='steelblue', label='County avg')
x_rng = np.linspace(county_df_clean['shannon'].min(), county_df_clean['shannon'].max(), 200)
ax.plot(x_rng,
        model_clean.params['Intercept'] + model_clean.params['shannon'] * x_rng,
        color='firebrick', linewidth=2, label='OLS fit')
ax.set_xlabel("Shannon Diversity Index (H')", fontsize=12)
ax.set_ylabel('Indemnity per Acre ($)', fontsize=12)
ax.set_title('Crop Diversity vs. Insurance Loss Intensity by County', fontsize=13)
ax.legend()
ax.annotate(f"R² = {model_clean.rsquared:.3f}   p = {model_clean.pvalues['shannon']:.3e}",
            xy=(0.05, 0.92), xycoords='axes fraction', fontsize=11)
plt.tight_layout()
plt.show()
