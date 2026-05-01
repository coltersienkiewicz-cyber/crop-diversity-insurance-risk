"""Choropleth maps of Simpson Diversity Index by county, one PNG per year.
Also produces the 2004 map with ERS Farm Resource Region overlay.

Reads:  data/county_year_df.csv, Farm Resource Regions shapefile + reglink.xls
Writes: simpson_maps/simpson_<year>.png
"""
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

PROJ    = '/Users/coltms/Downloads/543_Project'
SHP     = os.path.join(PROJ, 'Farm Resource Regions', 'tl_2025_us_county', 'tl_2025_us_county.shp')
REGLINK = os.path.join(PROJ, 'Farm Resource Regions', 'reglink.xls')
OUT_DIR = os.path.join(PROJ, 'simpson_maps')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────

county_year_df = pd.read_csv(os.path.join(PROJ, 'data', 'county_year_df.csv'), dtype=str)
county_year_df['simpson'] = pd.to_numeric(county_year_df['simpson'], errors='coerce')

simpson_by_year = (
    county_year_df
    .dropna(subset=['simpson'])
    .groupby(['commodity_year', 'state_code', 'county_code'])['simpson']
    .mean()
    .reset_index()
)

vmin = simpson_by_year['simpson'].min()
vmax = simpson_by_year['simpson'].max()
years = sorted(simpson_by_year['commodity_year'].unique())

gdf_base   = gpd.read_file(SHP)
conus_base = gdf_base[~gdf_base['STATEFP'].isin(['02', '15', '60', '66', '69', '72', '78'])].copy()

# ── Per-year Simpson maps ─────────────────────────────────────────────────────

for year in years:
    yr_data = simpson_by_year[simpson_by_year['commodity_year'] == year]
    gdf = conus_base.merge(yr_data,
                           left_on=['STATEFP', 'COUNTYFP'],
                           right_on=['state_code', 'county_code'],
                           how='left')
    fig, ax = plt.subplots(figsize=(16, 9))
    gdf.plot(column='simpson', ax=ax, cmap='YlGn',
             vmin=vmin, vmax=vmax, linewidth=0.1, edgecolor='white',
             legend=True,
             missing_kwds={'color': 'lightgrey', 'label': 'No data'},
             legend_kwds={'label': 'Simpson Diversity Index (D)', 'shrink': 0.5})
    ax.set_axis_off()
    ax.set_title(f'Crop Diversity by County — Simpson Index ({year})',
                 fontsize=15, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'simpson_{year}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved {year}')

print(f'\nAll Simpson maps saved to {OUT_DIR}')

# ── 2004 map with ERS Farm Resource Region overlay ────────────────────────────

region_labels = {
    1: 'Heartland', 2: 'Northern Crescent', 3: 'Northern Great Plains',
    4: 'Prairie Gateway', 5: 'Eastern Uplands', 6: 'Southern Seaboard',
    7: 'Fruitful Rim', 8: 'Basin and Range', 9: 'Mississippi Portal',
}
reglink = pd.read_excel(REGLINK, header=2, usecols=['Fips', 'ERS resource region']
                        ).dropna(subset=['Fips', 'ERS resource region'])
reglink['GEOID']       = reglink['Fips'].astype(int).astype(str).str.zfill(5)
reglink['region_name'] = reglink['ERS resource region'].astype(int).map(region_labels)

conus_reg        = conus_base.merge(reglink[['GEOID', 'region_name']], on='GEOID', how='left')
region_boundaries = conus_reg.dissolve(by='region_name').reset_index()

yr_2004  = simpson_by_year[simpson_by_year['commodity_year'] == '2004']
gdf_2004 = conus_base.merge(yr_2004,
                             left_on=['STATEFP', 'COUNTYFP'],
                             right_on=['state_code', 'county_code'],
                             how='left')

fig, ax = plt.subplots(figsize=(16, 9))
gdf_2004.plot(column='simpson', ax=ax, cmap='YlGn',
              vmin=vmin, vmax=vmax, linewidth=0.1, edgecolor='white',
              legend=True,
              missing_kwds={'color': 'lightgrey', 'label': 'No data'},
              legend_kwds={'label': 'Simpson Diversity Index (D)', 'shrink': 0.5})
region_boundaries.boundary.plot(ax=ax, color='black', linewidth=1.2)

for _, row in region_boundaries.iterrows():
    if row['region_name'] and row.geometry:
        c = row.geometry.centroid
        ax.annotate(row['region_name'], xy=(c.x, c.y), ha='center',
                    fontsize=7, fontweight='bold', color='black',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5, ec='none'))

ax.set_axis_off()
ax.set_title('Crop Diversity by County — Simpson Index (2004) with ERS Farm Resource Regions',
             fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'simpson_2004_ers_regions.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'saved 2004 ERS overlay → {out_path}')
