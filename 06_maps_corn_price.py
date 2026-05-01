"""Choropleth maps of corn gross value vs national average by ERS Farm Resource Region, 2000–2024.

Reads:  Experiment_1/CornCostReturn.csv, Farm Resource Regions shapefile + reglink.xls
Writes: corn_price_maps/corn_price_<year>.png
"""
import os

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

PROJ     = '/Users/coltms/Downloads/543_Project'
SHP      = os.path.join(PROJ, 'Farm Resource Regions', 'tl_2025_us_county', 'tl_2025_us_county.shp')
REGLINK  = os.path.join(PROJ, 'Farm Resource Regions', 'reglink.xls')
CORN_CSV = os.path.join(PROJ, 'USDA_PRICE_DATA', 'CornCostReturn.csv')
OUT_DIR  = os.path.join(PROJ, 'corn_price_maps')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Corn price data ───────────────────────────────────────────────────────────

corn = pd.read_csv(CORN_CSV)
corn = corn[
    (corn['Item'] == 'Primary product, grain') &
    (corn['Category'] == 'Gross value of production') &
    (corn['Year'].between(2000, 2024))
].copy()

national = (corn[corn['Region'] == 'U.S. total']
            [['Year', 'Value']].rename(columns={'Value': 'national_avg'}))
regional = (corn[corn['Region'] != 'U.S. total']
            [['Year', 'Region', 'Value']].rename(columns={'Value': 'regional_price'}))

price_df = regional.merge(national, on='Year')
price_df['above_national'] = price_df['regional_price'] > price_df['national_avg']

# align CSV region names to ERS region names from reglink
region_name_map = {
    'Heartland':             'Heartland',
    'Northern Crescent':     'Northern Crescent',
    'Northern Great Plains': 'Northern Great Plains',
    'Prairie Gateway':       'Prairie Gateway',
    'Eastern Uplands':       'Eastern Uplands',
    'Southern Seaboard':     'Southern Seaboard',
}
price_df['region_name'] = price_df['Region'].map(region_name_map)
price_df = price_df.dropna(subset=['region_name'])

# ── Build ERS region geometries ───────────────────────────────────────────────

region_labels = {
    1: 'Heartland', 2: 'Northern Crescent', 3: 'Northern Great Plains',
    4: 'Prairie Gateway', 5: 'Eastern Uplands', 6: 'Southern Seaboard',
    7: 'Fruitful Rim', 8: 'Basin and Range', 9: 'Mississippi Portal',
}
reglink = pd.read_excel(REGLINK, header=2, usecols=['Fips', 'ERS resource region']
                        ).dropna(subset=['Fips', 'ERS resource region'])
reglink['GEOID']       = reglink['Fips'].astype(int).astype(str).str.zfill(5)
reglink['region_name'] = reglink['ERS resource region'].astype(int).map(region_labels)

gdf_base   = gpd.read_file(SHP)
conus_base = gdf_base[~gdf_base['STATEFP'].isin(['02', '15', '60', '66', '69', '72', '78'])].copy()
conus_reg  = conus_base.merge(reglink[['GEOID', 'region_name']], on='GEOID', how='left')
region_gdf = conus_reg.dissolve(by='region_name').reset_index()

# ── Per-year maps ─────────────────────────────────────────────────────────────

for year in range(2000, 2025):
    yr = price_df[price_df['Year'] == year]
    if yr.empty:
        continue

    gdf_yr = region_gdf.merge(
        yr[['region_name', 'regional_price', 'above_national']],
        on='region_name', how='left'
    )

    def region_color(row):
        if pd.isna(row['above_national']):
            return '#cccccc'
        return '#2ca02c' if row['above_national'] else '#d62728'

    gdf_yr['color'] = gdf_yr.apply(region_color, axis=1)

    fig, ax = plt.subplots(figsize=(14, 8))
    gdf_yr.plot(color=gdf_yr['color'], ax=ax, edgecolor='white', linewidth=0.5)

    for _, row in gdf_yr.iterrows():
        if row.geometry is None:
            continue
        c = row.geometry.centroid
        label = row['region_name'] or ''
        if not pd.isna(row.get('regional_price')):
            label += f"\n${row['regional_price']:.0f}/acre"
        ax.annotate(label, xy=(c.x, c.y), ha='center', fontsize=7,
                    fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.2', fc='none', ec='none'))

    nat_avg = national[national['Year'] == year]['national_avg'].values
    if len(nat_avg):
        ax.text(0.01, 0.02, f'US avg: ${nat_avg[0]:.0f}/acre',
                transform=ax.transAxes, fontsize=9, color='black',
                bbox=dict(fc='white', ec='gray', alpha=0.8))

    ax.legend(handles=[
        mpatches.Patch(color='#2ca02c', label='Above national avg'),
        mpatches.Patch(color='#d62728', label='Below national avg'),
        mpatches.Patch(color='#cccccc', label='No regional data'),
    ], loc='lower left', fontsize=9)
    ax.set_axis_off()
    ax.set_title(f'Corn Gross Value of Production vs. National Average — {year}',
                 fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'corn_price_{year}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved {year}')

print(f'\nAll corn price maps saved to {OUT_DIR}')
