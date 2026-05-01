"""Load raw RMA pipe-delimited .txt files and save to data/ as CSVs."""
import glob
import os

import pandas as pd

PROJ     = '/Users/coltms/Downloads/543_Project'
BASE     = os.path.join(PROJ, 'crop_loss_data')
DATA_DIR = os.path.join(PROJ, 'data')

COLSOM_COLS = [
    'commodity_year', 'state_code', 'state_abbreviation', 'county_code', 'county_name',
    'commodity_code', 'commodity_name', 'insurance_plan_code', 'insurance_plan_abbr',
    'coverage_category', 'stage_code', 'cause_of_loss_code', 'cause_of_loss_description',
    'month_of_loss', 'month_of_loss_name', 'year_of_loss', 'policies_earning_premium',
    'policies_indemnified', 'net_planted_quantity', 'net_endorsed_acres', 'liability',
    'total_premium', 'producer_paid_premium', 'subsidy', 'state_private_subsidy',
    'additional_subsidy', 'efa_premium_discount', 'net_determined_quantity',
    'indemnity_amount', 'loss_ratio',
]
SOBSCCC_COLS = [
    'commodity_year', 'location_state_code', 'location_state_abbreviation',
    'location_county_code', 'location_county_name', 'commodity_code', 'commodity_name',
    'insurance_plan_code', 'insurance_plan_abbr', 'coverage_category', 'delivery_type',
    'coverage_level', 'policies_sold_count', 'policies_earning_premium_count',
    'policies_indemnified_count', 'units_earning_premium_count', 'units_indemnified_count',
    'quantity_type', 'net_reported_quantity', 'endorsed_companion_acres', 'liability_amount',
    'total_premium_amount', 'subsidy_amount', 'state_private_subsidy', 'additional_subsidy',
    'efa_premium_discount', 'indemnity_amount', 'loss_ratio',
]
SOBTPU_COLS = [
    'commodity_year', 'state_code', 'state_name', 'state_abbreviation', 'county_code',
    'county_name', 'commodity_code', 'commodity_name', 'insurance_plan_code',
    'insurance_plan_abbreviation', 'coverage_type_code', 'coverage_level_percent',
    'delivery_id', 'type_code', 'type_name', 'practice_code', 'practice_name',
    'unit_structure_code', 'unit_structure_name', 'net_reporting_level_amount',
    'reporting_level_type', 'liability_amount', 'total_premium_amount', 'subsidy_amount',
    'indemnity_amount', 'loss_ratio', 'endorsed_commodity_reporting_level_amount',
]


def load_txt_dir(folder, cols):
    frames = []
    for path in sorted(glob.glob(os.path.join(folder, '*.txt'))):
        df = pd.read_csv(path, sep='|', header=None, names=cols,
                         dtype=str, encoding='latin-1')
        str_cols = df.select_dtypes('object').columns
        df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
        frames.append(df)
        print(f'  {os.path.basename(path):35s}  {len(df):>10,} rows')
    if not frames:
        print(f'  warning: no .txt files found in {folder}')
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)


datasets = {
    'colsommonth': (os.path.join(BASE, 'cost_of_loss'),       COLSOM_COLS),
    'sobsccc':     (os.path.join(BASE, 'state_county_crop'),   SOBSCCC_COLS),
    'sobtpu':      (os.path.join(BASE, 'type_practice_usage'), SOBTPU_COLS),
}

os.makedirs(DATA_DIR, exist_ok=True)

for name, (folder, cols) in datasets.items():
    print(f'\nLoading {name}...')
    df = load_txt_dir(folder, cols)
    out = os.path.join(DATA_DIR, f'{name}.csv')
    df.to_csv(out, index=False)
    print(f'  → saved {len(df):,} rows to {out}')
