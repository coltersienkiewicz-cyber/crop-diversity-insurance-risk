import os
import glob
import pandas as pd

# ── Column definitions from RMA key documents ─────────────────────────────────

# colsommonth_allyears-pdf.pdf  (cost_of_loss files, 30 fields)
COLSOM_COLS = [
    'commodity_year',
    'state_code',
    'state_abbreviation',
    'county_code',
    'county_name',
    'commodity_code',
    'commodity_name',
    'insurance_plan_code',
    'insurance_plan_abbr',
    'coverage_category',
    'stage_code',
    'cause_of_loss_code',
    'cause_of_loss_description',
    'month_of_loss',
    'month_of_loss_name',
    'year_of_loss',
    'policies_earning_premium',
    'policies_indemnified',
    'net_planted_quantity',
    'net_endorsed_acres',
    'liability',
    'total_premium',
    'producer_paid_premium',
    'subsidy',
    'state_private_subsidy',
    'additional_subsidy',
    'efa_premium_discount',
    'net_determined_quantity',
    'indemnity_amount',
    'loss_ratio',
]

# sobsccc_1989forward-pdf.pdf  (state_county_crop files, 28 fields)
SOBSCCC_COLS = [
    'commodity_year',
    'location_state_code',
    'location_state_abbreviation',
    'location_county_code',
    'location_county_name',
    'commodity_code',
    'commodity_name',
    'insurance_plan_code',
    'insurance_plan_abbr',
    'coverage_category',
    'delivery_type',
    'coverage_level',
    'policies_sold_count',
    'policies_earning_premium_count',
    'policies_indemnified_count',
    'units_earning_premium_count',
    'units_indemnified_count',
    'quantity_type',
    'net_reported_quantity',
    'endorsed_companion_acres',
    'liability_amount',
    'total_premium_amount',
    'subsidy_amount',
    'state_private_subsidy',
    'additional_subsidy',
    'efa_premium_discount',
    'indemnity_amount',
    'loss_ratio',
]

# sobtpu_allyears-doc.docx  (type_practice_usage files, 27 fields)
SOBTPU_COLS = [
    'commodity_year',
    'state_code',
    'state_name',
    'state_abbreviation',
    'county_code',
    'county_name',
    'commodity_code',
    'commodity_name',
    'insurance_plan_code',
    'insurance_plan_abbreviation',
    'coverage_type_code',
    'coverage_level_percent',
    'delivery_id',
    'type_code',
    'type_name',
    'practice_code',
    'practice_name',
    'unit_structure_code',
    'unit_structure_name',
    'net_reporting_level_amount',
    'reporting_level_type',
    'liability_amount',
    'total_premium_amount',
    'subsidy_amount',
    'indemnity_amount',
    'loss_ratio',
    'endorsed_commodity_reporting_level_amount',
]


# ── Loader helper ─────────────────────────────────────────────────────────────

def _load_dir(directory, columns):
    """Read all pipe-delimited .txt files in a directory into one DataFrame."""
    files = sorted(glob.glob(os.path.join(directory, '*.txt')))
    if not files:
        print(f"Warning: no .txt files found in '{directory}'")
        return pd.DataFrame(columns=columns)

    frames = []
    for path in files:
        try:
            df = pd.read_csv(
                path,
                sep='|',
                header=None,
                names=columns,
                dtype=str,
                encoding='latin-1',
            )
            frames.append(df)
            print(f"  loaded {os.path.basename(path):30s}  ({len(df):,} rows)")
        except Exception as e:
            print(f"  warning: could not load {path}: {e}")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


# ── Load each dataset ─────────────────────────────────────────────────────────

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'crop_loss_data')

print("Loading cost-of-loss (colsommonth) data…")
colsom_df = _load_dir(os.path.join(BASE, 'cost_of_loss'), COLSOM_COLS)

print("\nLoading state/county/crop coverage (sobsccc) data…")
sobsccc_df = _load_dir(os.path.join(BASE, 'state_county_crop'), SOBSCCC_COLS)

print("\nLoading type/practice/unit-structure (sobtpu) data…")
sobtpu_df = _load_dir(os.path.join(BASE, 'type_practice_usage'), SOBTPU_COLS)

# ── Strip whitespace from string columns ──────────────────────────────────────

for df in (colsom_df, sobsccc_df, sobtpu_df):
    str_cols = df.select_dtypes(include='object').columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\ncolsom_df  : {colsom_df.shape[0]:>10,} rows × {colsom_df.shape[1]} columns")
print(f"sobsccc_df : {sobsccc_df.shape[0]:>10,} rows × {sobsccc_df.shape[1]} columns")
print(f"sobtpu_df  : {sobtpu_df.shape[0]:>10,} rows × {sobtpu_df.shape[1]} columns")
