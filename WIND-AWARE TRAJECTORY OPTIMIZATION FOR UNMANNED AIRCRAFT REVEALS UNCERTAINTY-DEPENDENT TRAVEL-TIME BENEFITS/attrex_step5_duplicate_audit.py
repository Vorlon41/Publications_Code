
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
CLEAN = ROOT / 'analysis_step2'
OUT = ROOT / 'analysis_step5'
OUT.mkdir(exist_ok=True)
NAMES = [
    'ATTREX-MMS-1HZ_GHawk_20140313_R3_clean.csv',
    'ATTREX-MMS-1HZ_GHawk_20140314_R0_clean.csv',
]

def load(name):
    path = CLEAN / name
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}')
    df = pd.read_csv(path, low_memory=False)
    required = ['TIME_UTC', 'G_LAT', 'G_LONG', 'G_ALT']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'{name} missing columns: {missing}. Available: {list(df.columns)}')
    for col in required + [c for c in ['TAS', 'U', 'V'] if c in df]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=required).copy()
    df['lon_normalized'] = ((df['G_LONG'] + 180) % 360) - 180
    df['lat_round'] = df['G_LAT'].round(4)
    df['lon_round'] = df['lon_normalized'].round(4)
    df['alt_round'] = df['G_ALT'].round(0)
    return df

a, b = map(load, NAMES)
print('March 13 rows:', len(a), 'March 14 rows:', len(b))
print('March 13 TIME_UTC range:', a.TIME_UTC.min(), a.TIME_UTC.max())
print('March 14 TIME_UTC range:', b.TIME_UTC.min(), b.TIME_UTC.max())

# Exact matching rounded positions can reveal replayed tracks even with shifted dates.
key = ['lat_round', 'lon_round', 'alt_round']
unique_a = a[key].drop_duplicates()
unique_b = b[key].drop_duplicates()
matched = unique_a.merge(unique_b, on=key, how='inner')
frac_a = len(matched) / max(len(unique_a), 1)
frac_b = len(matched) / max(len(unique_b), 1)
print(f'Unique rounded position matches: {len(matched):,}')
print(f'Match fraction of Mar 13 unique positions: {frac_a:.2%}')
print(f'Match fraction of Mar 14 unique positions: {frac_b:.2%}')

# Compare records at identical seconds-of-day, while recognizing that
# flight segments can span midnight and dates may differ.
cols = ['TIME_UTC', 'G_LAT', 'G_LONG', 'G_ALT']
x = a[cols].drop_duplicates('TIME_UTC')
y = b[cols].drop_duplicates('TIME_UTC')
joined = x.merge(y, on='TIME_UTC', suffixes=('_13', '_14'))
if len(joined):
    lat_diff = (joined.G_LAT_13 - joined.G_LAT_14).abs()
    lon_diff = ((joined.G_LONG_13 - joined.G_LONG_14 + 180) % 360 - 180).abs()
    alt_diff = (joined.G_ALT_13 - joined.G_ALT_14).abs()
    print('Same TIME_UTC samples:', len(joined))
    print('Median latitude difference (deg):', lat_diff.median())
    print('Median longitude difference (deg):', lon_diff.median())
    print('Median altitude difference (m):', alt_diff.median())
else:
    print('No matching TIME_UTC values; comparing coordinate overlap only.')

summary = pd.DataFrame([{
    'file_a': NAMES[0], 'file_b': NAMES[1],
    'valid_rows_a': len(a), 'valid_rows_b': len(b),
    'unique_positions_a': len(unique_a), 'unique_positions_b': len(unique_b),
    'shared_rounded_positions': len(matched),
    'fraction_positions_a_shared': frac_a,
    'fraction_positions_b_shared': frac_b,
    'matching_seconds_of_day': len(joined),
}])
summary.to_csv(OUT / 'duplicate_audit_summary.csv', index=False)
print('\nSaved:', OUT / 'duplicate_audit_summary.csv')
print('Interpretation: High positional overlap warrants checking flight dates,')
print('UTC rollover, original ICARTT headers, and source provenance before deduplication.')
