from pathlib import Path
import csv
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
OUT = ROOT / 'analysis_step2'
OUT.mkdir(exist_ok=True)
VARIABLES = ['TIME_UTC','G_LAT','G_LONG','G_ALT','TAS','U','V','W','HDG','ROLL','PITCH']

def parse_ict(path):
    with path.open('r', encoding='utf-8-sig', errors='replace') as f:
        first = f.readline().strip().split(',')
        header_count, ffi = int(first[0]), int(first[1])
        if ffi != 1001:
            raise ValueError(f'Unsupported ICARTT FFI {ffi}')
        f.seek(0)
        header = [f.readline().rstrip('\r\n') for _ in range(header_count)]
        data_text = f.read()
    nvar = int(header[9].split(',')[0].strip())
    scales = [float(x) for x in header[10].split(',')]
    missing = [float(x) for x in header[11].split(',')]
    descriptions = [line.split(',')[0].strip() for line in header[12:12+nvar]]
    # Last header line in ICARTT 1001 is commonly the CSV column header.
    cols = [x.strip() for x in header[-1].split(',')]
    if len(cols) != nvar+1:
        cols = ['TIME_UTC'] + descriptions
        print(f'NOTE {path.name}: using variable descriptions for column names')
    if len(scales)!=nvar or len(missing)!=nvar:
        raise ValueError('Inconsistent ICARTT header variable metadata')
    df = pd.read_csv(io.StringIO(data_text), header=None, names=cols,
                     skipinitialspace=True, on_bad_lines='warn', low_memory=False)
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for i, col in enumerate(cols[1:]):
        raw = df[col]
        df[col] = raw.mask(np.isclose(raw, missing[i], rtol=0, atol=1e-7)) * scales[i]
    # ICARTT scale factors apply to dependent variables, not the independent TIME_UTC.
    return df, cols

summaries = []
files = sorted(ROOT.rglob('ATTREX-MMS-1HZ*.ict'))
print(f'Found {len(files)} 1-Hz files')
if not files:
    raise SystemExit('No 1-Hz files found. Check ROOT path.')

for path in files:
    print('Processing:', path.name)
    try:
        df, cols = parse_ict(path)
        missing_cols = [c for c in VARIABLES if c not in df.columns]
        if missing_cols:
            print('  Missing expected columns:', missing_cols)
        summary = {'file':path.name, 'rows':len(df), 'columns': '|'.join(cols)}
        for col in VARIABLES:
            if col in df:
                summary[f'{col}_valid_pct'] = round(100*df[col].notna().mean(),2)
        if 'TIME_UTC' in df:
            t = df['TIME_UTC'].dropna()
            summary['time_start_s'] = t.min() if len(t) else np.nan
            summary['time_end_s'] = t.max() if len(t) else np.nan
            summary['time_nonincreasing_count'] = int((t.diff().dropna()<=0).sum())
        if {'G_LAT','G_LONG'}.issubset(df):
            geo = df[['G_LAT','G_LONG']].dropna()
            geo = geo[geo.G_LAT.between(-90,90) & geo.G_LONG.between(-180,360)]
            summary['valid_geo_rows'] = len(geo)
            if len(geo):
                summary['latitude_min'] = geo.G_LAT.min()
                summary['latitude_max'] = geo.G_LAT.max()
                summary['longitude_min'] = geo.G_LONG.min()
                summary['longitude_max'] = geo.G_LONG.max()
        if 'G_ALT' in df:
            alt = df['G_ALT'].dropna()
            if len(alt):
                summary['altitude_min_m'] = alt.min()
                summary['altitude_max_m'] = alt.max()
        if {'U','V'}.issubset(df):
            wind = np.hypot(df['U'],df['V'])
            summary['wind_speed_median_mps'] = wind.median()
            summary['wind_speed_p95_mps'] = wind.quantile(.95)
        summaries.append(summary)
        # Store cleaned 1-Hz measurements, one file per flight. No mixing with 20-Hz data.
        df.to_csv(OUT / (path.stem+'_clean.csv'), index=False)
    except Exception as exc:
        print('  ERROR:', repr(exc))
        summaries.append({'file':path.name, 'error':repr(exc)})

qc = pd.DataFrame(summaries)
qc.to_csv(OUT/'flight_quality_summary.csv',index=False)
print('\nQUALITY SUMMARY:')
print(qc.to_string(index=False, max_cols=14))
print('\nOutputs:', OUT)

# Plot first successfully parsed flight with valid coordinates.
for path in files:
    clean = OUT/(path.stem+'_clean.csv')
    if not clean.exists():
        continue
    df = pd.read_csv(clean, usecols=lambda c:c in ['G_LAT','G_LONG','TIME_UTC','U','V'])
    if not {'G_LAT','G_LONG'}.issubset(df):
        continue
    coords = df[['G_LAT','G_LONG']].dropna()
    coords = coords[coords.G_LAT.between(-90,90) & coords.G_LONG.between(-180,360)]
    if len(coords)<100:
        continue
    fig, ax = plt.subplots(figsize=(9,6))
    ax.plot(coords.G_LONG,coords.G_LAT,lw=.6)
    ax.set(xlabel='Longitude (degrees)',ylabel='Latitude (degrees)',
           title=f'Observed Global Hawk ground track — {path.stem}')
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT/'first_flight_ground_track.png',dpi=180)
    plt.show()
    break
