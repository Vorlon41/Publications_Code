
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
CLEAN = ROOT / 'analysis_step2'
OUT = ROOT / 'analysis_step3'
OUT.mkdir(exist_ok=True)
REQUIRED = ['TIME_UTC', 'G_LAT', 'G_LONG', 'G_ALT', 'TAS', 'U', 'V']
R = 6371000.0

def angular_diff(a, b):
    return (a - b + 180) % 360 - 180

def inspect(csv):
    df = pd.read_csv(csv, low_memory=False)
    if not set(REQUIRED).issubset(df.columns):
        return None, f'missing {set(REQUIRED)-set(df.columns)}'
    for col in REQUIRED + [c for c in ['HDG','ROLL','PITCH'] if c in df]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('TIME_UTC').drop_duplicates('TIME_UTC').reset_index(drop=True)
    # ICARTT time is elapsed seconds since UTC midnight; do not wrap at 86400.
    valid = df[REQUIRED].notna().all(axis=1) & df.G_LAT.between(-90,90) & df.G_LONG.between(-180.01,180.01) & (df.TAS > 20) & (df.TAS < 400)
    df['valid'] = valid
    lat = np.radians(df.G_LAT.to_numpy(dtype=float))
    lon = np.radians(df.G_LONG.to_numpy(dtype=float))
    dlat = np.diff(lat)
    dlon = (np.diff(lon)+np.pi)%(2*np.pi)-np.pi
    a = np.sin(dlat/2)**2 + np.cos(lat[:-1])*np.cos(lat[1:])*np.sin(dlon/2)**2
    dist = 2*R*np.arcsin(np.sqrt(np.clip(a,0,1)))
    dt = np.diff(df.TIME_UTC.to_numpy(dtype=float))
    speed = np.divide(dist,dt,out=np.full_like(dist,np.nan),where=dt>0)
    north = R*dlat
    east = R*np.cos((lat[:-1]+lat[1:])/2)*dlon
    vn = np.divide(north,dt,out=np.full_like(north,np.nan),where=dt>0)
    ve = np.divide(east,dt,out=np.full_like(east,np.nan),where=dt>0)
    ground_course = (np.degrees(np.arctan2(ve,vn))+360)%360
    # Difference between observed ground velocity and wind approximates air-relative velocity.
    ae = ve-df.U.to_numpy(dtype=float)[1:]
    an = vn-df.V.to_numpy(dtype=float)[1:]
    reconstructed_tas = np.hypot(ae,an)
    tas_error = reconstructed_tas-df.TAS.to_numpy(dtype=float)[1:]
    pair_valid = valid.to_numpy()[1:] & valid.to_numpy()[:-1] & (dt >= .5) & (dt <= 2) & (speed < 350)
    df['groundspeed_mps'] = np.r_[np.nan,speed]
    df['course_deg'] = np.r_[np.nan,ground_course]
    df['tas_reconstruction_error_mps'] = np.r_[np.nan,tas_error]
    df['pair_valid'] = np.r_[False,pair_valid]
    # Candidate cruise segments: high altitude, continuous observations, plausible kinematics.
    cruise = pair_valid & (df.G_ALT.to_numpy()[1:] >= 14000) & (df.G_ALT.to_numpy()[:-1] >= 14000)
    # Segment boundaries at any non-cruise pair; retain continuous runs >= 10 minutes.
    segment_ids = np.cumsum(~np.r_[False,cruise])
    df['segment_id'] = segment_ids
    df['cruise'] = np.r_[False,cruise]
    segments = []
    for sid, group in df[df.cruise].groupby('segment_id'):
        if len(group)<600: continue
        segments.append(dict(file=csv.name, segment_id=int(sid), rows=len(group), start_s=float(group.TIME_UTC.iloc[0]), end_s=float(group.TIME_UTC.iloc[-1]), duration_min=float((group.TIME_UTC.iloc[-1]-group.TIME_UTC.iloc[0])/60), start_lat=float(group.G_LAT.iloc[0]),start_lon=float(group.G_LONG.iloc[0]),end_lat=float(group.G_LAT.iloc[-1]),end_lon=float(group.G_LONG.iloc[-1]),median_alt_m=float(group.G_ALT.median()),median_wind_mps=float(np.hypot(group.U,group.V).median()),median_tas_error_mps=float(group.tas_reconstruction_error_mps.median())))
    mask = df.pair_valid & df.tas_reconstruction_error_mps.notna()
    err = df.loc[mask,'tas_reconstruction_error_mps'].to_numpy()
    summary = dict(file=csv.name, total_rows=len(df), valid_pct=round(100*valid.mean(),2), valid_motion_pairs=int(pair_valid.sum()), median_abs_tas_error_mps=float(np.median(np.abs(err))) if len(err) else np.nan, p95_abs_tas_error_mps=float(np.percentile(np.abs(err),95)) if len(err) else np.nan, cruise_segments_10min=len(segments))
    return (df,summary,segments),None

files = sorted(CLEAN.rglob('*.csv'))
# Exclude summary files and any prior outputs.
files = [p for p in files if 'summary' not in p.name.lower() and 'inventory' not in p.name.lower() and 'quality' not in p.name.lower()]
print(f'Found {len(files)} candidate CSVs under {CLEAN}')
all_summary,all_segments=[],[]
example=None
for p in files:
    try:
        result,error=inspect(p)
        if error:
            print('SKIP',p.name,error)
            continue
        df,summary,segments=result
        all_summary.append(summary)
        all_segments.extend(segments)
        print('OK',p.name,'valid pairs:',summary['valid_motion_pairs'],'cruise segments:',len(segments))
        if example is None and '20131220' in p.name: example=df
    except Exception as exc:
        print('ERROR',p.name,repr(exc))
if not all_summary:
    print('No compatible cleaned CSV files found. Please share the names and columns in analysis_step2.')
else:
    pd.DataFrame(all_summary).to_csv(OUT/'kinematic_quality_summary.csv',index=False)
    pd.DataFrame(all_segments).to_csv(OUT/'candidate_cruise_segments.csv',index=False)
    if example is not None:
        fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
        axes[0].plot(example.TIME_UTC/3600,example.G_ALT/1000,lw=.7)
        axes[0].axhline(14,color='gray',ls='--',lw=.8)
        axes[0].set_ylabel('Altitude (km)')
        axes[1].plot(example.TIME_UTC/3600,np.hypot(example.U,example.V),lw=.7)
        axes[1].set_ylabel('Wind speed (m/s)')
        axes[1].set_xlabel('UTC elapsed hours')
        fig.suptitle('ATTREX 2013-12-20: altitude and observed wind')
        fig.tight_layout()
        fig.savefig(OUT/'example_altitude_wind.png',dpi=180)
        plt.close(fig)
    print('\nSaved to',OUT)
    print(pd.DataFrame(all_summary).to_string(index=False))
    print('\nNumber of candidate cruise segments:',len(all_segments))
