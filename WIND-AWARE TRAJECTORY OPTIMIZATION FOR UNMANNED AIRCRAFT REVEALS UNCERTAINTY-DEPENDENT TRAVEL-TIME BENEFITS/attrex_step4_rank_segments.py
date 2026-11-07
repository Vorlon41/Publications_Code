
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
CLEAN = ROOT / 'analysis_step2'
SEGMENTS = ROOT / 'analysis_step3' / 'candidate_cruise_segments.csv'
OUT = ROOT / 'analysis_step4'
OUT.mkdir(parents=True, exist_ok=True)
R = 6371.0088  # km

def haversine(lat1, lon1, lat2, lon2):
    a1, a2 = np.radians(np.asarray(lat1, dtype=float)), np.radians(np.asarray(lat2, dtype=float))
    dlat = a2 - a1
    dlon = (np.radians(np.asarray(lon2, dtype=float)-np.asarray(lon1, dtype=float)) + np.pi) % (2*np.pi) - np.pi
    h = np.sin(dlat/2)**2 + np.cos(a1)*np.cos(a2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(h,0,1)))

if not SEGMENTS.exists():
    raise FileNotFoundError(f'Missing {SEGMENTS}; run Step 3 first')
segments = pd.read_csv(SEGMENTS)
records = []
for _, s in segments.iterrows():
    src = CLEAN / str(s['file'])
    if not src.exists():
        print('SKIP missing file:',src)
        continue
    # Step 3 segment rows are the qualifying pair endpoints. Reconstruct same ID logic.
    df = pd.read_csv(src, usecols=lambda c: c in ['TIME_UTC','G_LAT','G_LONG','G_ALT','TAS','U','V'], low_memory=False)
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.sort_values('TIME_UTC').drop_duplicates('TIME_UTC').reset_index(drop=True)
    # Use time range from Step 3, not only segment_id, for exact compatibility.
    g = df.loc[df.TIME_UTC.between(float(s.start_s), float(s.end_s))].copy()
    g = g.dropna(subset=['TIME_UTC','G_LAT','G_LONG','G_ALT','TAS','U','V'])
    if len(g) < 2: continue
    step_km = haversine(g.G_LAT.to_numpy()[:-1],g.G_LONG.to_numpy()[:-1],g.G_LAT.to_numpy()[1:],g.G_LONG.to_numpy()[1:])
    observed_km = float(np.sum(step_km))
    direct_km = float(haversine(g.G_LAT.iloc[0],g.G_LONG.iloc[0],g.G_LAT.iloc[-1],g.G_LONG.iloc[-1]))
    wind = np.hypot(g.U.to_numpy(),g.V.to_numpy())
    time_diff = np.diff(g.TIME_UTC.to_numpy())
    # Turn proxy based on changes in geodesic course, downsampled to 60-s intervals.
    sub = g.iloc[::60].copy()
    if len(sub) >= 3:
        lat = np.radians(sub.G_LAT.to_numpy()); lon = np.radians(sub.G_LONG.to_numpy())
        dlon = (np.diff(lon)+np.pi)%(2*np.pi)-np.pi
        y = np.sin(dlon)*np.cos(lat[1:])
        x = np.cos(lat[:-1])*np.sin(lat[1:])-np.sin(lat[:-1])*np.cos(lat[1:])*np.cos(dlon)
        bearings = np.arctan2(y,x)
        turns = (np.diff(bearings)+np.pi)%(2*np.pi)-np.pi
        median_turn_deg = float(np.median(np.abs(np.degrees(turns))))
    else: median_turn_deg = np.nan
    records.append({
        'file':s['file'], 'segment_id':int(s.segment_id), 'start_s':float(s.start_s), 'end_s':float(s.end_s),
        'duration_min':float(s.duration_min), 'samples':len(g),
        'observed_distance_km':observed_km, 'direct_distance_km':direct_km,
        'route_ratio':observed_km/direct_km if direct_km > 1 else np.nan,
        'median_altitude_m':float(g.G_ALT.median()),
        'median_tas_mps':float(g.TAS.median()),
        'median_wind_mps':float(np.median(wind)),
        'p95_wind_mps':float(np.percentile(wind,95)),
        'wind_speed_sd_mps':float(np.std(wind)),
        'median_60s_turn_deg':median_turn_deg,
        'max_time_gap_s':float(np.max(time_diff)),
        'start_lat':float(g.G_LAT.iloc[0]),'start_lon':float(g.G_LONG.iloc[0]),
        'end_lat':float(g.G_LAT.iloc[-1]),'end_lon':float(g.G_LONG.iloc[-1]),
    })

out = pd.DataFrame(records)
if out.empty: raise RuntimeError('No segments processed; inspect Step 3 file paths and columns')
# Focus on substantial, continuous point-to-point cruise legs; avoid orbit-heavy tracks.
out['eligible_initial'] = ((out.direct_distance_km >= 100) & (out.duration_min >= 20) &
                           (out.route_ratio <= 2.0) & (out.max_time_gap_s <= 2.0))
# Descriptive ranking only; not a scientific performance score.
out['ranking_score'] = (out.direct_distance_km.clip(upper=1000)/1000 +
                        out.wind_speed_sd_mps.clip(upper=20)/20 -
                        (out.route_ratio-1).clip(lower=0,upper=2)/2)
out = out.sort_values(['eligible_initial','ranking_score'],ascending=[False,False])
out.to_csv(OUT/'segment_screening.csv',index=False)
print('Total segments:',len(out))
print('Initially eligible point-to-point segments:',int(out.eligible_initial.sum()))
cols = ['file','segment_id','duration_min','direct_distance_km','route_ratio',
        'median_wind_mps','wind_speed_sd_mps','max_time_gap_s','eligible_initial']
print('\nTOP CANDIDATES:')
print(out[cols].head(15).round(2).to_string(index=False))
print('\nSaved:',OUT/'segment_screening.csv')

# Plot the highest-ranked segment, with longitude unwrapped for date-line continuity.
best = out.iloc[0]
g = pd.read_csv(CLEAN/str(best.file),usecols=['TIME_UTC','G_LAT','G_LONG','U','V'])
g = g[g.TIME_UTC.between(best.start_s,best.end_s)].dropna()
lon_unwrapped = np.degrees(np.unwrap(np.radians(g.G_LONG.to_numpy())))
fig, ax = plt.subplots(figsize=(10,6))
sc = ax.scatter(lon_unwrapped[::30],g.G_LAT.to_numpy()[::30],
                c=np.hypot(g.U.to_numpy()[::30],g.V.to_numpy()[::30]),
                s=5,cmap='viridis')
fig.colorbar(sc,ax=ax,label='Observed wind speed (m/s)')
ax.set(xlabel='Unwrapped longitude (degrees)',ylabel='Latitude (degrees)',
       title=f'Candidate segment: {best.file}, segment {best.segment_id}')
fig.tight_layout()
fig.savefig(OUT/'top_candidate_track.png',dpi=180)
plt.close(fig)
print('Saved:',OUT/'top_candidate_track.png')
print('\nNOTE: Wind values are only observed along the actual flight path. Route optimization requires a defensible wind field beyond that path.')
