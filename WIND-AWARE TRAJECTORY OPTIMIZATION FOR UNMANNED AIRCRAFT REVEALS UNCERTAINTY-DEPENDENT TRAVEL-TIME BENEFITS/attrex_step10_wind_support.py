
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
INPUT = ROOT / 'analysis_step9'
OUTPUT = ROOT / 'analysis_step10'
OUTPUT.mkdir(parents=True, exist_ok=True)
FILES = ['20150305_segment_1925_observations.csv',
         '20140311_segment_1986_observations.csv',
         '20140311_segment_1450_observations.csv']
R = 6371.0088
rows = []
for name in FILES:
    f = INPUT / name
    if not f.exists():
        raise FileNotFoundError(f'Expected Step 9 output: {f}')
    d = pd.read_csv(f)
    required = ['TIME_UTC','G_LAT','G_LONG','G_ALT','U','V','TAS']
    missing = [c for c in required if c not in d.columns]
    if missing: raise ValueError(f'{name}: missing columns {missing}')
    d = d.replace([np.inf, -np.inf], np.nan)
    valid = d[required].notna().all(axis=1)
    d = d.loc[valid].sort_values('TIME_UTC').copy()
    if len(d)<3: raise ValueError(f'{name}: insufficient valid data')
    lat = np.deg2rad(d.G_LAT.to_numpy())
    lon = np.unwrap(np.deg2rad(d.G_LONG.to_numpy()))
    t = d.TIME_UTC.to_numpy()
    dt = np.diff(t)
    east = R * 1000 * np.cos((lat[1:]+lat[:-1])/2)*np.diff(lon)
    north = R * 1000 * np.diff(lat)
    good = (dt>0.5)&(dt<1.5)
    ve = np.divide(east,dt,out=np.full_like(east,np.nan),where=good)
    vn = np.divide(north,dt,out=np.full_like(north,np.nan),where=good)
    u = d.U.to_numpy(); v=d.V.to_numpy(); tas=d.TAS.to_numpy()
    air = np.hypot(ve-u[:-1],vn-v[:-1])
    err = np.abs(air-tas[:-1]); err = err[np.isfinite(err)]
    wind = np.hypot(u,v)
    # Heading of observed ground movement; head/tail component of observed wind
    speed = np.hypot(ve,vn)
    along = np.divide(u[:-1]*ve+v[:-1]*vn,speed,out=np.full_like(speed,np.nan),where=speed>1)
    # Local Cartesian approximation for visualization only
    x = R*np.cos(np.mean(lat))*(lon-lon[0]); y=R*(lat-lat[0])
    # Observational sampling extent is a track, not a 2D gridded wind field
    sample_idx=np.linspace(0,len(d)-1,min(200,len(d)),dtype=int)
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    sc=axs[0].scatter(x,y,c=wind,s=2,cmap='viridis',rasterized=True)
    axs[0].quiver(x[sample_idx],y[sample_idx],u[sample_idx],v[sample_idx],
                  angles='xy',scale_units='xy',scale=30,width=.0025,color='black',alpha=.45)
    axs[0].set(xlabel='Approx. east displacement (km)',ylabel='Approx. north displacement (km)',
               title='Observed wind only along flight track')
    axs[0].set_aspect('equal',adjustable='datalim'); fig.colorbar(sc,ax=axs[0],label='Observed wind (m/s)')
    axs[1].plot((t-t[0])/60,wind,label='Wind magnitude')
    axs[1].plot((t-t[0])/60,tas,label='True airspeed',alpha=.8)
    axs[1].set(xlabel='Elapsed time (min)',ylabel='Speed (m/s)',title='Time-varying observed conditions')
    axs[1].legend()
    fig.suptitle(name.replace('_observations.csv',''))
    fig.savefig(OUTPUT/name.replace('_observations.csv','_wind_support.png'),dpi=170)
    plt.close(fig)
    rows.append(dict(segment=name.replace('_observations.csv',''),n_valid=len(d),
                     duration_min=(t[-1]-t[0])/60,median_altitude_m=np.median(d.G_ALT),
                     median_tas_mps=np.median(tas),median_wind_mps=np.median(wind),
                     p95_wind_mps=np.percentile(wind,95),
                     median_along_track_wind_mps=np.nanmedian(along),
                     median_abs_tas_consistency_error_mps=np.median(err) if len(err) else np.nan,
                     note='Observed wind sampled only along actual aircraft track'))
    print(f'{name}: {len(d):,} valid rows; median wind {np.median(wind):.2f} m/s; '
          f'median altitude {np.median(d.G_ALT):.0f} m')
pd.DataFrame(rows).to_csv(OUTPUT/'wind_observation_support_summary.csv',index=False)
print('Saved:',OUTPUT)
print('NOTE: This is observational support/QC, NOT a reconstructed 2D/3D wind field.')
