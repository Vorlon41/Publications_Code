from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
OBS = ROOT / 'analysis_step23' / '20140311_segment_2798_observations.csv'
GRIB = ROOT / 'ERA5_ATTREX_20140311_20140312_expanded.grib'
OUT = ROOT / 'analysis_step25'
OFFSETS_KM = (-100, -75, -50, -25, 0, 25, 50, 75, 100)
ALONG_FRACTIONS = np.linspace(0.1, 0.9, 9)
EARTH_KM = 6371.0088

def great_circle_km(a, b):
    a, b = np.radians(np.asarray(a, float)), np.radians(np.asarray(b, float))
    h = np.sin((b[0]-a[0])/2)**2 + np.cos(a[0])*np.cos(b[0])*np.sin((b[1]-a[1])/2)**2
    return float(2*EARTH_KM*np.arcsin(np.sqrt(np.clip(h,0,1))))

def main():
    for p in (OBS, GRIB):
        if not p.exists(): raise FileNotFoundError(str(p))
    OUT.mkdir(parents=True, exist_ok=True)
    obs = pd.read_csv(OBS).sort_values('datetime_utc').dropna(subset=['G_LAT','G_LONG','P','TAS'])
    ds = xr.open_dataset(GRIB, engine='cfgrib', backend_kwargs={'indexpath': ''})
    ds = ds.sortby('time').sortby('latitude').sortby('longitude').sortby('isobaricInhPa')
    pressure = float(obs.P.median()); tas = float(obs.TAS.median())
    start = obs[['G_LAT','G_LONG']].iloc[0].to_numpy(float)
    end = obs[['G_LAT','G_LONG']].iloc[-1].to_numpy(float)
    t0 = pd.to_datetime(obs.datetime_utc.iloc[0], utc=True)
    t1 = pd.to_datetime(obs.datetime_utc.iloc[-1], utc=True)
    tm = t0 + (t1-t0)/2
    tsec = pd.to_datetime(ds.time.values, utc=True).astype('int64').to_numpy()/1e9
    midsec = tm.timestamp()
    if not (tsec[0] <= midsec <= tsec[-1]):
        raise ValueError('Mid-flight time outside ERA5 coverage')
    pds = ds[['u','v']].interp(isobaricInhPa=pressure)
    lat_grid = np.asarray(ds.latitude.values, float)
    lon_grid = np.asarray(ds.longitude.values, float)
    interp_u = RegularGridInterpolator((tsec,lat_grid,lon_grid), np.asarray(pds.u.values,float), bounds_error=False,fill_value=np.nan)
    interp_v = RegularGridInterpolator((tsec,lat_grid,lon_grid), np.asarray(pds.v.values,float), bounds_error=False,fill_value=np.nan)
    lat_ref = float((start[0]+end[0])/2)
    km_lat = 111.195
    km_lon = km_lat*np.cos(np.deg2rad(lat_ref))
    dx = (end[1]-start[1])*km_lon
    dy = (end[0]-start[0])*km_lat
    xy_length = np.hypot(dx,dy)
    along = np.array([dx,dy])/xy_length
    left = np.array([-along[1],along[0]])
    direct_km = great_circle_km(start,end)
    records=[]
    for fraction in ALONG_FRACTIONS:
        center = np.array([fraction*dx, fraction*dy])
        for offset in OFFSETS_KM:
            xy = center + offset*left
            lat = start[0]+xy[1]/km_lat
            lon = start[1]+xy[0]/km_lon
            lon_query = lon % 360 if lon_grid.min()>=0 else (lon+180)%360-180
            point = np.array([[midsec,lat,lon_query]])
            u = float(interp_u(point)[0]); v = float(interp_v(point)[0])
            tail = u*along[0]+v*along[1]
            cross = u*left[0]+v*left[1]
            feasible = bool(np.isfinite(u) and np.isfinite(v) and abs(cross)<tas)
            ground_speed = float(np.sqrt(tas**2-cross**2)+tail) if feasible else np.nan
            records.append(dict(fraction=float(fraction),offset_km=float(offset),latitude=lat,longitude=lon,
                                era5_u_mps=u,era5_v_mps=v,tailwind_mps=tail,crosswind_mps=cross,
                                modeled_groundspeed_mps=ground_speed,wind_feasible=feasible))
    corridor = pd.DataFrame(records)
    corridor.to_csv(OUT/'step25_corridor_wind_samples.csv',index=False)
    center = corridor[corridor.offset_km==0].set_index('fraction')
    paired=[]
    for off in OFFSETS_KM:
        if off==0:continue
        side=corridor[corridor.offset_km==off].set_index('fraction')
        diff=side.tailwind_mps-center.tailwind_mps
        paired.append(dict(offset_km=off,n_valid=int(diff.notna().sum()),
                           mean_tailwind_difference_mps=float(diff.mean()),
                           max_tailwind_difference_mps=float(diff.max()),
                           min_tailwind_difference_mps=float(diff.min()),
                           mean_abs_crosswind_mps=float(side.crosswind_mps.abs().mean())))
    summary=pd.DataFrame(paired)
    summary.to_csv(OUT/'step25_corridor_summary.csv',index=False)
    obsxy=obs[['G_LAT','G_LONG']].to_numpy(float)
    observed_km=sum(great_circle_km(a,b) for a,b in zip(obsxy[:-1],obsxy[1:]))
    overview=pd.DataFrame([dict(segment=2798,n_observations=len(obs),start_utc=str(t0),end_utc=str(t1),
                                era5_sample_utc=str(tm),median_pressure_hpa=pressure,median_tas_mps=tas,
                                observed_distance_km=observed_km,direct_distance_km=direct_km,
                                observed_to_direct_ratio=observed_km/direct_km,
                                valid_corridor_samples=int(corridor.wind_feasible.sum()),
                                total_corridor_samples=len(corridor),
                                era5_last_timestamp_utc=str(pd.to_datetime(ds.time.values[-1],utc=True)))])
    overview.to_csv(OUT/'step25_geometry_and_support.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,6))
    ax.plot(obs.G_LONG,obs.G_LAT,color='0.6',lw=1.2,label='Observed trajectory')
    ax.plot([start[1],end[1]],[start[0],end[0]],'k--',lw=1.3,label='Direct endpoint connection')
    sc=ax.scatter(corridor.longitude,corridor.latitude,c=corridor.tailwind_mps,s=35,cmap='viridis',label='Corridor samples')
    fig.colorbar(sc,ax=ax,label='ERA5 along-route tailwind (m/s)')
    ax.set(xlabel='Longitude (degrees E)',ylabel='Latitude (degrees N)',title='Step 25 | segment 2798 | sampled cross-track wind corridor')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'step25_corridor_map.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for off,g in corridor.groupby('offset_km'):
        ax.plot(g.fraction,g.tailwind_mps,label=f'{off:+g} km',lw=1.4)
    ax.set(xlabel='Fraction along direct corridor',ylabel='ERA5 along-route wind (m/s)',
           title='Step 25 | fixed-time cross-track tailwind sensitivity')
    ax.legend(ncol=3,fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'step25_tailwind_profiles.png',dpi=180);plt.close(fig)
    print('STEP 25 — Segment 2798 corridor screening')
    print(overview.to_string(index=False))
    print('\nCross-track tailwind differences versus centerline (not travel-time savings):')
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\nSaved:',OUT)
    print('CAUTION: samples at one fixed midpoint time; no route search or travel-time optimization.')
    print('CAUTION: cross-track offsets are not flight-feasible paths; turns, mission, airspace not evaluated.')
    print('CAUTION: along-track ERA5 agreement does not validate off-track wind estimates.')
    print('CAUTION: no inferred operational time or fuel savings.')

if __name__=='__main__':
    main()
