
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
OUT = ROOT / 'analysis_step23'
SHORTLIST = ROOT / 'analysis_step21' / 'step21_candidate_shortlist.csv'
ERA5 = ROOT / 'ERA5_ATTREX_20140311_20140312_expanded.grib'
TARGETS = (2798, 3595)
SAMPLE_SECONDS = 60.0
CORRIDOR_MARGIN_DEG = 1.0  # bounding box screening, not proof of all possible alternative routes
EARTH_KM = 6371.0088


def locate_clean(filename):
    exact = ROOT / 'analysis_step2' / filename
    if exact.is_file():
        return exact
    matches = [p for p in ROOT.rglob(filename) if 'analysis_step23' not in p.parts]
    if not matches:
        raise FileNotFoundError(f'Cleaned 1 Hz source not found: {filename}; inspect analysis_step2')
    matches.sort(key=lambda p: (0 if 'analysis_step2' in p.parts else 1, len(p.parts)))
    return matches[0]


def lon_to_domain(values, domain):
    a = np.asarray(values, float)
    return np.mod(a, 360.0) if np.nanmin(domain) >= 0 and np.nanmax(domain) > 180 else (a + 180) % 360 - 180


def distance_km(lat1, lon1, lat2, lon2):
    a, b = np.deg2rad(np.asarray(lat1, float)), np.deg2rad(np.asarray(lat2, float))
    dlat = b - a
    dlon = np.deg2rad((np.asarray(lon2, float)-np.asarray(lon1, float)+180)%360-180)
    h = np.sin(dlat/2)**2 + np.cos(a)*np.cos(b)*np.sin(dlon/2)**2
    return 2*EARTH_KM*np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def extract_segment(row):
    path = locate_clean(str(row['file']))
    wanted = {'TIME_UTC','P','TAS','U','V','G_LAT','G_LONG','G_ALT','HDG','ROLL','PITCH'}
    df = pd.read_csv(path, usecols=lambda col: col in wanted, low_memory=False)
    needed = {'TIME_UTC','G_LAT','G_LONG','P','TAS'}
    if not needed.issubset(df):
        raise ValueError(f'{path} lacks required columns: {sorted(needed-set(df))}')
    for col in df:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.replace([np.inf,-np.inf], np.nan)
    # Segment IDs in the screening output are starting row indices; verify against
    # recorded time bounds rather than relying on index positions.
    start_s, end_s = float(row.start_s), float(row.end_s)
    df = df.loc[df.TIME_UTC.between(start_s-0.001, end_s+0.001)].copy()
    df = df.dropna(subset=['TIME_UTC','G_LAT','G_LONG','P','TAS']).sort_values('TIME_UTC')
    df = df.loc[df.TIME_UTC.diff().fillna(1).gt(0)]
    if len(df) < 2:
        raise ValueError(f'No usable observations in time window {start_s}..{end_s}')
    if abs(float(df.TIME_UTC.iloc[0])-start_s)>5 or abs(float(df.TIME_UTC.iloc[-1])-end_s)>5:
        raise ValueError('Extracted endpoints differ by >5 seconds from screening metadata')
    date = pd.Timestamp(str(row.flight_date), tz='UTC')
    df['datetime_utc'] = date + pd.to_timedelta(df.TIME_UTC, unit='s')
    return df, path


def thin_track(df):
    # Preserve first and last; roughly one observation per SAMPLE_SECONDS.
    t = df.TIME_UTC.to_numpy()
    idx = np.unique(np.r_[0, np.searchsorted(t, np.arange(t[0], t[-1], SAMPLE_SECONDS)), len(t)-1])
    return df.iloc[idx].copy()


def main():
    import xarray as xr
    if not SHORTLIST.exists():
        raise FileNotFoundError(f'Missing Step 21 shortlist: {SHORTLIST}')
    if not ERA5.exists():
        raise FileNotFoundError(f'Missing ERA5 GRIB: {ERA5}')
    OUT.mkdir(parents=True, exist_ok=True)
    shortlist = pd.read_csv(SHORTLIST)
    ds = xr.open_dataset(ERA5, engine='cfgrib', backend_kwargs={'indexpath':''})
    try:
        if not {'u','v'}.issubset(ds.data_vars):
            raise ValueError('ERA5 file must contain u and v wind components')
        lat_grid = np.asarray(ds.latitude.values, float)
        lon_grid = np.asarray(ds.longitude.values, float)
        time_grid = pd.to_datetime(ds.time.values, utc=True)
        p_grid = np.asarray(ds.isobaricInhPa.values, float)
        latlo, lathi = float(lat_grid.min()), float(lat_grid.max())
        lonlo, lonhi = float(lon_grid.min()), float(lon_grid.max())
        plo, phi = float(p_grid.min()), float(p_grid.max())
        tlo, thi = time_grid.min(), time_grid.max()
        results, tracks = [], {}
        print('Dataset variables:', list(ds.data_vars))
        print('Dataset dimensions:', dict(ds.sizes))
        print('ERA5:', ERA5)
        print(f'Domain: lat {latlo:g}..{lathi:g}; lon {lonlo:g}..{lonhi:g}; pressure {plo:g}..{phi:g} hPa')
        print('Time:', tlo, 'to', thi)
        for seg in TARGETS:
            match = shortlist.loc[shortlist.segment_id.eq(seg) & shortlist.flight_date.astype(str).eq('20140311')]
            if len(match)!=1:
                raise ValueError(f'Expected one March 11 segment {seg} in Step 21 shortlist, found {len(match)}')
            row = match.iloc[0]
            obs, source = extract_segment(row)
            obs_path = OUT/f'20140311_segment_{seg}_observations.csv'
            obs.to_csv(obs_path, index=False)
            sample = thin_track(obs)
            lat = sample.G_LAT.to_numpy(float)
            lon = lon_to_domain(sample.G_LONG.to_numpy(float), lon_grid)
            press = sample.P.to_numpy(float)
            times = sample.datetime_utc
            spatial = (lat>=latlo)&(lat<=lathi)&(lon>=lonlo)&(lon<=lonhi)
            full_lat = obs.G_LAT.to_numpy(float)
            full_lon = lon_to_domain(obs.G_LONG.to_numpy(float), lon_grid)
            full_p = obs.P.to_numpy(float)
            full_t = obs.datetime_utc
            full_spatial = (full_lat>=latlo)&(full_lat<=lathi)&(full_lon>=lonlo)&(full_lon<=lonhi)
            full_temporal = ((full_t>=tlo)&(full_t<=thi)).to_numpy()
            full_pressure = (full_p>=plo)&(full_p<=phi)
            temporal = ((times>=tlo)&(times<=thi)).to_numpy()
            pressure_ok = (press>=plo)&(press<=phi)
            valid = spatial & temporal & pressure_ok
            # Test actual ERA5 wind interpolation at sampled observed locations,
            # only for points inside the coordinate domain.
            interpolated = np.zeros(len(sample), dtype=bool)
            if valid.any():
                ix = np.flatnonzero(valid)
                try:
                    coords = xr.Dataset(coords={
                        'point': np.arange(len(ix)),
                        'time_q': ('point', times.iloc[ix].to_numpy(dtype='datetime64[ns]')),
                        'latitude_q': ('point', lat[ix]),
                        'longitude_q': ('point', lon[ix]),
                        'pressure_q': ('point', press[ix]),
                    })
                    for start in range(0, len(ix), 200):
                        sel = ix[start:start+200]
                        uv = ds[['u','v']].interp(
                            time=xr.DataArray(times.iloc[sel].to_numpy(dtype='datetime64[ns]'), dims='point'),
                            latitude=xr.DataArray(lat[sel], dims='point'),
                            longitude=xr.DataArray(lon[sel], dims='point'),
                            isobaricInhPa=xr.DataArray(press[sel], dims='point'),
                            method='linear')
                        interpolated[sel] = np.isfinite(uv.u.values) & np.isfinite(uv.v.values)
                except Exception as e:
                    print(f'  WARNING: wind interpolation check failed for segment {seg}: {e}')
            # End-to-end bounding box plus 1-degree margin is only a heuristic
            # and may not include the observed track or possible detours.
            margin_box_ok = (min(lat)-CORRIDOR_MARGIN_DEG>=latlo and
                             max(lat)+CORRIDOR_MARGIN_DEG<=lathi and
                             min(lon)-CORRIDOR_MARGIN_DEG>=lonlo and
                             max(lon)+CORRIDOR_MARGIN_DEG<=lonhi)
            dleg = distance_km(obs.G_LAT.to_numpy()[:-1],obs.G_LONG.to_numpy()[:-1],
                               obs.G_LAT.to_numpy()[1:],obs.G_LONG.to_numpy()[1:])
            obs_km = float(np.nansum(dleg))
            direct_km = float(distance_km(obs.G_LAT.iloc[0],obs.G_LONG.iloc[0],
                                          obs.G_LAT.iloc[-1],obs.G_LONG.iloc[-1]))
            data = dict(flight_date='20140311',segment_id=seg,source_file=str(source),
                        observations_csv=str(obs_path),rows=len(obs),sampled_rows=len(sample),
                        start_utc=str(obs.datetime_utc.iloc[0]),end_utc=str(obs.datetime_utc.iloc[-1]),
                        start_lat=float(obs.G_LAT.iloc[0]),start_lon=float(obs.G_LONG.iloc[0]),
                        end_lat=float(obs.G_LAT.iloc[-1]),end_lon=float(obs.G_LONG.iloc[-1]),
                        observed_km=obs_km,direct_km=direct_km,
                        observed_to_direct_ratio=obs_km/direct_km if direct_km>0 else np.nan,
                        median_pressure_hpa=float(obs.P.median()),median_tas_mps=float(obs.TAS.median()),
                        observed_lat_min=float(obs.G_LAT.min()),observed_lat_max=float(obs.G_LAT.max()),
                        observed_lon_min=float(lon.min()),observed_lon_max=float(lon.max()),
                        full_spatial_fraction=float(full_spatial.mean()),
                        full_temporal_fraction=float(full_temporal.mean()),
                        full_pressure_fraction=float(full_pressure.mean()),
                        full_joint_domain_fraction=float((full_spatial & full_temporal & full_pressure).mean()),
                        sampled_spatial_fraction=float(spatial.mean()),
                        sampled_temporal_fraction=float(temporal.mean()),
                        sampled_pressure_fraction=float(pressure_ok.mean()),
                        sampled_joint_domain_fraction=float(valid.mean()),
                        sampled_finite_wind_fraction=float(interpolated.mean()),
                        entire_sample_within_domain=bool(valid.all()),
                        sampled_wind_all_finite=bool(interpolated.all()),
                        observed_track_margin_box_ok=bool(margin_box_ok),
                        meteorology_coverage_pass=bool(full_spatial.all() and full_temporal.all() and
                                                       full_pressure.all() and interpolated.all() and margin_box_ok),
                        routing_ready_for_mission_aware_comparison=False)
            results.append(data)
            tracks[seg] = (lat, lon, spatial)
            sample['era5_spatial_ok'] = spatial
            sample['era5_time_ok'] = temporal
            sample['era5_pressure_ok'] = pressure_ok
            sample['era5_wind_finite'] = interpolated
            sample.to_csv(OUT/f'20140311_segment_{seg}_era5_sample_audit.csv',index=False)
            print(f'\nSegment {seg}: {len(obs)} rows, {len(sample)} audit points')
            print(f'  Observed/direct distance: {obs_km:.1f}/{direct_km:.1f} km')
            print(f'  Full-track spatial/time/pressure: {full_spatial.mean():.1%}/{full_temporal.mean():.1%}/{full_pressure.mean():.1%}')
            print(f'  Sampled spatial/time/pressure: {spatial.mean():.1%}/{temporal.mean():.1%}/{pressure_ok.mean():.1%}')
            print(f'  Finite interpolated ERA5 winds: {interpolated.mean():.1%}')
            print(f'  1-degree observed-track margin inside domain: {margin_box_ok}')
        summary = pd.DataFrame(results)
        summary.to_csv(OUT/'step23_era5_coverage_summary.csv',index=False)
        fig,ax=plt.subplots(figsize=(10,7))
        ax.fill([lonlo,lonhi,lonhi,lonlo],[latlo,latlo,lathi,lathi],
                alpha=.07,color='gray',label='ERA5 domain')
        ax.plot([lonlo,lonhi,lonhi,lonlo,lonlo],
                [latlo,latlo,lathi,lathi,latlo],color='black',lw=1.5)
        for seg,(lat,lon,spatial) in tracks.items():
            ax.plot(lon,lat,lw=1.5,label=f'Segment {seg} observed track')
            if (~spatial).any():
                ax.scatter(lon[~spatial],lat[~spatial],s=9,marker='x',label=f'{seg}: outside ERA5')
            ax.scatter([lon[0],lon[-1]],[lat[0],lat[-1]],s=25)
        ax.set(xlabel='Longitude (degrees E)',ylabel='Latitude (degrees N)',
               title='ATTREX Step 23 | expanded ERA5 domain verification')
        ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
        fig.savefig(OUT/'step23_trajectory_era5_coverage.png',dpi=180)
        plt.close(fig)
        print('\nSaved:',OUT)
        print('CAUTION: meteorology coverage pass is not a routing or mission feasibility certification.')
        print('CAUTION: pressure outside ERA5 levels is flagged, never silently extrapolated.')
        print('CAUTION: no optimization or inferred savings; out-of-domain segments require more ERA5.')
    finally:
        ds.close()

if __name__=='__main__':
    main()
