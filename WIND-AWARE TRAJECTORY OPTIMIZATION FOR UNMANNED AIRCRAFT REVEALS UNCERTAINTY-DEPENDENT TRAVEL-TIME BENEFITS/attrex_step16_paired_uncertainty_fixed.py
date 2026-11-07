
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
OUT = ROOT / 'analysis_step16'
OUT.mkdir(parents=True, exist_ok=True)
GRIB = ROOT / '4218a346c038961a43934d722972a5ad.grib'
G = 9.80665
EARTH = 6371008.8
N_SIM = 300
SEED = 16025
# Spatial/temporal scales are ASSUMED scenarios, not fitted from off-track data.
SCENARIOS = {'short_25km_10min': (25., 10.), 'medium_100km_30min': (100., 30.), 'long_300km_120min': (300., 120.)}


def haversine(a, b):
    la1, lo1 = np.deg2rad(a); la2, lo2 = np.deg2rad(b)
    dl = la2-la1; dn = lo2-lo1
    h = np.sin(dl/2)**2 + np.cos(la1)*np.cos(la2)*np.sin(dn/2)**2
    return 2*EARTH*np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def leg_geometry(a, b):
    lat = np.deg2rad((a[0]+b[0])/2)
    north = EARTH*np.deg2rad(b[0]-a[0]); east = EARTH*np.cos(lat)*np.deg2rad(b[1]-a[1])
    dist = haversine(a,b)
    return dist, np.array([east,north]) / max(np.hypot(east,north), 1e-10)


def load_residuals(seg):
    if seg == 1450:
        p = ROOT/'analysis_step11b'/'20140311_segment_1450_era5_collocated.csv'
    else:
        p = ROOT/'analysis_step11b'/'20140311_segment_1986_era5_collocated.csv'
    if not p.exists():
        raise FileNotFoundError(f'Missing Step 11B file: {p}')
    d = pd.read_csv(p)
    du = pd.to_numeric(d['u_error_mps'],errors='coerce').to_numpy()
    dv = pd.to_numeric(d['v_error_mps'],errors='coerce').to_numpy()
    t = pd.to_numeric(d['TIME_UTC'],errors='coerce').to_numpy()
    good = np.isfinite(du)&np.isfinite(dv)&np.isfinite(t)
    d = d.loc[good].copy(); d['du'] = du[good]; d['dv'] = dv[good]; d['block'] = np.floor((t[good]-t[good].min())/300).astype(int)
    blocks = d.groupby('block')[['du','dv']].mean().to_numpy()
    if len(blocks) < 5: raise ValueError('Too few residual blocks')
    return blocks


def load_routes(seg):
    routes = {}
    for name, suffix in [('distance','distance'),('wind_aware','wind_aware')]:
        p = ROOT/'analysis_step14'/f'step14_segment_{seg}_{suffix}_route.csv'
        if not p.exists(): raise FileNotFoundError(f'Missing Step 14 route: {p}')
        a = pd.read_csv(p)[['latitude','longitude']].to_numpy(dtype=float)
        if len(a)<2: raise ValueError('Route has fewer than two waypoints')
        routes[name]=a
    return routes


def interp_at(ds, when, lat, lon, pressure):
    # Time-dependent ERA5, fixed pressure as in Step 14. No extrapolation.
    vals = ds[['u','v']].interp(time=np.datetime64(pd.Timestamp(when).tz_localize(None).to_datetime64(), 'ns'),
         isobaricInhPa=pressure, latitude=lat, longitude=lon, method='linear')
    return np.array([float(vals.u.values),float(vals.v.values)])


def make_field(rng, blocks, scale_km, scale_min):
    # Shared spatiotemporal random field: random Fourier features, common to BOTH routes.
    # Component covariance estimated from 5-min block means; bias preserved separately.
    mean = blocks.mean(axis=0)
    centered = blocks - mean
    cov = np.cov(centered.T)
    cov = cov + np.eye(2)*0.01
    L = np.linalg.cholesky(cov)
    nfeat = 96
    # Coordinates: east/north km in local tangent plane, time minutes.
    frequencies = rng.normal(size=(nfeat,3))
    phases = rng.uniform(0,2*np.pi,nfeat)
    weights = rng.normal(size=(nfeat,2)) @ L.T
    def evaluate(lat,lon,minute):
        x = np.array([(lon-147)*93./scale_km,(lat-23)*111./scale_km,minute/scale_min])
        phi = np.cos(frequencies @ x + phases)*np.sqrt(2/nfeat)
        return mean + phi @ weights
    return evaluate


def route_time(ds, coords, depart, tas, pressure, error_field=None):
    elapsed=0.; distance=0.
    for a,b in zip(coords[:-1],coords[1:]):
        dist, direction = leg_geometry(a,b)
        if dist < 1: continue
        # Midpoint predictor-corrector for time-dependent wind.
        lat,lon = (a+b)/2
        when = depart + pd.Timedelta(seconds=elapsed+dist/(2*tas))
        wind = interp_at(ds,when,lat,lon,pressure)
        if error_field is not None: wind += error_field(lat,lon,(elapsed+dist/(2*tas))/60)
        groundspeed = tas + np.dot(wind,direction)
        if not np.isfinite(groundspeed) or groundspeed<=5: return np.nan,np.nan
        dt = dist/groundspeed
        when = depart + pd.Timedelta(seconds=elapsed+dt/2)
        wind = interp_at(ds,when,lat,lon,pressure)
        if error_field is not None: wind += error_field(lat,lon,(elapsed+dt/2)/60)
        groundspeed = tas + np.dot(wind,direction)
        if not np.isfinite(groundspeed) or groundspeed<=5: return np.nan,np.nan
        elapsed += dist/groundspeed
        distance += dist
    return elapsed/60,distance/1000



def cached_legs(ds, coords, depart, tas, pressure):
    """Cache nominal leg-midpoint ERA5 winds and nominal arrival times.

    The perturbation experiment keeps these nominal sample times fixed.
    """
    legs = []
    elapsed = 0.0
    for a, b in zip(coords[:-1], coords[1:]):
        dist, direction = leg_geometry(a, b)
        if dist < 1:
            continue
        lat, lon = (a + b) / 2
        # Two-pass midpoint estimate, consistent with route_time().
        dt = dist / tas
        for _ in range(2):
            mid_s = elapsed + dt / 2
            when = depart + pd.Timedelta(seconds=mid_s)
            wind = interp_at(ds, when, lat, lon, pressure)
            speed = tas + float(np.dot(wind, direction))
            if not np.isfinite(speed) or speed <= 5:
                raise ValueError('Invalid nominal ground speed while caching legs')
            dt = dist / speed
        mid_s = elapsed + dt / 2
        when = depart + pd.Timedelta(seconds=mid_s)
        wind = interp_at(ds, when, lat, lon, pressure)
        if not np.all(np.isfinite(wind)):
            raise ValueError('Nonfinite ERA5 wind in cached leg')
        legs.append((dist, direction, float(lat), float(lon), mid_s / 60, wind))
        elapsed += dt
    return legs


def cached_route_time(legs, tas, error_field):
    """Evaluate a fixed route using cached nominal winds and shared perturbations."""
    elapsed = 0.0
    for dist, direction, lat, lon, nominal_minute, wind in legs:
        perturbed = wind + error_field(lat, lon, nominal_minute)
        speed = tas + float(np.dot(perturbed, direction))
        if not np.isfinite(speed) or speed <= 5:
            return np.nan
        elapsed += dist / speed
    return elapsed / 60


def main():
    if not GRIB.exists(): raise FileNotFoundError(f'ERA5 file not found: {GRIB}')
    ds = xr.open_dataset(GRIB, engine='cfgrib', backend_kwargs={'indexpath':''})
    ds = ds.sortby('isobaricInhPa').sortby('latitude').sortby('longitude')
    settings = {1450: ('2014-03-11T17:18:14.200',167.02,106.07),
                1986: ('2014-03-11T21:20:15.200',176.58,99.09)}
    rng=np.random.default_rng(SEED)
    records=[]
    for seg,(depart,tas,pressure) in settings.items():
        routes=load_routes(seg); blocks=load_residuals(seg)
        depart=pd.Timestamp(depart, tz='UTC')
        base={k:route_time(ds,v,depart,tas,pressure) for k,v in routes.items()}
        print(f'\nSegment {seg} | Step 16 nominal route times: {base}')
        if any(not np.isfinite(v[0]) for v in base.values()):
            raise ValueError('Nonfinite nominal travel time; check ERA5 coverage')
        cached={k:cached_legs(ds,v,depart,tas,pressure) for k,v in routes.items()}
        for scenario,(km,minutes) in SCENARIOS.items():
            for i in range(N_SIM):
                field=make_field(rng,blocks,km,minutes)
                td=cached_route_time(cached['distance'],tas,field)
                tw=cached_route_time(cached['wind_aware'],tas,field)
                records.append(dict(segment=seg,scenario=scenario,trial=i,
                    distance_time_min=td,wind_aware_time_min=tw,
                    delta_seconds=(td-tw)*60,saving_pct=100*(td-tw)/td))
            print('  completed',scenario)
    df=pd.DataFrame(records)
    df.to_csv(OUT/'paired_trials.csv',index=False)
    rows=[]
    for (seg,scenario),part in df.groupby(['segment','scenario']):
        a=part['delta_seconds'].dropna().to_numpy()
        rows.append(dict(segment=seg,scenario=scenario,n_valid=len(a),
            nominal_step14_saving_pct={1450:0.075293,1986:0.110809}[seg],
            median_delta_seconds=np.median(a),
            p025_delta_seconds=np.quantile(a,.025),p975_delta_seconds=np.quantile(a,.975),
            fraction_wind_aware_faster=np.mean(a>0),
            fraction_distance_faster=np.mean(a<0)))
        fig,ax=plt.subplots(figsize=(8,4.5))
        ax.hist(a,bins=35,edgecolor='white')
        ax.axvline(0,color='black',ls='--',lw=1)
        ax.set(title=f'Step 16 | segment {seg} | {scenario}',xlabel='Distance-route time minus wind-aware-route time (seconds)',ylabel='Trials')
        fig.tight_layout();fig.savefig(OUT/f'segment_{seg}_{scenario}.png',dpi=180);plt.close(fig)
    summary=pd.DataFrame(rows)
    summary.to_csv(OUT/'paired_uncertainty_summary.csv',index=False)
    print('\n',summary.to_string(index=False))
    print('\nSaved to',OUT)
    print('IMPORTANT: fixed routes; no reoptimization. Off-track covariance scales are assumptions.')
    print('IMPORTANT: 5-minute block means underestimate point-scale variance; sensitivity only.')
    print('IMPORTANT: ERA5 leg times are fixed at nominal arrival estimates (first-order approximation).')
    print('IMPORTANT: Step 16 nominal evaluator may differ from Step 14 edge discretization.')

if __name__=='__main__': main()
