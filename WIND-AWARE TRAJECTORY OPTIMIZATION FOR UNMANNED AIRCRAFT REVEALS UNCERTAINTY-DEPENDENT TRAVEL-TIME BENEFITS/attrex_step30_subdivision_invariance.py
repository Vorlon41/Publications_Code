
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

PROJECT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
CODE = PROJECT.parent / 'Code'
SOURCE = PROJECT / 'analysis_step27'
OUT = PROJECT / 'analysis_step30'
GRIB = PROJECT / 'ERA5_ATTREX_20140311_20140312_expanded.grib'
OBS = PROJECT / 'analysis_step23' / '20140311_segment_2798_observations.csv'
SPACINGS = (5.0, 2.0, 1.0, 0.5)
EARTH = 6371008.8

def load_script(filename, name):
    path = CODE / filename
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}. Save the original script in ATTREX\\Code.')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def great_circle(a, b, f):
    a, b = np.deg2rad(a), np.deg2rad(b)
    va = np.array([np.cos(a[0])*np.cos(a[1]), np.cos(a[0])*np.sin(a[1]), np.sin(a[0])])
    vb = np.array([np.cos(b[0])*np.cos(b[1]), np.cos(b[0])*np.sin(b[1]), np.sin(b[0])])
    theta = np.arccos(np.clip(va@vb, -1, 1))
    if theta < 1e-12: v = va
    else: v = (np.sin((1-f)*theta)*va + np.sin(f*theta)*vb)/np.sin(theta)
    return np.array([np.rad2deg(np.arctan2(v[2], np.hypot(v[0],v[1]))),
                     np.rad2deg(np.arctan2(v[1],v[0]))])

def subdivide(a, b, n):
    return np.array([great_circle(a,b,k/n) for k in range(n+1)])

def evaluate(base, points, depart, wind, tas, spacing):
    base.MAX_LEG_KM = float(spacing)
    elapsed = 0.0
    for a,b in zip(points[:-1],points[1:]):
        dt = base.travel(a,b,depart+elapsed,wind,tas)
        if not np.isfinite(dt): return np.nan
        elapsed += dt
    return elapsed

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    s27 = load_script('attrex_step27_uncertainty_reoptimization.py','s27_step30')
    s26 = load_script('attrex_step26_time_dependent_corridor.py','s26_step30')
    base = s26.import_base()
    trials = pd.read_csv(SOURCE/'step27_trials.csv')
    obs = pd.read_csv(OBS).sort_values('TIME_UTC').dropna(subset=['TIME_UTC','G_LAT','G_LONG','P','TAS'])
    a = obs[['G_LAT','G_LONG']].iloc[0].to_numpy(float)
    b = obs[['G_LAT','G_LONG']].iloc[-1].to_numpy(float)
    depart = pd.Timestamp('2014-03-11',tz='UTC').timestamp()+float(obs.TIME_UTC.iloc[0])
    tas = float(obs.TAS.median())
    pressure = float(obs.P.median())
    ds = xr.open_dataset(GRIB,engine='cfgrib',backend_kwargs={'indexpath':''})
    reference = base.Wind(ds,pressure)
    rng = np.random.default_rng(s27.SEED)
    records=[]
    checked=0
    for width in s27.WIDTHS_KM:
        for corr in s27.CORRELATION_KM:
            for trial in range(s27.N_TRIALS):
                wind = s27.PerturbedWind(reference,rng,corr,(a+b)/2,depart,s27.APPLY_MEAN_BIAS_CORRECTION)
                match=trials[(trials.width_km==width)&(trials.correlation_km==corr)&(trials.trial==trial)]
                if len(match)!=1: raise ValueError(f'Missing trial {width} {corr} {trial}')
                if str(match.iloc[0].same_route).lower()=='true': continue
                files={mode:SOURCE/f'width{int(width)}_corr{int(corr)}_trial{trial:03d}_{mode}.csv'
                       for mode in ('distance','wind_aware')}
                if not all(f.exists() for f in files.values()):
                    print('Skipping missing pair',corr,trial)
                    continue
                paths={mode:pd.read_csv(path)[['latitude','longitude']].to_numpy(float)
                       for mode,path in files.items()}
                if not np.allclose(paths['distance'][0],paths['wind_aware'][0],atol=1e-9) or not np.allclose(paths['distance'][-1],paths['wind_aware'][-1],atol=1e-9):
                    raise ValueError('Endpoints do not match')
                checked+=1
                # Same physical direct route, with artificial great-circle waypoints.
                canonical={n:subdivide(a,b,n) for n in (1,4,16)}
                for spacing in SPACINGS:
                    values={mode:evaluate(base,pts,depart,wind,tas,spacing)
                            for mode,pts in paths.items()}
                    values.update({f'canonical_{n}':evaluate(base,pts,depart,wind,tas,spacing)
                                   for n,pts in canonical.items()})
                    records.append(dict(correlation_km=corr,trial=trial,spacing_km=spacing,
                        saved_advantage_s=float(match.iloc[0].advantage_s),
                        original_advantage_s=values['distance']-values['wind_aware'],
                        direct_time_s=values['canonical_1'],
                        split4_minus_direct_s=values['canonical_4']-values['canonical_1'],
                        split16_minus_direct_s=values['canonical_16']-values['canonical_1'],
                        wind_route_minus_direct_s=values['wind_aware']-values['canonical_1']))
                print(f'Completed correlation {corr:g} km, trial {trial:03d}',flush=True)
    if not records: raise RuntimeError('No differing-route pairs found; verify analysis_step27 files')
    df=pd.DataFrame(records)
    df.to_csv(OUT/'step30_subdivision_details.csv',index=False)
    summary=df.groupby(['correlation_km','spacing_km']).agg(
        n_pairs=('trial','count'),
        max_abs_original_advantage_s=('original_advantage_s',lambda v:np.max(np.abs(v))),
        max_abs_split4_error_s=('split4_minus_direct_s',lambda v:np.max(np.abs(v))),
        max_abs_split16_error_s=('split16_minus_direct_s',lambda v:np.max(np.abs(v))),
        max_abs_wind_vs_direct_s=('wind_route_minus_direct_s',lambda v:np.max(np.abs(v)))) .reset_index()
    summary.to_csv(OUT/'step30_subdivision_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,5))
    for corr,part in summary.groupby('correlation_km'):
        ax.plot(part.spacing_km,part.max_abs_split16_error_s,'o-',label=f'{corr:g} km correlation')
    ax.axhline(0,color='black',lw=0.8)
    ax.set(xlabel='Maximum integration subleg (km)',ylabel='Max |16-waypoint minus direct time| (s)',
           title='Step 30: subdivision invariance diagnostic')
    ax.legend();fig.tight_layout()
    fig.savefig(OUT/'step30_subdivision_invariance.png',dpi=180)
    plt.close(fig)
    print('\nSUMMARY\n',summary.to_string(index=False))
    print('\nSaved:',OUT,'| pairs:',checked)
    print('NOTE: Differences diagnose numerical representation dependence; no route optimization is performed.')
    print('NOTE: Original 1-km advantages should agree with Step 27 if replay RNG and files are unchanged.')
    print('NOTE: A canonical shared geometry gives the same time by construction, not proof of model accuracy.')

if __name__=='__main__': main()
