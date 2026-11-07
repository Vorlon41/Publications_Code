
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

PROJECT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
CODE = PROJECT.parent / 'Code'
OUT = PROJECT / 'analysis_step29'
SRC = PROJECT / 'analysis_step27'
GRIB = PROJECT / 'ERA5_ATTREX_20140311_20140312_expanded.grib'
OBS = PROJECT / 'analysis_step23' / '20140311_segment_2798_observations.csv'
SPACINGS = (5., 2., 1.)

def load(path, name):
    if not path.exists(): raise FileNotFoundError(f'Missing dependency: {path}')
    spec=importlib.util.spec_from_file_location(name,str(path))
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def coords_file(width, corr, trial, mode):
    return SRC / f'width{int(width)}_corr{int(corr)}_trial{int(trial):03d}_{mode}.csv'

def route_eval(mod, base, points, depart, wind, tas, spacing):
    return mod.eval_route(base, points, list(range(len(points))), depart, wind, tas, spacing)[0]

def cross_track_m(points, a, b):
    # Approximate perpendicular offset from great-circle a->b; this is an
    # infinite great-circle cross-track distance, not a segment-clamped Hausdorff metric.
    R=6371008.8
    def ang(x,y):
        p,q=np.deg2rad(x),np.deg2rad(y)
        h=np.sin((q[0]-p[0])/2)**2+np.cos(p[0])*np.cos(q[0])*np.sin((q[1]-p[1])/2)**2
        return 2*np.arcsin(np.sqrt(np.clip(h,0,1)))
    def bearing(x,y):
        p,q=np.deg2rad(x),np.deg2rad(y)
        dl=q[1]-p[1]
        return np.arctan2(np.sin(dl)*np.cos(q[0]),np.cos(p[0])*np.sin(q[0])-np.sin(p[0])*np.cos(q[0])*np.cos(dl))
    theta=bearing(a,b)
    return max(abs(np.arcsin(np.clip(np.sin(ang(a,p))*np.sin(bearing(a,p)-theta),-1,1))*R) for p in points)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    s27=load(CODE/'attrex_step27_uncertainty_reoptimization.py','s27_replay')
    mod=load(CODE/'attrex_step26_time_dependent_corridor.py','s26_replay')
    base=mod.import_base()
    df=pd.read_csv(SRC/'step27_trials.csv')
    obs=pd.read_csv(OBS).sort_values('TIME_UTC').dropna(subset=['TIME_UTC','G_LAT','G_LONG','P','TAS'])
    a=obs[['G_LAT','G_LONG']].iloc[0].to_numpy(float)
    b=obs[['G_LAT','G_LONG']].iloc[-1].to_numpy(float)
    depart=pd.Timestamp('2014-03-11',tz='UTC').timestamp()+float(obs.TIME_UTC.iloc[0])
    tas=float(obs.TAS.median());pressure=float(obs.P.median())
    ds=xr.open_dataset(GRIB,engine='cfgrib',backend_kwargs={'indexpath':''})
    ref=base.Wind(ds,pressure)
    rng=np.random.default_rng(s27.SEED)
    records=[]; checked=0
    # Must replay EVERY trial in original nesting order to preserve RNG state.
    for width in s27.WIDTHS_KM:
        for corr in s27.CORRELATION_KM:
            for trial in range(s27.N_TRIALS):
                wind=s27.PerturbedWind(ref,rng,corr,(a+b)/2,depart,s27.APPLY_MEAN_BIAS_CORRECTION)
                matches=df[(df.width_km==width)&(df.correlation_km==corr)&(df.trial==trial)]
                if len(matches)!=1: raise ValueError(f'Missing or duplicate trial row: {width},{corr},{trial}')
                expected=matches.iloc[0]
                if str(expected.same_route).strip().lower()=='true':continue
                paths={}
                for mode in ('distance','wind_aware'):
                    path=coords_file(width,corr,trial,mode)
                    if not path.exists():
                        print('MISSING route CSV:',path.name)
                        paths={};break
                    paths[mode]=pd.read_csv(path)[['latitude','longitude']].to_numpy(float)
                if len(paths)!=2:continue
                checked+=1
                d=paths['distance'];w=paths['wind_aware']
                if np.max(np.abs(d[0]-w[0]))>1e-8 or np.max(np.abs(d[-1]-w[-1]))>1e-8:
                    raise ValueError('Route endpoints differ; cannot use canonical comparator')
                canonical=np.vstack((d[0],d[-1]))
                cross=cross_track_m(w,canonical[0],canonical[-1])
                for spacing in SPACINGS:
                    td=route_eval(mod,base,d,depart,wind,tas,spacing)
                    tw=route_eval(mod,base,w,depart,wind,tas,spacing)
                    tc=route_eval(mod,base,canonical,depart,wind,tas,spacing)
                    records.append(dict(width_km=width,correlation_km=corr,trial=trial,
                        spacing_km=spacing,distance_waypoints=len(d),wind_waypoints=len(w),
                        max_wind_cross_track_m=cross,distance_time_s=td,wind_time_s=tw,
                        original_advantage_s=td-tw,canonical_time_s=tc,
                        distance_minus_canonical_s=td-tc,wind_minus_canonical_s=tw-tc,
                        step27_saved_advantage_s=float(expected.advantage_s),
                        step27_replay_difference_s=(td-tw-float(expected.advantage_s)) if spacing==s27.VALIDATION_SPACING_KM else np.nan))
                print(f'REPLAY corr={corr:g} trial={trial:03d}: cross-track max {cross:.3f} m; '
                      f'advantage at 1 km {records[-1]["original_advantage_s"]:.6f} s')
    if not records:
        print('No saved differing-route pairs found. Confirm analysis_step27 files.');return
    result=pd.DataFrame(records)
    result.to_csv(OUT/'step29_replay_details.csv',index=False)
    summary=result.groupby(['correlation_km','spacing_km']).agg(
        n_pairs=('trial','count'),max_cross_track_m=('max_wind_cross_track_m','max'),
        median_advantage_s=('original_advantage_s','median'),
        max_abs_advantage_s=('original_advantage_s',lambda x:float(np.max(np.abs(x)))),
        max_abs_replay_error_s=('step27_replay_difference_s',lambda x:float(np.nanmax(np.abs(x))) if np.isfinite(x).any() else np.nan)
    ).reset_index()
    summary.to_csv(OUT/'step29_replay_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,5))
    for corr,g in result.groupby('correlation_km'):
        vals=g.groupby('spacing_km').original_advantage_s.apply(lambda v:float(np.max(np.abs(v))))
        ax.plot(vals.index,vals.values,'o-',label=f'{corr:g}-km correlation')
    ax.axhline(0,color='black',lw=0.8)
    ax.set(xlabel='Maximum integration subleg (km)',ylabel='Largest absolute paired time difference (s)',
           title='Step 29 | representation sensitivity under replayed winds')
    ax.legend();fig.tight_layout();fig.savefig(OUT/'step29_convergence.png',dpi=180);plt.close(fig)
    print('\nSUMMARY\n',summary.to_string(index=False))
    print('\nSaved:',OUT,'| Audited route pairs:',checked)
    print('IMPORTANT: replay only matches Step 27 if original script settings and RNG sequence are unchanged.')
    print('IMPORTANT: common direct-route timing tests geometry equivalence; it is not reoptimization.')
    print('IMPORTANT: off-track wind-error correlation is assumed, not empirically validated.')
if __name__=='__main__':main()
