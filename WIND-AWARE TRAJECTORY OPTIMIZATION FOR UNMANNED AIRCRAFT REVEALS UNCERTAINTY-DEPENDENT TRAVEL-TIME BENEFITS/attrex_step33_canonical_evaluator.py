
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import xarray as xr

PROJECT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
CODE = PROJECT.parent / 'Code'
SOURCE = PROJECT / 'analysis_step27'
OUT = PROJECT / 'analysis_step33'
GRIB = PROJECT / 'ERA5_ATTREX_20140311_20140312_expanded.grib'
OBS = PROJECT / 'analysis_step23' / '20140311_segment_2798_observations.csv'
SPACINGS = (5., 2., 1., .5)
R = 6371.0088

def load_script(filename, name):
    path = CODE / filename
    if not path.exists(): raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def unit(p):
    lat,lon=np.deg2rad(p)
    return np.array([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])

def arc(a,b):
    return R*np.arctan2(np.linalg.norm(np.cross(unit(a),unit(b))),np.dot(unit(a),unit(b)))

def slerp(a,b,f):
    va,vb=unit(a),unit(b)
    theta=np.arctan2(np.linalg.norm(np.cross(va,vb)),np.dot(va,vb))
    if theta<1e-12: v=va
    else: v=(np.sin((1-f)*theta)*va+np.sin(f*theta)*vb)/np.sin(theta)
    v=v/np.linalg.norm(v)
    return np.array([np.rad2deg(np.arctan2(v[2],np.hypot(v[0],v[1]))),np.rad2deg(np.arctan2(v[1],v[0]))])

def deviation_km(p,a,b):
    # Cross-track angular distance to great circle, clamped to segment ends.
    va,vb,vp=unit(a),unit(b),unit(p)
    n=np.cross(va,vb)
    if np.linalg.norm(n)<1e-12: return min(arc(p,a),arc(p,b))
    n/=np.linalg.norm(n)
    projected=vp-n*np.dot(vp,n)
    projected/=np.linalg.norm(projected)
    angle_ab=np.arctan2(np.linalg.norm(np.cross(va,vb)),np.dot(va,vb))
    angle_ap=np.arctan2(np.dot(n,np.cross(va,projected)),np.dot(va,projected))
    if angle_ap<0 or angle_ap>angle_ab: return min(arc(p,a),arc(p,b))
    return R*np.arcsin(np.clip(abs(np.dot(vp,n)),0,1))

def canonicalize(points, tolerance_km=.05):
    points=np.asarray(points,float)
    if len(points)<2: raise ValueError('Route needs two points')
    # Collapse a route to its geodesic only when every interior waypoint lies
    # within the prespecified tolerance of the same endpoint geodesic.
    if max((deviation_km(p,points[0],points[-1]) for p in points[1:-1]),default=0)<=tolerance_km:
        return points[[0,-1]]
    return points

def global_sample(points,spacing_km):
    # Arc-length sampling is anchored to route origin, not to graph-edge boundaries.
    lengths=np.array([arc(a,b) for a,b in zip(points[:-1],points[1:])])
    cumulative=np.r_[0,np.cumsum(lengths)]
    total=cumulative[-1]
    if total<=0: raise ValueError('Zero-length route')
    stations=np.r_[np.arange(0,total,spacing_km),total]
    stations=np.unique(stations)
    output=[]
    for s in stations:
        j=min(np.searchsorted(cumulative,s,side='right')-1,len(lengths)-1)
        f=0 if lengths[j]==0 else np.clip((s-cumulative[j])/lengths[j],0,1)
        output.append(slerp(points[j],points[j+1],f))
    return np.array(output)

def evaluate(base,points,depart,wind,tas,spacing):
    points=canonicalize(points)
    sampled=global_sample(points,spacing)
    # One travel-time propagation over a common global-distance sampling mesh.
    # Force travel() not to subdivide these already short legs.
    base.MAX_LEG_KM=float(spacing)*1.01
    elapsed=0.
    for a,b in zip(sampled[:-1],sampled[1:]):
        dt=base.travel(a,b,depart+elapsed,wind,tas)
        if not np.isfinite(dt): return np.nan
        elapsed+=dt
    return elapsed

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    s27=load_script('attrex_step27_uncertainty_reoptimization.py','s27_step33')
    s26=load_script('attrex_step26_time_dependent_corridor.py','s26_step33')
    base=s26.import_base()
    trials=pd.read_csv(SOURCE/'step27_trials.csv')
    obs=pd.read_csv(OBS).sort_values('TIME_UTC').dropna(subset=['TIME_UTC','G_LAT','G_LONG','P','TAS'])
    a=obs[['G_LAT','G_LONG']].iloc[0].to_numpy(float)
    b=obs[['G_LAT','G_LONG']].iloc[-1].to_numpy(float)
    depart=pd.Timestamp('2014-03-11',tz='UTC').timestamp()+float(obs.TIME_UTC.iloc[0])
    tas=float(obs.TAS.median());pressure=float(obs.P.median())
    records=[]
    with xr.open_dataset(GRIB,engine='cfgrib',backend_kwargs={'indexpath':''}) as ds:
        reference=base.Wind(ds,pressure)
        rng=np.random.default_rng(s27.SEED)
        for width in s27.WIDTHS_KM:
            for corr in s27.CORRELATION_KM:
                for trial in range(s27.N_TRIALS):
                    # Regenerate every scenario, including trials whose paths were identical.
                    wind=s27.PerturbedWind(reference,rng,corr,(a+b)/2,depart,s27.APPLY_MEAN_BIAS_CORRECTION)
                    match=trials[(trials.width_km==width)&(trials.correlation_km==corr)&(trials.trial==trial)]
                    if len(match)!=1: raise ValueError('Trial indexing mismatch')
                    if str(match.iloc[0].same_route).lower()=='true':continue
                    files={mode:SOURCE/f'width{int(width)}_corr{int(corr)}_trial{trial:03d}_{mode}.csv' for mode in ('distance','wind_aware')}
                    if not all(p.exists() for p in files.values()):
                        print('Missing route pair:',corr,trial);continue
                    paths={mode:pd.read_csv(path)[['latitude','longitude']].to_numpy(float) for mode,path in files.items()}
                    if not np.allclose(paths['distance'][0],paths['wind_aware'][0],atol=1e-9) or not np.allclose(paths['distance'][-1],paths['wind_aware'][-1],atol=1e-9):raise ValueError('Endpoint mismatch')
                    deviations={mode:max((deviation_km(p,pts[0],pts[-1]) for p in pts[1:-1]),default=0) for mode,pts in paths.items()}
                    for spacing in SPACINGS:
                        td=evaluate(base,paths['distance'],depart,wind,tas,spacing)
                        tw=evaluate(base,paths['wind_aware'],depart,wind,tas,spacing)
                        records.append(dict(width_km=width,correlation_km=corr,trial=trial,spacing_km=spacing,
                                            distance_time_s=td,wind_aware_time_s=tw,advantage_s=td-tw,
                                            original_advantage_s=float(match.iloc[0].advantage_s),
                                            distance_max_deviation_m=deviations['distance']*1000,
                                            wind_max_deviation_m=deviations['wind_aware']*1000,
                                            distance_canonical_points=len(canonicalize(paths['distance'])),
                                            wind_canonical_points=len(canonicalize(paths['wind_aware']))))
                    print(f'Completed corr={corr:g}, trial={trial:03d}',flush=True)
    if not records:raise RuntimeError('No saved differing route pairs found')
    df=pd.DataFrame(records)
    df.to_csv(OUT/'step33_details.csv',index=False)
    summary=df.groupby(['correlation_km','spacing_km']).agg(n_pairs=('trial','count'),max_abs_advantage_s=('advantage_s',lambda x:float(np.max(np.abs(x)))),max_lateral_deviation_m=('wind_max_deviation_m','max')).reset_index()
    summary.to_csv(OUT/'step33_summary.csv',index=False)
    print('\n',summary.to_string(index=False))
    fine=summary[summary.spacing_km==min(SPACINGS)]
    passed=bool((fine.max_abs_advantage_s<.01).all())
    print('\nSubdivision invariance (<0.01 s at 0.5 km):', 'PASS' if passed else 'FAIL')
    print('Saved:',OUT)
    print('CAUTION: near-geodesic canonicalization tolerance is 50 m; test other tolerances for robustness.')
    print('CAUTION: this is a fixed-route numerical audit, not an optimizer correction or FIFO certification.')

if __name__=='__main__':main()
