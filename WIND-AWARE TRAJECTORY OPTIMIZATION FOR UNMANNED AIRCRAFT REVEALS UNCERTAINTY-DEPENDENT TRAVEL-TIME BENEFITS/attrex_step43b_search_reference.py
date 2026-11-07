
from pathlib import Path
import importlib.util
import itertools
import json
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
CODE = ROOT.parent / 'Code'
ERA5 = ROOT / 'analysis_step31' / 'ERA5_ATTREX_March11_12_2014.nc'
OBS = ROOT / 'analysis_step23' / '20140311_segment_3595_observations.csv'
OUT = ROOT / 'analysis_step43b'
SEARCH_SPACING_KM = 5.0
VALIDATION_SPACING_KM = 2.0
FIFO_OFFSETS_MIN = (0, 10, 20, 40, 60, 90)


def load_module(filename, alias):
    path = CODE / filename
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_reference_graph(ev, a, b):
    """Five nodes: endpoints of first 25% of great circle, three offset choices."""
    end = ev.interpolate(a, b, 0.25)
    mids = [ev.interpolate(a, end, f) for f in (0.25, 0.50, 0.75)]
    offsets_km = (35., -30., 25.)
    for p, off in zip(mids, offsets_km):
        # Small longitude shift, latitude-adjusted; creates distinct paths.
        p[1] += off / (111.32 * np.cos(np.deg2rad(p[0])))
    coords = np.vstack([a, *mids, end])
    # Directed complete graph, with no self-loops. Enumerate every simple path.
    adj = [[j for j in range(len(coords)) if j != i] for i in range(len(coords))]
    return coords, adj, 0, len(coords)-1


def exhaustive_simple_paths(adj, start, goal):
    interior = [i for i in range(len(adj)) if i not in (start, goal)]
    for k in range(len(interior)+1):
        for perm in itertools.permutations(interior, k):
            path = (start,) + perm + (goal,)
            if all(v in adj[u] for u, v in zip(path[:-1], path[1:])):
                yield path


def path_arrival(coords, path, depart, tas, wind, ev, spacing):
    arrival = float(depart)
    for i, j in zip(path[:-1], path[1:]):
        seconds, _, _ = ev.evaluate(coords[[i,j]], spacing, arrival, tas, wind)
        if not np.isfinite(seconds):
            return np.inf
        arrival += seconds
    return arrival


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    ev = load_module('attrex_step36_era5_evaluator_regression.py', 'step43b_ev')
    search = load_module('attrex_step37_corrected_route_search.py', 'step43b_search')
    search.SEARCH_SPACING_KM = SEARCH_SPACING_KM
    obs = pd.read_csv(OBS).dropna(subset=['TIME_UTC','G_LAT','G_LONG','P','TAS'])
    obs = obs.sort_values('TIME_UTC').reset_index(drop=True)
    a = obs[['G_LAT','G_LONG']].iloc[0].to_numpy(float)
    b = obs[['G_LAT','G_LONG']].iloc[-1].to_numpy(float)
    depart = pd.Timestamp('2014-03-11', tz='UTC').timestamp() + float(obs.TIME_UTC.iloc[0])
    tas = float(obs.TAS.median())
    pressure = float(obs.P.median())
    with xr.open_dataset(ERA5) as ds:
        if ds.sizes.get('time') != 48:
            raise ValueError('Expected 48-hour ERA5 dataset')
        ds_wind = ds[['u','v']].load()
    wind = ev.Wind(ds_wind, pressure)
    coords, adj, start, goal = build_reference_graph(ev, a, b)
    if not ((coords[:,0]>=wind.lat.min()).all() and (coords[:,0]<=wind.lat.max()).all()
            and (coords[:,1]>=wind.lon.min()).all() and (coords[:,1]<=wind.lon.max()).all()):
        raise ValueError('Reference graph outside ERA5 domain')
    paths = list(exhaustive_simple_paths(adj, start, goal))
    rows=[]
    for path in paths:
        arrival = path_arrival(coords, path, depart, tas, wind, ev, SEARCH_SPACING_KM)
        rows.append({'path':'-'.join(map(str,path)), 'nodes':len(path),
                     'arrival_epoch_s':arrival, 'duration_s':arrival-depart})
    df=pd.DataFrame(rows).sort_values('arrival_epoch_s')
    df.to_csv(OUT/'reference_paths.csv', index=False)
    best_ref=float(df.arrival_epoch_s.iloc[0])
    label_path, expansions=search.shortest_time(coords, adj, start, goal, depart, tas, wind, ev)
    label_arrival=path_arrival(coords,label_path,depart,tas,wind,ev,SEARCH_SPACING_KM)
    gap_s=label_arrival-best_ref
    # Test FIFO over EVERY directed edge, at a wider departure-time sample than Step 38.
    fifo_rows=[]
    for i, neighbors in enumerate(adj):
        for j in neighbors:
            arrivals=[]
            for offset in FIFO_OFFSETS_MIN:
                t=depart+offset*60
                dt,_,_=ev.evaluate(coords[[i,j]], SEARCH_SPACING_KM, t, tas, wind)
                arrivals.append(t+dt if np.isfinite(dt) else np.nan)
            finite=np.isfinite(arrivals)
            violations=int(sum(arrivals[k+1]<arrivals[k]-1e-6 for k in range(len(arrivals)-1)
                               if finite[k] and finite[k+1]))
            fifo_rows.append({'from':i,'to':j,'finite_samples':int(finite.sum()),'violations':violations})
    fifo=pd.DataFrame(fifo_rows)
    fifo.to_csv(OUT/'fifo_edge_audit.csv',index=False)
    # Finer re-evaluation is NOT an independent optimality proof.
    top_path=tuple(map(int,df.path.iloc[0].split('-')))
    fine_ref=path_arrival(coords,top_path,depart,tas,wind,ev,VALIDATION_SPACING_KM)
    fine_label=path_arrival(coords,label_path,depart,tas,wind,ev,VALIDATION_SPACING_KM)
    result={
        'reference_graph_nodes':len(coords),'directed_edges':sum(map(len,adj)),
        'enumerated_simple_paths':len(paths),'label_path':label_path,
        'reference_path':list(top_path),'search_expansions':expansions,
        'search_spacing_km':SEARCH_SPACING_KM,
        'search_minus_reference_s':gap_s,
        'fine_spacing_km':VALIDATION_SPACING_KM,
        'fine_label_minus_reference_s':fine_label-fine_ref,
        'fifo_edges_checked':len(fifo),'fifo_sampled_violations':int(fifo.violations.sum()),
        'result':'PASS' if abs(gap_s)<0.01 else 'FAIL',
        'limitation':'Bounded five-node subgraph only; same physical evaluator; no global FIFO proof.'
    }
    (OUT/'step43b_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    pd.DataFrame([result]).to_csv(OUT/'step43b_summary.csv',index=False)
    print('\nSTEP 43B REFERENCE CHECK')
    for k,v in result.items():print(f'{k}: {v}')
    print('\nSaved:',OUT)
    if result['result']=='FAIL':
        print('WARNING: label-setting search disagrees with exhaustive reference; do not freeze results.')
    if result['fifo_sampled_violations']:
        print('WARNING: sampled FIFO violations detected; search optimality is not assured.')

if __name__=='__main__':
    main()
