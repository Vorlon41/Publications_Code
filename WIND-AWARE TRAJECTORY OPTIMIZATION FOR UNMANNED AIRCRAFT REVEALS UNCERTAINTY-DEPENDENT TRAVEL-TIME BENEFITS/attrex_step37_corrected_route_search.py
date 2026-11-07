
from pathlib import Path
import importlib.util
import heapq
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
CODE = ROOT.parent / 'Code'
OUT = ROOT / 'analysis_step37'
WIDTHS = (0, 50, 100)
SEARCH_SPACING_KM = 2.0
VALIDATION_SPACINGS = (2.0, 1.0, 0.5, 0.25)
FIFO_SAMPLES = 24

def load_script(name, alias):
    path = CODE / name
    if not path.is_file():
        raise FileNotFoundError(f'Required script not found: {path}')
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def path_from_prev(prev, start, goal):
    route = [goal]
    while route[-1] != start:
        p = int(prev[route[-1]])
        if p < 0:
            raise RuntimeError('No path from start to goal')
        route.append(p)
    return route[::-1]

def shortest_distance(coords, adj, start, goal, geometry):
    best = np.full(len(coords), np.inf); best[start] = 0.0
    prev = np.full(len(coords), -1, dtype=int)
    heap = [(0.0, start)]
    while heap:
        cost, i = heapq.heappop(heap)
        if cost > best[i] + 1e-7: continue
        if i == goal: break
        for j in adj[i]:
            new = cost + geometry.arc(coords[i], coords[j])
            if new < best[j] - 1e-7:
                best[j] = new; prev[j] = i
                heapq.heappush(heap, (new, j))
    return path_from_prev(prev, start, goal)

def shortest_time(coords, adj, start, goal, depart, tas, wind, evaluator):
    """Earliest-arrival label-setting. Validity conditional on FIFO."""
    best = np.full(len(coords), np.inf); best[start] = depart
    prev = np.full(len(coords), -1, dtype=int)
    heap = [(depart, start)]; expanded = 0
    while heap:
        arrival, i = heapq.heappop(heap)
        if arrival > best[i] + 1e-7: continue
        expanded += 1
        if i == goal: break
        for j in adj[i]:
            duration, _, _ = evaluator.evaluate(
                coords[[i,j]], SEARCH_SPACING_KM, arrival, tas, wind)
            if not np.isfinite(duration): continue
            new = arrival + duration
            if new < best[j] - 1e-7:
                best[j] = new; prev[j] = i
                heapq.heappush(heap, (new, j))
    return path_from_prev(prev, start, goal), expanded

def audit_fifo(coords, adj, departure, tas, wind, evaluator):
    """Sample edges and departure times. This is NOT a formal FIFO proof."""
    edges = [(i,j) for i, row in enumerate(adj) for j in row]
    rng = np.random.default_rng(3701)
    selected = rng.choice(len(edges), min(FIFO_SAMPLES, len(edges)), replace=False)
    failures = 0; checked = 0
    # Departure perturbations within the first 10 minutes of the flight.
    for k in selected:
        i,j = edges[int(k)]
        arrivals = []
        for dt in (0., 300., 600.):
            sec,_,_ = evaluator.evaluate(coords[[i,j]], SEARCH_SPACING_KM,
                                         departure + dt, tas, wind)
            arrivals.append(departure + dt + sec)
        if np.all(np.isfinite(arrivals)):
            checked += 1
            if np.any(np.diff(arrivals) < -1e-6): failures += 1
    return checked, failures

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    graph = load_script('attrex_step26_time_dependent_corridor.py', 'attrex37graph')
    ev = load_script('attrex_step36_era5_evaluator_regression.py', 'attrex37eval')
    # Step 26 builds its graph using Step 18 edge_geometry.
    base = graph.import_base()
    obs_path = ROOT / 'analysis_step23' / '20140311_segment_2798_observations.csv'
    grib_path = ROOT / 'ERA5_ATTREX_20140311_20140312_expanded.grib'
    if not obs_path.is_file() or not grib_path.is_file():
        raise FileNotFoundError('Missing Step 23 observations or expanded March 11 GRIB')
    obs = pd.read_csv(obs_path).sort_values('TIME_UTC').dropna(subset=['TIME_UTC','G_LAT','G_LONG','P','TAS'])
    a = obs[['G_LAT','G_LONG']].iloc[0].to_numpy(float)
    b = obs[['G_LAT','G_LONG']].iloc[-1].to_numpy(float)
    depart = pd.Timestamp('2014-03-11', tz='UTC').timestamp() + float(obs.TIME_UTC.iloc[0])
    pressure, tas = float(obs.P.median()), float(obs.TAS.median())
    with xr.open_dataset(grib_path, engine='cfgrib', backend_kwargs={'indexpath':''}) as ds:
        wind = ev.Wind(ds, pressure)
    print('Step 37 segment 2798, departure:', pd.to_datetime(depart,unit='s',utc=True))
    print('Fixed pressure hPa:', pressure, 'fixed TAS m/s:', tas)
    details=[]; pairs=[]
    for width in WIDTHS:
        coords, adj, start, goal = graph.build_corridor(base, a, b, width)
        if np.any(coords[:,0] < wind.lat.min()) or np.any(coords[:,0] > wind.lat.max()) or np.any(coords[:,1] < wind.lon.min()) or np.any(coords[:,1] > wind.lon.max()):
            raise ValueError(f'Width {width}: graph outside ERA5 spatial domain')
        checked, failed = audit_fifo(coords, adj, depart, tas, wind, ev)
        if failed:
            raise RuntimeError(f'FIFO violation in {failed}/{checked} sampled edges; label-setting search not justified')
        dpath = shortest_distance(coords, adj, start, goal, ev)
        wpath, expanded = shortest_time(coords, adj, start, goal, depart, tas, wind, ev)
        paths = {'distance':dpath, 'wind_aware':wpath}
        for name, path in paths.items():
            xy = coords[path]
            pd.DataFrame(xy, columns=['latitude','longitude']).to_csv(OUT/f'width{width}_{name}_route.csv', index=False)
            for spacing in VALIDATION_SPACINGS:
                sec, km, n = ev.evaluate(xy, spacing, depart, tas, wind)
                details.append(dict(width_km=width,route=name,spacing_km=spacing,
                                    time_s=sec,distance_km=km,waypoints=len(path),sublegs=n,
                                    fifo_edges_checked=checked,fifo_violations=failed))
        for spacing in VALIDATION_SPACINGS:
            d = next(x for x in details if x['width_km']==width and x['spacing_km']==spacing and x['route']=='distance')
            w = next(x for x in details if x['width_km']==width and x['spacing_km']==spacing and x['route']=='wind_aware')
            pairs.append(dict(width_km=width,spacing_km=spacing,distance_time_s=d['time_s'],
                              wind_aware_time_s=w['time_s'],advantage_s=d['time_s']-w['time_s'],
                              distance_km=d['distance_km'],wind_aware_km=w['distance_km'],
                              same_nodes=dpath==wpath,expanded=expanded,
                              fifo_edges_checked=checked,fifo_violations=failed))
        fig,ax=plt.subplots(figsize=(9,6))
        ax.plot(obs.G_LONG,obs.G_LAT,color='0.7',lw=1,label='Observed science track')
        for name,color in [('distance','black'),('wind_aware','red')]:
            xy=coords[paths[name]]
            ax.plot(xy[:,1],xy[:,0],color=color,lw=2,label=name)
        ax.scatter([a[1],b[1]],[a[0],b[0]],c=['green','blue'],s=40,zorder=5)
        ax.set(xlabel='Longitude (degrees E)',ylabel='Latitude (degrees N)',
               title=f'Step 37 | segment 2798 | corridor +/-{width} km')
        ax.grid(alpha=.2);ax.legend();fig.tight_layout()
        fig.savefig(OUT/f'width{width}_routes.png',dpi=170);plt.close(fig)
        fine=next(p for p in pairs if p['width_km']==width and p['spacing_km']==0.25)
        print(f'Width +/-{width} km: advantage {fine["advantage_s"]:.6f} s; same nodes {fine["same_nodes"]}; FIFO sampled {checked}, violations {failed}')
    pd.DataFrame(details).to_csv(OUT/'step37_route_details.csv',index=False)
    paired=pd.DataFrame(pairs)
    paired.to_csv(OUT/'step37_paired_comparison.csv',index=False)
    print('\n',paired.to_string(index=False))
    print('\nSaved:',OUT)
    print('CAUTION: sampled FIFO audit does NOT certify FIFO; search optimality conditional.')
    print('CAUTION: search uses 2-km edges; 0.25-km evaluation is a post-search check.')
    print('CAUTION: no aircraft dynamics, mission, airspace, fuel or operational savings.')

if __name__=='__main__':
    main()
