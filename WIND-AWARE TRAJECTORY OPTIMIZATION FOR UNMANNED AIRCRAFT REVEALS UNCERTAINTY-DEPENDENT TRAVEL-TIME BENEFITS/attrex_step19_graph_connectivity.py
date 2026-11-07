
from pathlib import Path
import importlib.util
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from pathlib import Path

CODE_DIR = Path(
    r"C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\Code"
)

STEP18 = CODE_DIR / "attrex_step18_exact_endpoint_astar.py"

print("Step 18 found:", STEP18.exists())
if not STEP18.exists():
    raise FileNotFoundError('Place attrex_step18_exact_endpoint_astar.py alongside this script')
spec = importlib.util.spec_from_file_location('attrex_step18', STEP18)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

ROOT = base.ROOT
OUT = ROOT / 'analysis_step19'
SEGMENTS = (1450, 1986)
# Run expanded connectivity on 0.25 degree first to limit runtime.
CASES = [(0.25, 'direct'), (0.25, 'local'), (0.25, 'expanded'),
         (0.125, 'direct'), (0.125, 'local')]
# Explicitly forbid any direct edge in local/expanded configurations.
# Expanded uses all offsets up to two grid cells, including knight moves.


def build_graph(ds, segment, grid_deg, graph_type):
    coords, _, start, goal, depart, pressure, tas, obs = base.load_case(ds, segment, grid_deg)
    # Infer rectangular grid dimensions from coordinates: rows share latitude.
    grid = coords[:-2]
    lats = np.unique(grid[:, 0]); lons = np.unique(grid[:, 1])
    ny, nx = len(lats), len(lons)
    assert ny * nx == len(grid), 'Unexpected grid layout'
    n = len(grid)
    adj = [set() for _ in range(n+2)]
    reach = 2 if graph_type == 'expanded' else 1
    for y in range(ny):
        for x in range(nx):
            i = y * nx + x
            for dy in range(-reach, reach+1):
                for dx in range(-reach, reach+1):
                    if dy == 0 and dx == 0: continue
                    yy, xx = y+dy, x+dx
                    if 0 <= yy < ny and 0 <= xx < nx:
                        adj[i].add(yy*nx+xx)
    k = 16 if graph_type == 'expanded' else 8
    for terminal in (start, goal):
        d = np.array([base.edge_geometry(coords[terminal], p)[0] for p in grid])
        for j in np.argsort(d)[:min(k, n)]:
            adj[terminal].add(int(j)); adj[int(j)].add(terminal)
    if graph_type == 'direct':
        adj[start].add(goal); adj[goal].add(start)
    return coords, [sorted(a) for a in adj], start, goal, depart, pressure, tas, obs


def solve_distance(coords, adj, start, goal):
    return base.solve(coords, adj, start, goal, 0., None, 1., 'distance')


def solve_time(coords, adj, start, goal, depart, wind, tas):
    return base.solve(coords, adj, start, goal, depart, wind, tas, 'time')


def main():
    import xarray as xr
    OUT.mkdir(parents=True, exist_ok=True)
    if not base.GRIB.exists():
        raise FileNotFoundError(f'Missing GRIB: {base.GRIB}')
    ds = xr.open_dataset(base.GRIB, engine='cfgrib', backend_kwargs={'indexpath': ''})
    records = []
    for segment in SEGMENTS:
        for grid_deg, graph_type in CASES:
            print(f'Running segment {segment}, grid {grid_deg}, graph {graph_type}', flush=True)
            coords, adj, start, goal, depart, pressure, tas, obs = build_graph(
                ds, segment, grid_deg, graph_type)
            wind = base.Wind(ds, pressure)
            if not wind.times[0] <= depart <= wind.times[-1]:
                raise ValueError('Departure outside ERA5 coverage')
            results = {}
            for mode in ('distance', 'wind_aware'):
                if mode == 'distance':
                    route, expanded = solve_distance(coords, adj, start, goal)
                else:
                    route, expanded = solve_time(coords, adj, start, goal, depart, wind, tas)
                time_min, km = base.evaluate(coords, route, depart, wind, tas)
                xy = coords[route]
                max_turn, turns30 = base.turns(xy)
                results[mode] = (time_min, km, xy)
                pd.DataFrame(xy, columns=['latitude', 'longitude']).to_csv(
                    OUT / f'segment_{segment}_grid_{grid_deg:g}_{graph_type}_{mode}_route.csv', index=False)
                records.append(dict(segment=segment, grid_deg=grid_deg, graph=graph_type,
                                    route=mode, modeled_time_min=time_min, distance_km=km,
                                    n_waypoints=len(xy), expanded=expanded,
                                    max_turn_deg=max_turn, turns_over_30=turns30,
                                    direct_edge_selected=len(route)==2,
                                    original_start_error_m=base.edge_geometry(xy[0],coords[start])[0],
                                    original_end_error_m=base.edge_geometry(xy[-1],coords[goal])[0]))
            d, w = results['distance'], results['wind_aware']
            advantage = (d[0]-w[0])*60
            if advantage < -0.01:
                print('WARNING: wind-aware route is slower under common evaluator; '
                      'investigate time-dependent FIFO/search optimality.')
            print(f'  distance {d[0]:.5f} min / {d[1]:.2f} km; '
                  f'wind-aware {w[0]:.5f} min / {w[1]:.2f} km; '
                  f'advantage {advantage:.3f} s')
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(obs.G_LONG, obs.G_LAT, color='0.75', lw=1, label='Observed science track')
            for mode, color in [('distance','black'), ('wind_aware','red')]:
                xy = results[mode][2]
                ax.plot(xy[:,1], xy[:,0], color=color, lw=1.8, label=mode)
            ax.scatter([coords[start,1],coords[goal,1]],
                       [coords[start,0],coords[goal,0]], c=['green','blue'], s=45, zorder=5)
            ax.set(xlabel='Longitude (degrees E)', ylabel='Latitude (degrees N)',
                   title=f'Step 19 | segment {segment} | {graph_type} | grid {grid_deg:g} | {advantage:.2f} s')
            ax.legend(); ax.grid(alpha=0.2); fig.tight_layout()
            fig.savefig(OUT/f'segment_{segment}_grid_{grid_deg:g}_{graph_type}.png',dpi=180)
            plt.close(fig)
    detail = pd.DataFrame(records)
    detail.to_csv(OUT/'step19_route_details.csv', index=False)
    paired = detail.pivot(index=['segment','grid_deg','graph'], columns='route',
                          values='modeled_time_min').reset_index()
    paired['wind_aware_advantage_seconds'] = 60*(paired['distance']-paired['wind_aware'])
    paired['saving_pct'] = 100*(paired['distance']-paired['wind_aware'])/paired['distance']
    paired.to_csv(OUT/'step19_paired_comparison.csv', index=False)
    print('\n',paired.to_string(index=False))
    print('\nSaved:',OUT)
    print('CAUTION: expanded edges are sampled along their lengths; fixed pressure and TAS.')
    print('CAUTION: time-dependent Dijkstra assumes FIFO; this is not formally certified.')
    print('CAUTION: no turn radius, airspace, fuel, or science-mission constraints.')

if __name__ == '__main__':
    main()
