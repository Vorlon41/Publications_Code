
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
CODE = ROOT.parent / 'Code'
OUT = ROOT / 'analysis_step38'
ERA5 = ROOT / 'analysis_step31' / 'ERA5_ATTREX_March11_12_2014.nc'
OBS = ROOT / 'analysis_step23' / '20140311_segment_3595_observations.csv'
WIDTHS = (0, 50, 100)
SEARCH_SPACING_KM = 2.0
VALIDATION_SPACINGS_KM = (2.0, 1.0, 0.5, 0.25)
STATION_STEP_KM = 100.0  # coarse exploratory grid; not grid-converged
LATERAL_STEP_KM = 50.0
FIFO_SAMPLES = 30

def load_module(filename, alias):
    path = CODE / filename
    if not path.exists():
        raise FileNotFoundError(f'Missing dependency: {path}')
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def collapse_collinear(xy, geometry, tolerance_m=1.0):
    """For output only: remove intermediate vertices within tolerance of geodesic.
    Retain actual optimizer geometry for travel-time calculations.
    """
    xy = np.asarray(xy, float)
    if len(xy) <= 2:
        return xy
    keep = [xy[0]]
    i = 0
    while i < len(xy) - 1:
        j = len(xy) - 1
        while j > i + 1:
            a, b = xy[i], xy[j]
            length = geometry.arc(a, b)
            if length < 1e-5:
                j -= 1
                continue
            # A conservative geometric check: sum of two geodesic distances
            # minus direct geodesic distance (not an exact cross-track distance).
            excess = max(geometry.arc(a, p) + geometry.arc(p, b) - length
                         for p in xy[i+1:j])
            if excess <= tolerance_m:
                break
            j -= 1
        keep.append(xy[j]); i = j
    return np.asarray(keep)

def main():
    if not ERA5.exists() or not OBS.exists():
        raise FileNotFoundError(f'Check ERA5={ERA5.exists()} OBS={OBS.exists()}')
    OUT.mkdir(parents=True, exist_ok=True)
    graph = load_module('attrex_step26_time_dependent_corridor.py', 'step38graph')
    ev = load_module('attrex_step36_era5_evaluator_regression.py', 'step38eval')
    search = load_module('attrex_step37_corrected_route_search.py', 'step38search')
# Configure the imported Step 37 optimizer
    search.SEARCH_SPACING_KM = SEARCH_SPACING_KM

    print(
        f"Optimization integration spacing: "
        f"{search.SEARCH_SPACING_KM} km"
    )
    base = graph.import_base()
    graph.STATION_STEP_KM = STATION_STEP_KM
    graph.LATERAL_STEP_KM = LATERAL_STEP_KM

    obs = pd.read_csv(OBS).dropna(subset=['TIME_UTC','G_LAT','G_LONG','P','TAS'])
    obs = obs.sort_values('TIME_UTC').reset_index(drop=True)
    a = obs[['G_LAT','G_LONG']].iloc[0].to_numpy(float)
    b = obs[['G_LAT','G_LONG']].iloc[-1].to_numpy(float)
    depart = pd.Timestamp('2014-03-11',tz='UTC').timestamp() + float(obs.TIME_UTC.iloc[0])
    pressure = float(obs.P.median()); tas = float(obs.TAS.median())
    print('Departure:', pd.to_datetime(depart, unit='s', utc=True))
    print('Median pressure:',pressure,'hPa; TAS:',tas,'m/s')
    # ============================================================
    # Load combined ERA5 data safely
    # ============================================================

    with xr.open_dataset(ERA5) as ds:

        print("\nLoading combined ERA5 dataset...")
        print("Original dimensions:", dict(ds.sizes))

        if ds.sizes.get("time") != 48:
            raise ValueError(
                "Expected 48 hourly ERA5 timestamps "
                "covering March 11–12, 2014."
            )

        if not {"u", "v"}.issubset(ds.data_vars):
            raise ValueError("ERA5 dataset missing U or V wind")

        # Load arrays into memory BEFORE sorting.
        wind_ds = ds[["u", "v"]].load()

        print("ERA5 wind arrays loaded into memory.")
        print("Wind dimensions:", dict(wind_ds.sizes))

        # Initialize the Step 36 wind interpolator.
        wind = ev.Wind(wind_ds, pressure)

    print("ERA5 wind interpolator initialized.")

    print(
        "ERA5 bounds:",
        pd.to_datetime(
            wind.times[[0, -1]],
            unit="s",
            utc=True
        )
    )
    if not wind.times[0] <= depart <= wind.times[-1]:
        raise ValueError('Departure outside ERA5 time coverage')
    if depart + ev.arc(a,b)/max(1,tas-60) > wind.times[-1]:
        print('WARNING: conservative travel-time horizon exceeds ERA5; invalid edges will be rejected')

    details=[]; paired=[]
    for width in WIDTHS:
        print(f'\nRunning corridor +/-{width} km', flush=True)
        coords, adj, start, goal = graph.build_corridor(base, a, b, width)
        if (np.any(coords[:,0] < wind.lat.min()) or np.any(coords[:,0] > wind.lat.max())
            or np.any(coords[:,1] < wind.lon.min()) or np.any(coords[:,1] > wind.lon.max())):
            raise ValueError(f'Corridor +/-{width} km extends outside ERA5 grid')
        checked, violations = search.audit_fifo(coords, adj, depart, tas, wind, ev)
        print('FIFO sampled:',checked,'violations:',violations)
        if violations:
            print('WARNING: sampled FIFO violation; skipping this corridor (no optimality claim)')
            continue
        distance_path = search.shortest_distance(coords, adj, start, goal, ev)
        try:
            wind_path, expansions = search.shortest_time(coords, adj, start, goal, depart, tas, wind, ev)
        except RuntimeError as exc:
            print('No valid wind-aware route:',exc)
            continue
        routes={'distance':distance_path,'wind_aware':wind_path}
        for label, route in routes.items():
            xy=coords[route]
            pd.DataFrame(xy,columns=['latitude','longitude']).to_csv(
                OUT/f'width{width}_{label}_route.csv',index=False)
            simplified=collapse_collinear(xy,ev)
            pd.DataFrame(simplified,columns=['latitude','longitude']).to_csv(
                OUT/f'width{width}_{label}_simplified_FOR_DISPLAY.csv',index=False)
            for spacing in VALIDATION_SPACINGS_KM:
                sec, km, n = ev.evaluate(xy,spacing,depart,tas,wind)
                details.append(dict(width_km=width,route=label,spacing_km=spacing,
                                    time_s=sec,distance_km=km,waypoints=len(route),
                                    simplified_waypoints=len(simplified),sublegs=n,
                                    fifo_checked=checked,fifo_violations=violations,
                                    expanded_wind_search=expansions))
        for spacing in VALIDATION_SPACINGS_KM:
            d=next(r for r in details if r['width_km']==width and r['route']=='distance' and r['spacing_km']==spacing)
            w=next(r for r in details if r['width_km']==width and r['route']=='wind_aware' and r['spacing_km']==spacing)
            advantage=d['time_s']-w['time_s']
            paired.append(dict(width_km=width,spacing_km=spacing,
                               distance_time_s=d['time_s'],wind_aware_time_s=w['time_s'],
                               advantage_s=advantage,
                               saving_pct=100*advantage/d['time_s'] if np.isfinite(d['time_s']) else np.nan,
                               distance_km=d['distance_km'],wind_aware_km=w['distance_km'],
                               same_nodes=distance_path==wind_path,
                               fifo_checked=checked,fifo_violations=violations))
        fig,ax=plt.subplots(figsize=(9,6))
        ax.plot(obs.G_LONG,obs.G_LAT,color='0.7',lw=0.8,label='Observed science track')
        for label,color in [('distance','black'),('wind_aware','red')]:
            xy=coords[routes[label]]
            ax.plot(xy[:,1],xy[:,0],color=color,lw=1.8,label=label)
        ax.scatter([a[1],b[1]],[a[0],b[0]],c=['green','blue'],s=35,zorder=5)
        ax.set(xlabel='Longitude (deg E)',ylabel='Latitude (deg N)',
               title=f'Step 38 | segment 3595 | corridor +/-{width} km')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(OUT/f'width{width}_routes.png',dpi=180);plt.close(fig)
        finest=next(r for r in paired if r['width_km']==width and r['spacing_km']==0.25)
        print('0.25-km advantage (s):',finest['advantage_s'], 'same nodes:',finest['same_nodes'])

    pd.DataFrame(details).to_csv(OUT/'step38_route_details.csv',index=False)
    result=pd.DataFrame(paired)
    result.to_csv(OUT/'step38_paired_comparison.csv',index=False)
    print('\n',result.to_string(index=False) if len(result) else 'No valid route comparisons')
    print('\nSaved:',OUT)
    print('CAUTION: coarse grid, conditional FIFO, fixed pressure/TAS; no aircraft/mission constraints.')
    print('CAUTION: no operational or observed flight-time savings are inferred.')

if __name__ == '__main__':
    main()
