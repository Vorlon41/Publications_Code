
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
CODE = ROOT.parent / 'Code'
OUT = ROOT / 'analysis_step40'
ERA5 = ROOT / 'analysis_step31' / 'ERA5_ATTREX_March11_12_2014.nc'
OBS = ROOT / 'analysis_step23' / '20140311_segment_3595_observations.csv'
METRICS = ROOT / 'analysis_step32' / 'step32_wind_metrics.csv'
WIDTH_KM = 100
STATION_STEP_KM = 100.0
LATERAL_STEP_KM = 50.0
SEARCH_SPACING_KM = 2.0
VALIDATION_SPACING_KM = 1.0
CORR_KM = (50., 150.)
N_TRIALS = 20
SEED = 4001
BIAS_CORRECT = True
FIFO_EDGES_PER_TRIAL = 8


def load_script(filename, alias):
    path = CODE / filename
    if not path.is_file():
        raise FileNotFoundError(f'Missing dependency: {path}')
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    for path in (ERA5, OBS, METRICS):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT.mkdir(parents=True, exist_ok=True)
    graph = load_script('attrex_step26_time_dependent_corridor.py', 'attrex40graph')
    ev = load_script('attrex_step36_era5_evaluator_regression.py', 'attrex40eval')
    search = load_script('attrex_step37_corrected_route_search.py', 'attrex40search')
    uncertainty = load_script('attrex_step39_fixed_route_uncertainty.py', 'attrex40uncertainty')
    graph.STATION_STEP_KM = STATION_STEP_KM
    graph.LATERAL_STEP_KM = LATERAL_STEP_KM
    search.SEARCH_SPACING_KM = SEARCH_SPACING_KM
    search.FIFO_SAMPLES = FIFO_EDGES_PER_TRIAL
    print('Search integration:', search.SEARCH_SPACING_KM, 'km')

    obs = pd.read_csv(OBS).dropna(subset=['TIME_UTC','G_LAT','G_LONG','P','TAS'])
    obs = obs.sort_values('TIME_UTC').reset_index(drop=True)
    a = obs[['G_LAT','G_LONG']].iloc[0].to_numpy(float)
    b = obs[['G_LAT','G_LONG']].iloc[-1].to_numpy(float)
    depart = pd.Timestamp('2014-03-11', tz='UTC').timestamp() + float(obs.TIME_UTC.iloc[0])
    pressure = float(obs.P.median())
    tas = float(obs.TAS.median())
    metrics = pd.read_csv(METRICS).iloc[0]
    bias = np.array([metrics.bias_u_mps, metrics.bias_v_mps], float)
    rmse = np.array([metrics.rmse_u_mps, metrics.rmse_v_mps], float)
    sigma = np.sqrt(np.maximum(rmse**2 - bias**2, 0))
    correction = -bias if BIAS_CORRECT else np.zeros(2)
    print('Departure:', pd.to_datetime(depart, unit='s', utc=True))
    print('Pressure/TAS:', pressure, 'hPa /', tas, 'm/s')
    print('Assumed wind correction U/V:', correction, 'm/s; sigma U/V:', sigma, 'm/s')
    with xr.open_dataset(ERA5) as ds:
        if ds.sizes.get('time') != 48:
            raise ValueError('Expected 48-hour March 11–12 ERA5 dataset')
        wind_data = ds[['u','v']].load()  # load BEFORE Step 36 sorting
        base_wind = ev.Wind(wind_data, pressure)
    if not (base_wind.times[0] <= depart <= base_wind.times[-1]):
        raise ValueError('Departure outside ERA5 time coverage')

    coords, adj, start, goal = graph.build_corridor(graph.import_base(), a, b, WIDTH_KM)
    if (np.any(coords[:,0] < base_wind.lat.min()) or
        np.any(coords[:,0] > base_wind.lat.max()) or
        np.any(coords[:,1] < base_wind.lon.min()) or
        np.any(coords[:,1] > base_wind.lon.max())):
        raise ValueError('Corridor extends outside ERA5 spatial bounds')
    distance_path = search.shortest_distance(coords, adj, start, goal, ev)
    distance_xy = coords[distance_path]
    print('Graph nodes:', len(coords), 'edges:', sum(map(len,adj)))
    print('Distance baseline length:', sum(ev.arc(p,q) for p,q in zip(distance_xy[:-1],distance_xy[1:]))/1000, 'km')

    rng = np.random.default_rng(SEED)
    records = []
    for corr in CORR_KM:
        for trial in range(N_TRIALS):
            perturbed = uncertainty.PerturbedWind(base_wind, rng, corr, sigma, correction, depart)
            # This is a sampled diagnostic, not a FIFO certificate.
            checked, violations = search.audit_fifo(coords, adj, depart, tas, perturbed, ev)
            status = 'ok'
            route = None
            expanded = 0
            if violations:
                status = 'sampled_fifo_violation'
            else:
                try:
                    route, expanded = search.shortest_time(coords, adj, start, goal,
                                                            depart, tas, perturbed, ev)
                except RuntimeError:
                    status = 'no_valid_route'
            if route is not None:
                wind_xy = coords[route]
                d_search, d_km, _ = ev.evaluate(distance_xy, SEARCH_SPACING_KM, depart, tas, perturbed)
                w_search, w_km, _ = ev.evaluate(wind_xy, SEARCH_SPACING_KM, depart, tas, perturbed)
                d_fine, _, _ = ev.evaluate(distance_xy, VALIDATION_SPACING_KM, depart, tas, perturbed)
                w_fine, _, _ = ev.evaluate(wind_xy, VALIDATION_SPACING_KM, depart, tas, perturbed)
                if not all(np.isfinite(v) for v in (d_search,w_search,d_fine,w_fine)):
                    status = 'nonfinite_evaluation'
                advantage_search = d_search-w_search
                advantage_fine = d_fine-w_fine
                same_nodes = route == distance_path
                path_string = ';'.join(map(str,route))
            else:
                d_km=w_km=advantage_search=advantage_fine=np.nan
                same_nodes=False
                path_string=''
            records.append(dict(correlation_km=corr, trial=trial, status=status,
                                fifo_checked=checked, fifo_violations=violations,
                                expanded=expanded, search_spacing_km=SEARCH_SPACING_KM,
                                validation_spacing_km=VALIDATION_SPACING_KM,
                                distance_km=d_km, optimized_km=w_km,
                                extra_distance_km=w_km-d_km,
                                advantage_search_s=advantage_search,
                                advantage_validation_s=advantage_fine,
                                same_waypoint_sequence=same_nodes,
                                optimized_path_indices=path_string))
            print(f'corr={corr:g} km trial={trial+1:02d}/{N_TRIALS} '
                  f'status={status} fine_advantage={advantage_fine:.3f} s',flush=True)
            pd.DataFrame(records).to_csv(OUT/'step40_trial_results.csv',index=False)

    detail = pd.DataFrame(records)
    summary = []
    for corr, group in detail.groupby('correlation_km'):
        valid = group[(group.status=='ok') & np.isfinite(group.advantage_validation_s)]
        x = valid.advantage_validation_s.to_numpy(float)
        summary.append(dict(correlation_km=corr, n_total=len(group), n_valid=len(valid),
                            n_fifo_violations=int((group.status=='sampled_fifo_violation').sum()),
                            n_failed=int((group.status!='ok').sum()),
                            median_advantage_s=float(np.median(x)) if len(x) else np.nan,
                            p025_advantage_s=float(np.percentile(x,2.5)) if len(x) else np.nan,
                            p975_advantage_s=float(np.percentile(x,97.5)) if len(x) else np.nan,
                            fraction_positive=float(np.mean(x>0)) if len(x) else np.nan,
                            fraction_different_route=float(np.mean(~valid.same_waypoint_sequence)) if len(x) else np.nan,
                            median_extra_distance_km=float(np.median(valid.extra_distance_km)) if len(x) else np.nan))
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT/'step40_summary.csv',index=False)
    print('\n',summary.to_string(index=False))
    fig, ax = plt.subplots(figsize=(9,5))
    for corr in CORR_KM:
        x = detail.loc[(detail.correlation_km==corr)&(detail.status=='ok'), 'advantage_validation_s']
        if len(x): ax.hist(x,bins=10,alpha=.5,label=f'{corr:g}-km correlation')
    ax.axvline(0,color='black',ls='--',lw=1)
    ax.set(xlabel='Distance-route time minus reoptimized-route time (s)',ylabel='Trials',
           title='Step 40 | In-sample paired reoptimization under assumed wind errors')
    ax.legend();fig.tight_layout()
    fig.savefig(OUT/'step40_advantages.png',dpi=180)
    plt.close(fig)
    print('\nSaved:',OUT)
    print('CAUTION: 20 trials per scenario; covariance is assumed, not empirically calibrated.')
    print('CAUTION: search at 2 km; validation at 1 km; finer convergence remains necessary.')
    print('CAUTION: same-field reoptimization is in-sample; it does not measure forecast skill.')
    print('CAUTION: sampled FIFO audit is not proof; aircraft/mission constraints omitted.')

if __name__ == '__main__':
    main()
