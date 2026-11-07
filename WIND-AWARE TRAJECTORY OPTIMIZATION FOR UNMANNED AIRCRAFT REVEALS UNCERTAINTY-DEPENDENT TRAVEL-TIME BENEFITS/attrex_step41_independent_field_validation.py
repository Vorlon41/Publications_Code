
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
CODE = ROOT.parent / 'Code'
OUT = ROOT / 'analysis_step41'
ERA5 = ROOT / 'analysis_step31' / 'ERA5_ATTREX_March11_12_2014.nc'
OBS = ROOT / 'analysis_step23' / '20140311_segment_3595_observations.csv'
METRICS = ROOT / 'analysis_step32' / 'step32_wind_metrics.csv'
WIDTH_KM = 100
CORR_KM = (50., 150.)
N_TRIALS = 20
SEARCH_SPACING_KM = 2.0
VALIDATION_SPACING_KM = 1.0
STATION_STEP_KM = 100.0
LATERAL_STEP_KM = 50.0
FIFO_EDGES_PER_TRIAL = 8
BIAS_CORRECT = True
SEED = 4101


def import_script(name, alias):
    path = CODE / name
    if not path.exists():
        raise FileNotFoundError(f'Missing script: {path}')
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def summarize(group):
    good = group[(group.status == 'ok') & np.isfinite(group.out_sample_advantage_s)]
    def stats(column):
        a = good[column].to_numpy(dtype=float)
        if not len(a):
            return dict(median=np.nan, p025=np.nan, p975=np.nan, fraction_positive=np.nan)
        return dict(median=float(np.median(a)),
                    p025=float(np.percentile(a, 2.5)),
                    p975=float(np.percentile(a, 97.5)),
                    fraction_positive=float(np.mean(a > 0)))
    ins = stats('in_sample_advantage_s')
    outs = stats('out_sample_advantage_s')
    return dict(n_total=len(group), n_valid=len(good),
                n_failed=int((group.status != 'ok').sum()),
                n_sampled_fifo_violations=int((group.status == 'sampled_fifo_violation').sum()),
                median_in_sample_s=ins['median'],
                median_out_sample_s=outs['median'],
                p025_out_sample_s=outs['p025'],
                p975_out_sample_s=outs['p975'],
                fraction_out_sample_positive=outs['fraction_positive'],
                fraction_out_sample_negative=float(np.mean(good.out_sample_advantage_s < 0)) if len(good) else np.nan,
                fraction_different_path=float(np.mean(~good.same_baseline_nodes.astype(bool))) if len(good) else np.nan)


def main():
    for p in (ERA5, OBS, METRICS):
        if not p.exists():
            raise FileNotFoundError(p)
    OUT.mkdir(parents=True, exist_ok=True)
    graph = import_script('attrex_step26_time_dependent_corridor.py', 's41graph')
    ev = import_script('attrex_step36_era5_evaluator_regression.py', 's41eval')
    search = import_script('attrex_step37_corrected_route_search.py', 's41search')
    uncertainty = import_script('attrex_step39_fixed_route_uncertainty.py', 's41uncertainty')
    graph.STATION_STEP_KM = STATION_STEP_KM
    graph.LATERAL_STEP_KM = LATERAL_STEP_KM
    search.SEARCH_SPACING_KM = SEARCH_SPACING_KM
    search.FIFO_SAMPLES = FIFO_EDGES_PER_TRIAL

    obs = pd.read_csv(OBS).dropna(subset=['TIME_UTC','G_LAT','G_LONG','P','TAS'])
    obs = obs.sort_values('TIME_UTC').reset_index(drop=True)
    a = obs[['G_LAT','G_LONG']].iloc[0].to_numpy(float)
    b = obs[['G_LAT','G_LONG']].iloc[-1].to_numpy(float)
    depart = pd.Timestamp('2014-03-11', tz='UTC').timestamp() + float(obs.TIME_UTC.iloc[0])
    pressure, tas = float(obs.P.median()), float(obs.TAS.median())
    metrics = pd.read_csv(METRICS).iloc[0]
    bias = np.array([metrics.bias_u_mps, metrics.bias_v_mps], float)
    rmse = np.array([metrics.rmse_u_mps, metrics.rmse_v_mps], float)
    sigma = np.sqrt(np.maximum(rmse**2 - bias**2, 0))
    correction = -bias if BIAS_CORRECT else np.zeros(2)

    with xr.open_dataset(ERA5) as ds:
        if ds.sizes.get('time') != 48:
            raise ValueError('Expected 48 hourly timestamps, March 11–12')
        if not {'u','v'}.issubset(ds.data_vars):
            raise ValueError('Missing u or v')
        wind_data = ds[['u','v']].load()  # load before sort in Wind()
        base_wind = ev.Wind(wind_data, pressure)
    if not base_wind.times[0] <= depart <= base_wind.times[-1]:
        raise ValueError('Departure outside ERA5')

    coords, adj, start, goal = graph.build_corridor(graph.import_base(), a, b, WIDTH_KM)
    if (np.any(coords[:,0] < base_wind.lat.min()) or
        np.any(coords[:,0] > base_wind.lat.max()) or
        np.any(coords[:,1] < base_wind.lon.min()) or
        np.any(coords[:,1] > base_wind.lon.max())):
        raise ValueError('Corridor extends outside ERA5 domain')
    distance_path = search.shortest_distance(coords, adj, start, goal, ev)
    baseline = coords[distance_path]
    print('Segment 3595 | ±100-km corridor')
    print('Departure:', pd.to_datetime(depart, unit='s', utc=True))
    print('Search/validation spacing (km):', SEARCH_SPACING_KM, VALIDATION_SPACING_KM)
    print('Wind bias correction:', correction, 'sigma:', sigma)
    print('Graph nodes:', len(coords), 'directed edges:', sum(map(len, adj)))
    rng = np.random.default_rng(SEED)
    records = []
    for corr in CORR_KM:
        for trial in range(N_TRIALS):
            # Independent draws, shared ERA5 and shared bias correction.
            planning = uncertainty.PerturbedWind(base_wind, rng, corr, sigma, correction, depart)
            evaluation = uncertainty.PerturbedWind(base_wind, rng, corr, sigma, correction, depart)
            checked, violations = search.audit_fifo(coords, adj, depart, tas, planning, ev)
            status = 'ok'
            route = None
            expanded = 0
            if violations:
                status = 'sampled_fifo_violation'
            else:
                try:
                    route, expanded = search.shortest_time(
                        coords, adj, start, goal, depart, tas, planning, ev)
                except RuntimeError:
                    status = 'no_valid_route'
            in_adv = out_adv = out_d = out_w = extra_km = np.nan
            same_nodes = False
            path_indices = ''
            if route is not None:
                xy = coords[route]
                same_nodes = (route == distance_path)
                path_indices = ';'.join(map(str, route))
                d_plan, _, _ = ev.evaluate(baseline, VALIDATION_SPACING_KM, depart, tas, planning)
                w_plan, _, _ = ev.evaluate(xy, VALIDATION_SPACING_KM, depart, tas, planning)
                out_d, d_km, _ = ev.evaluate(baseline, VALIDATION_SPACING_KM, depart, tas, evaluation)
                out_w, w_km, _ = ev.evaluate(xy, VALIDATION_SPACING_KM, depart, tas, evaluation)
                if not all(np.isfinite(v) for v in (d_plan, w_plan, out_d, out_w)):
                    status = 'nonfinite_evaluation'
                else:
                    in_adv = d_plan - w_plan
                    out_adv = out_d - out_w
                    extra_km = w_km - d_km
            records.append(dict(correlation_km=corr, trial=trial + 1, status=status,
                                fifo_checked=checked, fifo_violations=violations,
                                search_expanded=expanded, same_baseline_nodes=same_nodes,
                                extra_distance_km=extra_km,
                                in_sample_advantage_s=in_adv,
                                out_sample_advantage_s=out_adv,
                                evaluation_distance_time_s=out_d,
                                evaluation_optimized_time_s=out_w,
                                optimized_path_indices=path_indices,
                                search_spacing_km=SEARCH_SPACING_KM,
                                validation_spacing_km=VALIDATION_SPACING_KM))
            pd.DataFrame(records).to_csv(OUT/'step41_trial_results.csv', index=False)
            print(f'corr={corr:g} trial={trial+1:02d}/{N_TRIALS} '
                  f'status={status} in={in_adv:.2f}s out={out_adv:.2f}s', flush=True)

    detail = pd.DataFrame(records)
    summary = pd.DataFrame([dict(correlation_km=corr, **summarize(g))
                            for corr, g in detail.groupby('correlation_km')])
    summary.to_csv(OUT/'step41_summary.csv', index=False)
    print('\n', summary.to_string(index=False))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, corr in zip(axes, CORR_KM):
        valid = detail[(detail.correlation_km == corr) & (detail.status == 'ok')]
        ax.scatter(valid.in_sample_advantage_s, valid.out_sample_advantage_s, alpha=.8)
        ax.axhline(0, color='black', ls='--', lw=1)
        ax.axvline(0, color='0.6', ls=':', lw=1)
        ax.set(title=f'{corr:g}-km assumed correlation',
               xlabel='Planning-field advantage (s)')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Independent-perturbation advantage (s)')
    fig.suptitle('Step 41 — in-sample versus separate-field evaluation')
    fig.tight_layout()
    fig.savefig(OUT/'step41_generalization.png', dpi=180)
    plt.close(fig)
    print('\nSaved:', OUT)
    print('CAUTION: independent random draws share ERA5, bias correction, and assumed covariance.')
    print('CAUTION: NOT independent observed meteorological validation or operational savings.')
    print('CAUTION: sampled FIFO is not global proof; fixed TAS/pressure, no mission constraints.')
    print('CAUTION: 20 trials/scenario, 1-km validation; further convergence required.')

if __name__ == '__main__':
    main()
