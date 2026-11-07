
from pathlib import Path
import importlib.util
import time
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

PROJECT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
CODE_DIR = PROJECT.parent / 'Code'
STEP26 = CODE_DIR / 'attrex_step26_time_dependent_corridor.py'
GRIB = PROJECT / 'ERA5_ATTREX_20140311_20140312_expanded.grib'
OBS = PROJECT / 'analysis_step23' / '20140311_segment_2798_observations.csv'
OUT = PROJECT / 'analysis_step27'
WIDTHS_KM = (100,)  # Optional: (50, 100); larger widths substantially increase runtime
N_TRIALS = 24       # Increase to 100+ for more stable scenario percentiles; expect long runtime
SEED = 279827
SEARCH_SPACING_KM = 5.0
VALIDATION_SPACING_KM = 1.0
# Step 24: ERA5 minus aircraft bias, RMSE. Treat residual std as a scenario scale,
# NOT a verified off-track uncertainty distribution.
BIAS_ERA_MINUS_OBS = np.array([-1.81362, -0.84687])
RMSE_COMPONENT = np.array([3.37415, 5.27545])
RESIDUAL_STD = np.sqrt(np.maximum(0, RMSE_COMPONENT**2 - BIAS_ERA_MINUS_OBS**2))
CORRELATION_KM = (50., 150.)  # assumed, not estimated from aircraft track
TEMPORAL_CORRELATION_MIN = 60.  # assumed
APPLY_MEAN_BIAS_CORRECTION = True


def load_script(path, name):
    if not path.is_file():
        raise FileNotFoundError(f'Required script missing: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PerturbedWind:
    """Smooth Gaussian random perturbation, deterministic for each trial.

    Gaussian RBFs yield continuous spatial fields with approximate length scales.
    Temporal variation is a smoothly blended independent realization. Each
    component is standardized at sampled locations to approximately its target
    scale; not a fitted covariance or uncertainty posterior.
    """
    def __init__(self, original, rng, corr_km, center, depart, bias_correct=True):
        self.original = original
        self.times = original.times
        self.lat = original.lat
        self.lon = original.lon
        self.center = np.asarray(center, float)
        self.depart = float(depart)
        self.length_km = float(corr_km)
        self.bias = -BIAS_ERA_MINUS_OBS if bias_correct else np.zeros(2)
        self.std = RESIDUAL_STD
        # Independent, smoothly varying spatial random fields at two times.
        # Locations extend beyond the corridor to reduce boundary artifacts.
        self.centers = np.column_stack((rng.uniform(30.0, 35.0, 35),
                                        rng.uniform(152.0, 160.0, 35)))
        self.coeff = rng.normal(size=(2, 35, 2))
        self.coeff /= np.sqrt(35.)
        self.phase = rng.uniform(0, 2*np.pi)
        self._norm = self._estimate_normalization()

    def _raw(self, lat, lon):
        dy = (self.centers[:, 0]-lat)*111.2
        dx = (self.centers[:, 1]-lon)*111.2*np.cos(np.deg2rad(lat))
        k = np.exp(-0.5*(dx*dx+dy*dy)/(self.length_km**2))
        return np.einsum('k,tkc->tc', k, self.coeff)

    def _estimate_normalization(self):
        # Standardize around route corridor, rather than arbitrary global grid.
        samples = np.array([self._raw(la, lo) for la in np.linspace(31.0, 34.0, 9)
                            for lo in np.linspace(153.5, 158.5, 13)])
        scales = np.std(samples.reshape(-1, 2), axis=0)
        return np.maximum(scales, 0.1)

    def at(self, t, lat, lon):
        base = self.original.at(t, lat, lon)
        if not np.isfinite(base).all():
            return base
        fields = self._raw(lat, lon)
        # Smooth temporal interpolation of two spatial realizations.
        angle = self.phase + (t-self.depart)/(TEMPORAL_CORRELATION_MIN*60.)
        noise = (np.cos(angle)*fields[0] + np.sin(angle)*fields[1])/self._norm
        return base + self.bias + self.std*noise


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mod = load_script(STEP26, 'attrex_step26_step27')
    base = mod.import_base()
    if not GRIB.is_file(): raise FileNotFoundError(GRIB)
    if not OBS.is_file(): raise FileNotFoundError(OBS)
    obs = pd.read_csv(OBS).sort_values('TIME_UTC').dropna(
        subset=['TIME_UTC','G_LAT','G_LONG','P','TAS'])
    a = obs[['G_LAT','G_LONG']].iloc[0].to_numpy(float)
    b = obs[['G_LAT','G_LONG']].iloc[-1].to_numpy(float)
    depart = pd.Timestamp('2014-03-11', tz='UTC').timestamp()+float(obs.TIME_UTC.iloc[0])
    pressure, tas = float(obs.P.median()), float(obs.TAS.median())
    ds = xr.open_dataset(GRIB, engine='cfgrib', backend_kwargs={'indexpath':''})
    reference = base.Wind(ds, pressure)
    if not (reference.times[0] <= depart < reference.times[-1]):
        raise ValueError('Departure outside ERA5 temporal support')
    rng = np.random.default_rng(SEED)
    records = []
    print('Step 27 | segment 2798 | paired field perturbation and REOPTIMIZATION')
    print('Trials:', N_TRIALS, '| widths:', WIDTHS_KM)
    print('Scenario residual std U/V:', RESIDUAL_STD, 'm/s')
    print('Assumed spatial scales:', CORRELATION_KM, 'km')
    print('Caution: scenario parameters are not independently calibrated.')
    for width in WIDTHS_KM:
        coords, adj, start, goal = mod.build_corridor(base, a, b, width)
        print(f'Width {width} km: {len(coords)} nodes; {sum(map(len,adj))} directed edges', flush=True)
        for corr in CORRELATION_KM:
            for trial in range(N_TRIALS):
                t0 = time.perf_counter()
                wind = PerturbedWind(reference, rng, corr, (a+b)/2, depart,
                                     APPLY_MEAN_BIAS_CORRECTION)
                base.MAX_LEG_KM = SEARCH_SPACING_KM
                distance_route, exp_d = mod.solve(base,coords,adj,start,goal,depart,wind,tas,'distance')
                wind_route, exp_w = mod.solve(base,coords,adj,start,goal,depart,wind,tas,'time')
                # Reevaluate BOTH routes with the SAME perturbed field at 1 km.
                d_time,d_km = mod.eval_route(base,coords,distance_route,depart,wind,tas,
                                              VALIDATION_SPACING_KM)
                w_time,w_km = mod.eval_route(base,coords,wind_route,depart,wind,tas,
                                              VALIDATION_SPACING_KM)
                adv = d_time-w_time
                same = distance_route == wind_route
                row = dict(width_km=width, correlation_km=corr, trial=trial,
                           distance_time_s=d_time, wind_aware_time_s=w_time,
                           advantage_s=adv, distance_km=d_km, wind_aware_km=w_km,
                           same_route=same, distance_waypoints=len(distance_route),
                           wind_waypoints=len(wind_route), expansions_distance=exp_d,
                           expansions_wind=exp_w, runtime_s=time.perf_counter()-t0,
                           finite=np.isfinite([d_time,w_time]).all())
                records.append(row)
                print(f'  width={width:3d} corr={corr:5.0f} trial={trial+1:3d}/{N_TRIALS} '
                      f'adv={adv:9.4f}s same={same} elapsed={row["runtime_s"]:.1f}s',flush=True)
                if not same:
                    for name, route in [('distance',distance_route),('wind_aware',wind_route)]:
                        pd.DataFrame(coords[route],columns=['latitude','longitude']).to_csv(
                            OUT/f'width{width}_corr{int(corr)}_trial{trial:03d}_{name}.csv',index=False)
        pd.DataFrame(records).to_csv(OUT/'step27_trials.csv',index=False)
    df = pd.DataFrame(records)
    summary = []
    for (width,corr),g in df.groupby(['width_km','correlation_km']):
        finite = g[g.finite]
        v = finite.advantage_s.to_numpy(float)
        summary.append(dict(width_km=width,correlation_km=corr,n_trials=len(g),
            n_finite=len(finite),n_different_routes=int((~finite.same_route).sum()),
            fraction_different_routes=float((~finite.same_route).mean()) if len(finite) else np.nan,
            median_advantage_s=float(np.median(v)) if len(v) else np.nan,
            p2_5_s=float(np.percentile(v,2.5)) if len(v) else np.nan,
            p97_5_s=float(np.percentile(v,97.5)) if len(v) else np.nan,
            max_advantage_s=float(np.max(v)) if len(v) else np.nan,
            negative_advantage_count=int((v < -0.05).sum())))
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT/'step27_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,5))
    for (width,corr),g in df.groupby(['width_km','correlation_km']):
        ax.hist(g.advantage_s[g.finite],bins=15,alpha=.45,
                label=f'±{width} km / corr {corr:g} km')
    ax.axvline(0,color='black',lw=1)
    ax.set(xlabel='Distance minus wind-aware modeled travel time (s)',
           ylabel='Scenario count',title='Step 27 | reoptimized wind-field uncertainty')
    ax.legend();fig.tight_layout();fig.savefig(OUT/'step27_advantage_histogram.png',dpi=180)
    plt.close(fig)
    print('\nSCENARIO SUMMARY (not confidence intervals):\n',summary.to_string(index=False))
    print('\nSaved:',OUT)
    print('WARNING: correlated field is a stylized assumption, not a fitted error model.')
    print('WARNING: direct edge, fixed pressure/TAS, no turn or mission constraints.')
    print('WARNING: solver assumes FIFO; a negative advantage may indicate search/evaluation mismatch.')
    print('WARNING: 24 trials are exploratory; percentile estimates are unstable.')

if __name__ == '__main__':
    main()
