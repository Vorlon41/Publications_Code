
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
CODE = ROOT.parent / 'Code'
OUT = ROOT / 'analysis_step39'
ERA5 = ROOT / 'analysis_step31' / 'ERA5_ATTREX_March11_12_2014.nc'
OBS = ROOT / 'analysis_step23' / '20140311_segment_3595_observations.csv'
METRICS = ROOT / 'analysis_step32' / 'step32_wind_metrics.csv'
ROUTES = ROOT / 'analysis_step38'
WIDTHS = (50, 100)
CORR_KM = (50., 150.)  # hypothetical spatial correlation scales
N_TRIALS = 40          # exploratory; increase after runtime review
SPACING_KM = 5.0       # fast scenario sensitivity, not final convergence
CHECK_SPACING_KM = 1.0
CHECK_TRIALS = 3
TEMPORAL_SCALE_H = 1.0  # assumed
SEED = 3901
BIAS_CORRECT = True


def import_step36():
    p = CODE / 'attrex_step36_era5_evaluator_regression.py'
    if not p.exists():
        raise FileNotFoundError(p)
    spec = importlib.util.spec_from_file_location('attrex_step36_for39', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PerturbedWind:
    """Paired smooth random spatial/temporal error field. Not a fitted posterior."""
    def __init__(self, base, rng, corr_km, sigma, bias, departure):
        self.base = base
        self.corr_km = corr_km
        self.sigma = np.asarray(sigma, float)
        self.bias = np.asarray(bias, float)
        self.departure = departure
        self.times = base.times
        self.lat = base.lat
        self.lon = base.lon
        # Smooth random Fourier features with approximate correlation length.
        self.freq = rng.normal(size=(16, 2)) / corr_km
        self.phase = rng.uniform(0, 2*np.pi, 16)
        self.coeff = rng.normal(size=(2, 16, 2)) / np.sqrt(8)
        self.temporal_phase = rng.uniform(0, 2*np.pi)

    def at(self, t, p):
        w = self.base.at(t, p)
        if not np.all(np.isfinite(w)):
            return w
        # Local east/north projection about 23N, 151E, units km.
        x = (p[1] - 151.) * 111.2 * np.cos(np.deg2rad(23.))
        y = (p[0] - 23.) * 111.2
        basis = np.cos(self.freq @ np.array([x, y]) + self.phase)
        fields = np.einsum('k,tkc->tc', basis, self.coeff)
        theta = self.temporal_phase + (t-self.departure)/(TEMPORAL_SCALE_H*3600.)
        noise = np.cos(theta)*fields[0] + np.sin(theta)*fields[1]
        return w + self.bias + self.sigma*noise


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = import_step36()
    obs = pd.read_csv(OBS).dropna(subset=['TIME_UTC','P','TAS']).sort_values('TIME_UTC')
    depart = pd.Timestamp('2014-03-11',tz='UTC').timestamp() + float(obs.TIME_UTC.iloc[0])
    pressure = float(obs.P.median())
    tas = float(obs.TAS.median())
    metrics = pd.read_csv(METRICS).iloc[0]
    bias = np.array([metrics.bias_u_mps, metrics.bias_v_mps])
    rmse = np.array([metrics.rmse_u_mps, metrics.rmse_v_mps])
    sigma = np.sqrt(np.maximum(rmse**2 - bias**2, 0))
    # ERA5 - aircraft residual bias -> possible correction is negative bias.
    correction = -bias if BIAS_CORRECT else np.zeros(2)
    print('ERA5 residual bias U/V:', bias, 'm/s')
    print('Scenario sigma U/V:', sigma, 'm/s')
    print('Assumed correlation lengths:', CORR_KM, 'km')
    print('NOTE: covariance is hypothetical; off-track wind errors are not observed.')
    with xr.open_dataset(ERA5) as ds:
        if ds.sizes.get('time') != 48:
            raise ValueError('Expected 48 ERA5 hourly timestamps')
        wind_data = ds[['u','v']].load()  # avoids ScipyArrayWrapper.oindex
        base = ev.Wind(wind_data, pressure)
    if not base.times[0] <= depart <= base.times[-1]:
        raise ValueError('Departure outside ERA5 coverage')
    rng = np.random.default_rng(SEED)
    records = []
    nominal_by_width = {}
    for width in WIDTHS:
        route_data = {}
        for name in ('distance','wind_aware'):
            path = ROUTES / f'width{width}_{name}_route.csv'
            if not path.exists():
                raise FileNotFoundError(path)
            route_data[name] = pd.read_csv(path)[['latitude','longitude']].to_numpy(float)
        nominal = {}
        for name, route in route_data.items():
            nominal[name] = ev.evaluate(route, 0.5, depart, tas, base)[0]
        nominal_by_width[width] = nominal['distance'] - nominal['wind_aware']
        print(f'Width +/-{width} km; nominal advantage: {nominal_by_width[width]:.3f} s',flush=True)
        for corr in CORR_KM:
            for trial in range(N_TRIALS):
                perturb = PerturbedWind(base, rng, corr, sigma, correction, depart)
                spacing = SPACING_KM
                d = ev.evaluate(route_data['distance'], spacing, depart, tas, perturb)[0]
                w = ev.evaluate(route_data['wind_aware'], spacing, depart, tas, perturb)[0]
                records.append(dict(width_km=width,correlation_km=corr,trial=trial,
                                    spacing_km=spacing,distance_time_s=d,wind_aware_time_s=w,
                                    advantage_s=d-w,valid=np.isfinite(d) and np.isfinite(w)))
            print(f'  width {width}, corr {corr}: {N_TRIALS} paired trials done',flush=True)
    detail = pd.DataFrame(records)
    detail.to_csv(OUT/'step39_paired_trials.csv',index=False)
    rows=[]
    for (width,corr),g in detail.groupby(['width_km','correlation_km']):
        v=g[g.valid]
        a=v.advantage_s.to_numpy()
        rows.append(dict(width_km=width,correlation_km=corr,n_valid=len(v),n_total=len(g),
                         median_advantage_s=np.median(a) if len(a) else np.nan,
                         p025_advantage_s=np.percentile(a,2.5) if len(a) else np.nan,
                         p975_advantage_s=np.percentile(a,97.5) if len(a) else np.nan,
                         fraction_wind_aware_faster=np.mean(a>0) if len(a) else np.nan,
                         nominal_advantage_s=nominal_by_width[width]))
    summary=pd.DataFrame(rows)
    summary.to_csv(OUT/'step39_sensitivity_summary.csv',index=False)
    print('\n',summary.to_string(index=False))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax,width in zip(axes,WIDTHS):
        for corr in CORR_KM:
            x=detail[(detail.width_km==width)&(detail.correlation_km==corr)&detail.valid].advantage_s
            ax.hist(x,bins=12,alpha=.55,label=f'{corr:g}-km correlation')
        ax.axvline(0,color='black',ls='--',lw=1)
        ax.set(title=f'Corridor +/-{width} km',xlabel='Distance time minus wind-aware time (s)')
        ax.legend()
    axes[0].set_ylabel('Paired trials')
    fig.suptitle('Step 39: fixed-route wind uncertainty sensitivity (assumed covariance)')
    fig.tight_layout();fig.savefig(OUT/'step39_uncertainty.png',dpi=180);plt.close(fig)
    print('\nSaved:',OUT)
    print('CAUTION: fixed routes only; no reoptimization or validated off-track covariance.')
    print('CAUTION: scenario trials use 5-km integration; repeat at finer spacing before publication.')
    print('CAUTION: wind residuals may be nonstationary; model bias and mission feasibility unresolved.')

if __name__=='__main__':
    main()
