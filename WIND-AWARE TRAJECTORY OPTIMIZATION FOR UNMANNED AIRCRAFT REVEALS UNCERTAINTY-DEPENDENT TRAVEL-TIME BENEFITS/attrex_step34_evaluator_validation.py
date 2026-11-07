
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
OUT = ROOT / 'analysis_step34'
SPACINGS = (5., 2., 1., 0.5)
TAS = 50.0  # m/s, intentionally synthetic
TOL_S = 0.01

# All coordinates below are synthetic (x, y), in kilometers.
def canonicalize(points, tolerance_km=0.05):
    p = np.asarray(points, dtype=float)
    if len(p) < 2 or not np.isfinite(p).all():
        raise ValueError('Invalid route')
    a, b = p[0], p[-1]
    ab = b-a
    length = np.linalg.norm(ab)
    if length <= 0: raise ValueError('Identical endpoints')
    t = ((p-a) @ ab)/(length**2)
    lateral = np.abs(ab[0]*(p[:,1]-a[1])-ab[1]*(p[:,0]-a[0]))/length
    if np.all((t >= -1e-10) & (t <= 1+1e-10) & (lateral <= tolerance_km)) and np.all(np.diff(t) >= -1e-10):
        return p[[0, -1]]
    return p

def sample_route(points, spacing_km):
    p = canonicalize(points)
    lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)
    if np.any(lengths <= 0): raise ValueError('Zero-length route leg')
    cumulative = np.r_[0, np.cumsum(lengths)]
    stations = np.r_[np.arange(0, cumulative[-1], spacing_km), cumulative[-1]]
    output = []
    for s in stations:
        j = min(np.searchsorted(cumulative, s, side='right')-1, len(lengths)-1)
        f = (s-cumulative[j])/lengths[j]
        output.append(p[j] + f*(p[j+1]-p[j]))
    return np.asarray(output)

def wind(xy_km, elapsed_s, scenario):
    x, y = xy_km
    if scenario == 'zero': return np.array([0., 0.])
    if scenario == 'corridor':
        # Strong eastward tailwind only near y=30 km, smoothly tapering away.
        strength = 48.0*np.exp(-0.5*((y-30.)/9.)**2)
        # Modest time dependence to exercise evolving arrival-time propagation.
        return np.array([strength*(1+0.03*np.sin(elapsed_s/800.)), 0.])
    raise ValueError(scenario)

def travel_time(points, spacing_km, scenario):
    sampled = sample_route(points, spacing_km)
    elapsed = 0.0
    for a, b in zip(sampled[:-1], sampled[1:]):
        dxy = b-a
        dist_m = np.linalg.norm(dxy)*1000.
        heading = dxy/np.linalg.norm(dxy)
        # Midpoint integration with time propagation and fixed-point arrival-time update.
        dt = dist_m/TAS
        mid = (a+b)/2
        for _ in range(12):
            w = wind(mid, elapsed+dt/2, scenario)
            along = np.dot(w, heading)
            cross = np.dot(w, np.array([-heading[1], heading[0]]))
            if abs(cross) >= TAS: return np.nan
            speed = np.sqrt(TAS*TAS-cross*cross)+along
            if speed <= 0: return np.nan
            new_dt = dist_m/speed
            if abs(new_dt-dt) < 1e-10:
                dt = new_dt
                break
            dt = new_dt
        elapsed += dt
    return elapsed

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    straight = np.array([[0., 0.], [100., 0.]])
    split4 = np.column_stack([np.linspace(0,100,5), np.zeros(5)])
    split16 = np.column_stack([np.linspace(0,100,17), np.zeros(17)])
    detour = np.array([[0.,0.], [20.,30.], [80.,30.], [100.,0.]])
    cases = {'straight_2pt': straight, 'straight_5pt': split4,
             'straight_17pt': split16, 'detour': detour}
    rows=[]
    for scenario in ('zero','corridor'):
        for spacing in SPACINGS:
            for name, path in cases.items():
                rows.append(dict(scenario=scenario, spacing_km=spacing,
                                 route=name, canonical_points=len(canonicalize(path)),
                                 distance_km=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()),
                                 travel_time_s=travel_time(path,spacing,scenario)))
    df=pd.DataFrame(rows)
    df.to_csv(OUT/'step34_details.csv',index=False)
    records=[]
    for (scenario,spacing), group in df.groupby(['scenario','spacing_km']):
        times=group.set_index('route').travel_time_s
        discrepancy=max(abs(times['straight_2pt']-times[k]) for k in ('straight_5pt','straight_17pt'))
        advantage=times['straight_2pt']-times['detour']
        records.append(dict(scenario=scenario,spacing_km=spacing,
                            max_equivalent_route_difference_s=discrepancy,
                            direct_time_s=times['straight_2pt'],detour_time_s=times['detour'],
                            detour_advantage_s=advantage,
                            detour_preserved=len(canonicalize(detour))>2))
    summary=pd.DataFrame(records).sort_values(['scenario','spacing_km'])
    summary.to_csv(OUT/'step34_summary.csv',index=False)
    fine=summary[summary.spacing_km==min(SPACINGS)]
    equivalence_pass=bool((fine.max_equivalent_route_difference_s<TOL_S).all())
    distinct_pass=bool(fine.detour_preserved.all())
    corridor_pass=bool((fine[fine.scenario=='corridor'].detour_advantage_s>0).all())
    zero_pass=bool((fine[fine.scenario=='zero'].detour_advantage_s<0).all())
    print('\n',summary.to_string(index=False))
    print('\nEquivalent route (<0.01 s):', 'PASS' if equivalence_pass else 'FAIL')
    print('Distinct geometry preserved:', 'PASS' if distinct_pass else 'FAIL')
    print('Synthetic tailwind detour faster:', 'PASS' if corridor_pass else 'FAIL')
    print('Zero-wind detour slower:', 'PASS' if zero_pass else 'FAIL')
    fig,ax=plt.subplots(figsize=(9,5))
    for name in ('straight_2pt','detour'):
        p=cases[name];ax.plot(p[:,0],p[:,1],marker='o',label=name)
    ax.set(xlabel='Synthetic east displacement (km)',ylabel='Synthetic north displacement (km)',title='Step 34 — Controlled synthetic route geometries')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(OUT/'step34_routes.png',dpi=180);plt.close(fig)
    print('\nSaved:',OUT)
    print('CAUTION: Cartesian synthetic test only; no ERA5, aircraft dynamics, or FIFO certification.')
    print('CAUTION: passing canonicalization tests does not prove arbitrary-edge integration invariance.')

if __name__=='__main__':main()
