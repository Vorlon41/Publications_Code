
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
OUT = ROOT / 'analysis_step35'
TAS = 50.0
SPACINGS = (5., 2., 1., 0.5, 0.25)


def sample_polyline(points, max_spacing_km):
    """Preserve every original vertex; never integrate across a corner."""
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError('Invalid route coordinates')
    if max_spacing_km <= 0:
        raise ValueError('Spacing must be positive')
    chunks = [p[0]]
    for a, b in zip(p[:-1], p[1:]):
        length = float(np.linalg.norm(b-a))
        if length <= 0:
            raise ValueError('Repeated consecutive waypoints')
        n = max(1, int(np.ceil(length/max_spacing_km)))
        for j in range(1, n+1):
            chunks.append(a+(b-a)*(j/n))
    return np.asarray(chunks)


def wind(xy, elapsed_s, scenario):
    if scenario == 'zero':
        return np.zeros(2)
    if scenario == 'corridor':
        y = float(xy[1])
        strength = 48.0*np.exp(-0.5*((y-30.0)/9.0)**2)
        return np.array([strength*(1+0.03*np.sin(elapsed_s/800.0)), 0.0])
    raise ValueError(scenario)


def evaluate(points, spacing_km, scenario):
    sampled = sample_polyline(points, spacing_km)
    total_m = float(np.linalg.norm(np.diff(sampled, axis=0), axis=1).sum()*1000)
    elapsed = 0.0
    for a, b in zip(sampled[:-1], sampled[1:]):
        delta = b-a
        length_km = np.linalg.norm(delta)
        heading = delta/length_km
        cross_heading = np.array([-heading[1], heading[0]])
        mid = (a+b)/2
        dt = length_km*1000/TAS
        for _ in range(30):
            w = wind(mid, elapsed+dt/2, scenario)
            along = float(w@heading)
            cross = float(w@cross_heading)
            if abs(cross) >= TAS:
                raise ValueError('Wind exceeds available airspeed for this track')
            groundspeed = np.sqrt(TAS*TAS-cross*cross)+along
            if groundspeed <= 0:
                raise ValueError('Nonpositive ground speed')
            new_dt = length_km*1000/groundspeed
            if abs(new_dt-dt) < 1e-11:
                dt = new_dt
                break
            dt = new_dt
        elapsed += dt
    return elapsed, total_m/1000, len(sampled)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    straight = np.array([[0.,0.],[100.,0.]])
    split4 = np.column_stack([np.linspace(0,100,5), np.zeros(5)])
    split16 = np.column_stack([np.linspace(0,100,17), np.zeros(17)])
    detour = np.array([[0.,0.],[20.,30.],[80.,30.],[100.,0.]])
    paths = {'straight_2pt':straight,'straight_5pt':split4,
             'straight_17pt':split16,'detour':detour}
    rows = []
    for scenario in ('zero','corridor'):
        for spacing in SPACINGS:
            for name, path in paths.items():
                t, dist, n = evaluate(path, spacing, scenario)
                geometric = float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
                analytic = geometric*1000/TAS if scenario == 'zero' else np.nan
                rows.append(dict(scenario=scenario, spacing_km=spacing, route=name,
                                 geometric_distance_km=geometric, sampled_distance_km=dist,
                                 sample_count=n, travel_time_s=t,
                                 analytic_zero_wind_s=analytic,
                                 zero_wind_error_s=t-analytic if scenario=='zero' else np.nan))
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT/'step35_details.csv',index=False)
    summary=[]
    for (scenario, spacing), g in detail.groupby(['scenario','spacing_km']):
        m = g.set_index('route')
        equivalence = max(abs(m.loc['straight_2pt','travel_time_s']-m.loc[k,'travel_time_s'])
                          for k in ('straight_5pt','straight_17pt'))
        advantage = m.loc['straight_2pt','travel_time_s']-m.loc['detour','travel_time_s']
        summary.append(dict(scenario=scenario,spacing_km=spacing,
                            direct_time_s=m.loc['straight_2pt','travel_time_s'],
                            detour_time_s=m.loc['detour','travel_time_s'],
                            detour_advantage_s=advantage,
                            max_equivalent_difference_s=equivalence,
                            max_distance_error_m=float(np.max(np.abs(g.geometric_distance_km-g.sampled_distance_km))*1000),
                            max_zero_wind_analytic_error_s=float(np.nanmax(np.abs(g.zero_wind_error_s))) if scenario=='zero' else np.nan))
    s = pd.DataFrame(summary).sort_values(['scenario','spacing_km'])
    s.to_csv(OUT/'step35_summary.csv',index=False)
    print('\nSTEP 35 RESULTS\n',s.to_string(index=False))
    assert s.max_distance_error_m.max() < 1e-6, 'Polyline length not conserved'
    assert s.max_equivalent_difference_s.max() < 0.01, 'Subdivision invariance failed'
    assert s[s.scenario=='zero'].max_zero_wind_analytic_error_s.max() < 1e-8, 'Zero-wind analytical test failed'
    assert (s[s.scenario=='zero'].detour_advantage_s < 0).all(), 'Zero-wind route ordering failed'
    assert (s[s.scenario=='corridor'].detour_advantage_s > 0).all(), 'Synthetic corridor ordering failed'
    fine = s[(s.scenario=='corridor') & (s.spacing_km.isin([0.5,0.25]))].sort_values('spacing_km')
    difference = abs(float(fine.iloc[0].detour_advantage_s)-float(fine.iloc[1].detour_advantage_s))
    print(f'\n0.25 vs 0.5 km corridor advantage change: {difference:.6f} seconds')
    print('PASS: route length, analytic zero-wind, equivalent routes, and synthetic ordering')
    print('NOTE: wind-corridor convergence should be judged separately; no forced pass threshold.')
    fig, ax = plt.subplots(figsize=(8,5))
    for scenario, g in s.groupby('scenario'):
        ax.plot(g.spacing_km, g.detour_advantage_s, marker='o', label=scenario)
    ax.axhline(0,color='black',lw=0.8)
    ax.set(xlabel='Maximum subleg length (km)',ylabel='Direct time minus detour time (s)',
           title='Step 35 — vertex-preserving integration convergence')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    fig.savefig(OUT/'step35_convergence.png',dpi=180)
    plt.close(fig)
    print('Saved:',OUT)
    print('CAUTION: synthetic Cartesian experiment; does not certify ERA5, FIFO, aircraft dynamics or routing optimality.')

if __name__=='__main__':
    main()
