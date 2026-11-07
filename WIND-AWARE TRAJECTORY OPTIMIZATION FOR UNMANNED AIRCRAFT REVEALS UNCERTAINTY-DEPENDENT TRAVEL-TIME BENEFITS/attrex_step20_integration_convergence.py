
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX')
CODE_DIR = PROJECT / 'Code'
STEP18 = CODE_DIR / 'attrex_step18_exact_endpoint_astar.py'
if not STEP18.is_file():
    raise FileNotFoundError(f'Step 18 source missing: {STEP18}')
spec = importlib.util.spec_from_file_location('attrex_step18', STEP18)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
ROOT = base.ROOT
IN = ROOT / 'analysis_step19'
OUT = ROOT / 'analysis_step20'
SEGMENT = 1450
GRID_DEG = 0.25
GRAPH = 'expanded'
SPACINGS_KM = (25., 10., 5., 2., 1.)
SIMPLIFICATION_TOLERANCES_M = (1., 25., 100., 500.)
EARTH = 6371008.8


def route_file(mode):
    return IN / f'segment_{SEGMENT}_grid_{GRID_DEG:g}_{GRAPH}_{mode}_route.csv'


def route_distance_km(xy):
    return sum(base.edge_geometry(a,b)[0] for a,b in zip(xy[:-1],xy[1:]))/1000.


def travel_fine(a, b, departure, wind, tas, spacing_km):
    """Midpoint wind sampling with self-consistent subleg arrival time.

    Matches Step 18's per-subleg update rule except for user-selected spacing.
    The interpolation between coordinate waypoints is linear in lat/lon, as in
    Step 18; this is not a spherical geodesic interpolation.
    """
    distance, _ = base.edge_geometry(a, b)
    if distance < 0.01:
        return 0.
    n = max(1, int(np.ceil(distance/(spacing_km*1000.))))
    elapsed = 0.
    for j in range(n):
        p = a+(b-a)*(j/n)
        q = a+(b-a)*((j+1)/n)
        length, heading = base.edge_geometry(p,q)
        mid = (p+q)*0.5
        dt = length/tas
        for _ in range(3):
            w = wind.at(departure+elapsed+dt/2., *mid)
            if not np.all(np.isfinite(w)):
                return np.inf
            along = float(w@heading)
            cross = float(-w[0]*heading[1]+w[1]*heading[0])
            if abs(cross) >= tas:
                return np.inf
            gs = np.sqrt(tas*tas-cross*cross)+along
            if gs <= 1.:
                return np.inf
            dt = length/gs
        elapsed += dt
    return elapsed


def evaluate_fine(xy, depart, wind, tas, spacing_km):
    elapsed = 0.
    for a,b in zip(xy[:-1],xy[1:]):
        dt = travel_fine(a,b,depart+elapsed,wind,tas,spacing_km)
        if not np.isfinite(dt):
            return np.inf
        elapsed += dt
    return elapsed


def simplify_route(xy, tolerance_m):
    """Ramer-Douglas-Peucker in a local equirectangular projection.

    Only used as a geometry/subdivision diagnostic, not an aircraft route.
    """
    ref_lat = np.deg2rad(np.mean(xy[:,0]))
    xy_m = np.column_stack((np.deg2rad(xy[:,1])*EARTH*np.cos(ref_lat),
                            np.deg2rad(xy[:,0])*EARTH))
    keep = {0,len(xy)-1}
    stack = [(0,len(xy)-1)]
    while stack:
        lo,hi = stack.pop()
        if hi-lo < 2:
            continue
        a,b = xy_m[lo],xy_m[hi]
        v = b-a
        p = xy_m[lo+1:hi]
        if np.dot(v,v) < 1e-12:
            dist = np.linalg.norm(p-a,axis=1)
        else:
            t = np.clip(((p-a)@v)/np.dot(v,v),0.,1.)
            dist = np.linalg.norm(p-(a+t[:,None]*v),axis=1)
        ix = int(np.argmax(dist))
        if dist[ix] > tolerance_m:
            k = lo+1+ix
            keep.add(k)
            stack.extend([(lo,k),(k,hi)])
    return xy[sorted(keep)]


def main():
    import xarray as xr
    OUT.mkdir(parents=True,exist_ok=True)
    if not base.GRIB.exists():
        raise FileNotFoundError(f'ERA5 file missing: {base.GRIB}')
    paths = {m:route_file(m) for m in ('distance','wind_aware')}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f'Step 19 route missing: {path}')
    routes = {m:pd.read_csv(p)[['latitude','longitude']].to_numpy(dtype=float)
              for m,p in paths.items()}
    ds = xr.open_dataset(base.GRIB,engine='cfgrib',backend_kwargs={'indexpath':''})
    _,_,_,_,depart,pressure,tas,_ = base.load_case(ds,SEGMENT,GRID_DEG)
    wind = base.Wind(ds,pressure)
    if not (wind.times[0]<=depart<=wind.times[-1]):
        raise ValueError('Flight departure outside ERA5 time coverage')
    print('ERA5:',base.GRIB)
    print(f'Segment {SEGMENT}: median pressure {pressure:.2f} hPa; TAS {tas:.3f} m/s')
    print('IMPORTANT: modeled values, not observed flight-time savings')
    rows = []
    for tol in (None,)+SIMPLIFICATION_TOLERANCES_M:
        for mode,xy in routes.items():
            current = xy if tol is None else simplify_route(xy,tol)
            label = 'original' if tol is None else f'RDP_{tol:g}m'
            pd.DataFrame(current,columns=['latitude','longitude']).to_csv(
                OUT/f'{mode}_{label}_coordinates.csv',index=False)
            for spacing in SPACINGS_KM:
                seconds = evaluate_fine(current,depart,wind,tas,spacing)
                rows.append(dict(segment=SEGMENT,graph=GRAPH,grid_deg=GRID_DEG,
                                 representation=label,tolerance_m=np.nan if tol is None else tol,
                                 route=mode,spacing_km=spacing,waypoints=len(current),
                                 distance_km=route_distance_km(current),time_s=seconds))
                print(f'{label:>12} {mode:>10} spacing {spacing:>5g} km '
                      f'points {len(current):>3} time {seconds:.6f} s')
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT/'step20_integration_details.csv',index=False)
    paired = detail.pivot(index=['representation','spacing_km'],columns='route',values='time_s').reset_index()
    paired['advantage_s']=paired['distance']-paired['wind_aware']
    paired.to_csv(OUT/'step20_paired_convergence.csv',index=False)
    print('\nPaired convergence (positive = wind-aware faster):')
    print(paired.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('\nOriginal route geometry:')
    for mode,xy in routes.items():
        print(mode,'waypoints',len(xy),'length_km',f'{route_distance_km(xy):.9f}')
    fig,ax=plt.subplots(figsize=(8.5,5))
    for label,grp in paired.groupby('representation'):
        grp=grp.sort_values('spacing_km')
        ax.plot(grp.spacing_km,grp.advantage_s,'o-',label=label)
    ax.axhline(0,color='0.3',lw=1)
    ax.set(xlabel='Maximum integration subleg length (km)',
           ylabel='Distance time − wind-aware time (s)',
           title='Step 20 | segment 1450 | integration and waypoint sensitivity')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(OUT/'step20_convergence.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,8))
    for mode,color in (('distance','black'),('wind_aware','red')):
        xy=routes[mode]
        ax.plot(xy[:,1],xy[:,0],'-o',lw=1.4,markersize=2.5,color=color,label=mode)
    ax.set(xlabel='Longitude (degrees E)',ylabel='Latitude (degrees N)',
           title='Step 20 | original expanded-graph routes')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(OUT/'step20_route_geometry.png',dpi=200);plt.close(fig)
    print('\nSaved:',OUT)
    print('Cautions: RDP changes route geometry; fine sampling is not an independent wind model.')
    print('Time-dependent FIFO and aircraft feasibility remain unverified.')


if __name__=='__main__':
    main()
