
from pathlib import Path
from datetime import datetime, timezone
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
GRIB = ROOT / '4218a346c038961a43934d722972a5ad.grib'
OUT = ROOT / 'analysis_step14'
SEGMENTS = [1450, 1986]
GRID_DEG = 0.25
SIDE_MARGIN_DEG = 2.0  # candidate alternatives within +/- 2 degrees longitude
LAT_MARGIN_DEG = 0.5
EARTH_R = 6371000.0


def load_data():
    if not GRIB.exists():
        raise FileNotFoundError(f'Missing ERA5 GRIB: {GRIB}')
    ds = xr.open_dataset(GRIB, engine='cfgrib', backend_kwargs={'indexpath': ''})
    for var in ('u', 'v'):
        if var not in ds:
            raise ValueError(f'Missing ERA5 variable: {var}')
    return ds


def wind_cube(ds, pressure, lats, lons):
    """Interpolate pressure/lat/lon at each ERA5 hour; no extrapolation."""
    levels = np.asarray(ds.isobaricInhPa.values, float)
    lat0 = np.asarray(ds.latitude.values, float)
    lon0 = np.asarray(ds.longitude.values, float)
    ip, iy, ix = np.argsort(levels), np.argsort(lat0), np.argsort(lon0)
    if not (levels.min() <= pressure <= levels.max()):
        raise ValueError(f'Pressure {pressure} outside ERA5 [{levels.min()}, {levels.max()}]')
    yy, xx = np.meshgrid(lats, lons, indexing='ij')
    pts = np.column_stack([np.full(yy.size, pressure), yy.ravel(), xx.ravel()])
    cubes = []
    for var in ('u', 'v'):
        data = np.asarray(ds[var].values, float)[:, ip][:, :, iy][:, :, :, ix]
        out = np.empty((len(ds.time), len(lats), len(lons)))
        for k in range(len(ds.time)):
            f = RegularGridInterpolator((levels[ip], lat0[iy], lon0[ix]), data[k], bounds_error=False, fill_value=np.nan)
            out[k] = f(pts).reshape(yy.shape)
        cubes.append(out)
    times = np.asarray(ds.time.values, dtype='datetime64[ns]').astype('int64') / 1e9
    return times, cubes[0], cubes[1]


def geodesic_edge(a, b):
    la1, lo1 = np.radians(a)
    la2, lo2 = np.radians(b)
    dla, dlo = la2-la1, lo2-lo1
    h = np.sin(dla/2)**2 + np.cos(la1)*np.cos(la2)*np.sin(dlo/2)**2
    distance = EARTH_R*2*np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    east = np.sin(dlo)*np.cos(la2)
    north = np.cos(la1)*np.sin(la2)-np.sin(la1)*np.cos(la2)*np.cos(dlo)
    bearing = np.arctan2(east, north)
    return float(distance), float(np.sin(bearing)), float(np.cos(bearing))


def make_graph(lats, lons):
    ny, nx = len(lats), len(lons)
    edges = [[] for _ in range(ny*nx)]
    coords = np.array([(la, lo) for la in lats for lo in lons])
    for y in range(ny):
        for x in range(nx):
            a = y*nx+x
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == dx == 0: continue
                    yy, xx = y+dy, x+dx
                    if 0 <= yy < ny and 0 <= xx < nx:
                        b = yy*nx+xx
                        dist, east, north = geodesic_edge(coords[a], coords[b])
                        edges[a].append((b, dist, east, north))
    return coords, edges


def grid_index(lat, lon, lats, lons):
    iy = int(np.argmin(abs(lats-lat)))
    ix = int(np.argmin(abs(lons-lon)))
    return iy*len(lons)+ix


def edge_seconds(a, edge, departure, times, u, v, tas, nx):
    b, distance, east, north = edge
    # Approximate edge wind at midpoint of departure and predicted arrival.
    ya, xa = divmod(a, nx)
    yb, xb = divmod(b, nx)
    if departure < times[0] or departure > times[-1]: return np.inf
    def wind_at(t):
        if t < times[0] or t > times[-1]: return None
        j = int(np.clip(np.searchsorted(times, t, side='right')-1, 0, len(times)-2))
        alpha = (t-times[j])/(times[j+1]-times[j])
        uu = 0.5*(u[j,ya,xa]+u[j,yb,xb])*(1-alpha)+0.5*(u[j+1,ya,xa]+u[j+1,yb,xb])*alpha
        vv = 0.5*(v[j,ya,xa]+v[j,yb,xb])*(1-alpha)+0.5*(v[j+1,ya,xa]+v[j+1,yb,xb])*alpha
        return uu, vv
    def duration(t):
        w = wind_at(t)
        if w is None: return np.inf
        along = w[0]*east+w[1]*north
        cross = -w[0]*north+w[1]*east
        if abs(cross) >= tas: return np.inf
        speed = np.sqrt(max(0, tas*tas-cross*cross))+along
        return distance/speed if speed > 1 else np.inf
    d = duration(departure)
    if not np.isfinite(d): return np.inf
    d2 = duration(departure + 0.5*d)
    return d2 if np.isfinite(d2) else d


def shortest(edges, start, goal, depart, times, u, v, tas, nx, mode):
    n = len(edges)
    best = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)
    best[start] = 0.0
    heap = [(0.0, start)]
    expanded = 0
    while heap:
        cost, a = heapq.heappop(heap)
        if cost > best[a]+1e-7: continue
        expanded += 1
        if a == goal: break
        for e in edges[a]:
            b = e[0]
            weight = e[1] if mode == 'distance' else edge_seconds(a,e,depart+cost,times,u,v,tas,nx)
            if not np.isfinite(weight): continue
            new = cost+weight
            if new < best[b]-1e-7:
                best[b], parent[b] = new, a
                heapq.heappush(heap,(new,b))
    if not np.isfinite(best[goal]): return None, expanded
    route = [goal]
    while route[-1] != start:
        p = parent[route[-1]]
        if p < 0: return None, expanded
        route.append(int(p))
    return route[::-1], expanded


def evaluate(route, edges, depart, times, u, v, tas, nx):
    if route is None: return np.nan, np.nan
    total_time, total_dist = 0., 0.
    for a,b in zip(route[:-1],route[1:]):
        e = next(e for e in edges[a] if e[0]==b)
        seconds = edge_seconds(a,e,depart+total_time,times,u,v,tas,nx)
        if not np.isfinite(seconds): return np.nan, np.nan
        total_time += seconds
        total_dist += e[1]
    return total_dist/1000, total_time/60


def fifo_audit(edges, depart, times, u, v, tas, nx, samples=100):
    """Sample arrival-time monotonicity; a warning, not a proof of FIFO."""
    rng = np.random.default_rng(14)
    candidates = [(i,e) for i,es in enumerate(edges) for e in es]
    failures = 0; tested = 0
    if not candidates: return 0, 0
    for k in rng.choice(len(candidates),size=min(samples,len(candidates)),replace=False):
        a,e = candidates[k]
        ts = np.linspace(depart, min(times[-1]-1200,depart+4*3600),8)
        arr = np.array([t+edge_seconds(a,e,t,times,u,v,tas,nx) for t in ts])
        good = np.isfinite(arr)
        if good.sum() >= 2:
            tested += 1
            if np.any(np.diff(arr[good]) < -1e-6): failures += 1
    return failures,tested


def run_case(ds, segment):
    f = ROOT/'analysis_step9'/f'20140311_segment_{segment}_observations.csv'
    if not f.exists(): raise FileNotFoundError(f'Missing Step 9 file: {f}')
    d = pd.read_csv(f).sort_values('TIME_UTC')
    d = d.replace([np.inf,-np.inf],np.nan).dropna(subset=['TIME_UTC','G_LAT','G_LONG','P','TAS'])
    start, end = d.iloc[0], d.iloc[-1]
    # DATE + seconds, including any >86400; validate metadata before publication.
    depart = pd.Timestamp('2014-03-11',tz='UTC').timestamp()+float(start.TIME_UTC)
    tas = float(d.TAS.median()); pressure = float(d.P.median())
    minlat,maxlat = min(start.G_LAT,end.G_LAT),max(start.G_LAT,end.G_LAT)
    minlon,maxlon = min(start.G_LONG,end.G_LONG),max(start.G_LONG,end.G_LONG)
    latlo = max(float(ds.latitude.min()), minlat-LAT_MARGIN_DEG)
    lathi = min(float(ds.latitude.max()), maxlat+LAT_MARGIN_DEG)
    lonlo = max(float(ds.longitude.min()), minlon-SIDE_MARGIN_DEG)
    lonhi = min(float(ds.longitude.max()), maxlon+SIDE_MARGIN_DEG)
    lats = np.arange(np.floor(latlo/GRID_DEG)*GRID_DEG, lathi+GRID_DEG/2,GRID_DEG)
    lons = np.arange(np.floor(lonlo/GRID_DEG)*GRID_DEG, lonhi+GRID_DEG/2,GRID_DEG)
    lats = lats[(lats>=float(ds.latitude.min())) & (lats<=float(ds.latitude.max()))]
    lons = lons[(lons>=float(ds.longitude.min())) & (lons<=float(ds.longitude.max()))]
    times,u,v = wind_cube(ds,pressure,lats,lons)
    coords,edges = make_graph(lats,lons)
    s = grid_index(start.G_LAT,start.G_LONG,lats,lons)
    g = grid_index(end.G_LAT,end.G_LONG,lats,lons)
    print(f'\nSegment {segment}: {len(lats)}x{len(lons)} grid, fixed {pressure:.1f} hPa, TAS {tas:.1f} m/s')
    print('Original endpoints:',(start.G_LAT,start.G_LONG),(end.G_LAT,end.G_LONG))
    print('Snapped endpoints:',coords[s],coords[g])
    if s == g: raise ValueError('Grid too coarse: start equals goal')
    if not(times[0]<=depart<=times[-1]): raise ValueError('Departure outside ERA5 coverage')
    baseline,nb = shortest(edges,s,g,depart,times,u,v,tas,len(lons),'distance')
    aware,na = shortest(edges,s,g,depart,times,u,v,tas,len(lons),'time')
    if baseline is None or aware is None: raise RuntimeError('No feasible path; inspect bounds/ERA5 coverage')
    bd,bt = evaluate(baseline,edges,depart,times,u,v,tas,len(lons))
    ad,at = evaluate(aware,edges,depart,times,u,v,tas,len(lons))
    failures,tested = fifo_audit(edges,depart,times,u,v,tas,len(lons))
    saving = 100*(bt-at)/bt if np.isfinite(bt) and np.isfinite(at) and bt>0 else np.nan
    print(f'Distance route: {bd:.2f} km, {bt:.2f} min, {len(baseline)} waypoints, {nb} expanded')
    print(f'Wind-aware route: {ad:.2f} km, {at:.2f} min, {len(aware)} waypoints, {na} expanded')
    print(f'Modeled time saving: {saving:.3f}% | sampled FIFO violations: {failures}/{tested}')
    if failures: print('WARNING: FIFO violations detected; earliest-arrival Dijkstra may not be globally optimal.')
    if not np.isfinite(bt) or not np.isfinite(at): print('WARNING: route evaluation exceeded ERA5 coverage or became infeasible.')
    fig,ax=plt.subplots(figsize=(10,7))
    ax.plot(d.G_LONG,d.G_LAT,color='0.7',lw=1,label='Observed scientific track')
    for route,color,name in [(baseline,'black','Distance-minimizing'),(aware,'red','Time-dependent wind-aware')]:
        xy=coords[route]
        ax.plot(xy[:,1],xy[:,0],color=color,lw=2,label=name)
    ax.scatter([start.G_LONG,end.G_LONG],[start.G_LAT,end.G_LAT],s=35,c=['green','blue'],zorder=5)
    ax.set(xlabel='Longitude (°E)',ylabel='Latitude (°N)',title=f'Step 14 | segment {segment} | modeled saving {saving:.2f}%')
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(OUT/f'step14_segment_{segment}_routes.png',dpi=180); plt.close(fig)
    for route,name in [(baseline,'distance'),(aware,'wind_aware')]:
        pd.DataFrame({'latitude':coords[route,0],'longitude':coords[route,1]}).to_csv(OUT/f'step14_segment_{segment}_{name}_route.csv',index=False)
    return {'segment':segment,'grid_deg':GRID_DEG,'fixed_pressure_hpa':pressure,'fixed_tas_mps':tas,
            'baseline_distance_km':bd,'baseline_time_min':bt,'wind_aware_distance_km':ad,
            'wind_aware_time_min':at,'saving_pct':saving,'baseline_waypoints':len(baseline),
            'wind_aware_waypoints':len(aware),'baseline_expanded':nb,'wind_aware_expanded':na,
            'fifo_sample_failures':failures,'fifo_sample_tested':tested,
            'start_snap_error_km':geodesic_edge((start.G_LAT,start.G_LONG),coords[s])[0]/1000,
            'end_snap_error_km':geodesic_edge((end.G_LAT,end.G_LONG),coords[g])[0]/1000}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ds=load_data()
    results=[]
    for seg in SEGMENTS:
        results.append(run_case(ds,seg))
    p=OUT/'step14_time_dependent_summary.csv'
    pd.DataFrame(results).to_csv(p,index=False)
    print('\nSaved:',p)
    print('CAUTION: fixed pressure/TAS, grid-snapped endpoints, no aircraft turn dynamics or mission constraints.')
    print('CAUTION: time-dependent label-setting search requires FIFO; sample audit is not proof.')
    print('CAUTION: no causal or observed flight-time savings inferred.')

if __name__=='__main__': main()
