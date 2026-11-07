
from pathlib import Path
import importlib.util
import heapq
import time
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

PROJECT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
CODE_DIR = PROJECT.parent / 'Code'
STEP18 = CODE_DIR / 'attrex_step18_exact_endpoint_astar.py'
GRIB = PROJECT / 'ERA5_ATTREX_20140311_20140312_expanded.grib'
OBS = PROJECT / 'analysis_step23' / '20140311_segment_2798_observations.csv'
OUT = PROJECT / 'analysis_step26'
WIDTHS_KM = (0, 50, 100)
LATERAL_STEP_KM = 25
STATION_STEP_KM = 25
SEARCH_INTEGRATION_KM = 5.0
VALIDATION_INTEGRATION_KM = (2.0, 1.0)
EARTH = 6371008.8


def import_base():
    if not STEP18.is_file():
        raise FileNotFoundError(f'Missing Step 18 dependency: {STEP18}')
    spec = importlib.util.spec_from_file_location('attrex_step18_step26', STEP18)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spherical_interpolate(a, b, fraction):
    """Spherical great-circle interpolation, coordinates [lat, lon] degrees."""
    a, b = np.deg2rad(a), np.deg2rad(b)
    def vec(p):
        return np.array([np.cos(p[0])*np.cos(p[1]), np.cos(p[0])*np.sin(p[1]), np.sin(p[0])])
    u, v = vec(a), vec(b)
    angle = np.arccos(np.clip(u @ v, -1, 1))
    if angle < 1e-12:
        w = u
    else:
        w = (np.sin((1-fraction)*angle)*u + np.sin(fraction*angle)*v)/np.sin(angle)
    return np.array([np.rad2deg(np.arctan2(w[2], np.hypot(w[0], w[1]))),
                     np.rad2deg(np.arctan2(w[1], w[0]))])


def offset_point(point, heading, offset_m):
    """Cross-track displacement; positive = left of travel heading."""
    lat, lon = np.deg2rad(point)
    bearing = np.arctan2(heading[0], heading[1]) - np.pi/2
    delta = offset_m / EARTH
    lat2 = np.arcsin(np.sin(lat)*np.cos(delta)+np.cos(lat)*np.sin(delta)*np.cos(bearing))
    lon2 = lon + np.arctan2(np.sin(bearing)*np.sin(delta)*np.cos(lat),
                            np.cos(delta)-np.sin(lat)*np.sin(lat2))
    return np.rad2deg([lat2, lon2])


def build_corridor(base, a, b, width_km):
    length = base.edge_geometry(a, b)[0]
    nlegs = max(2, int(np.ceil(length/(STATION_STEP_KM*1000))))
    stations = np.linspace(0, 1, nlegs+1)
    offsets = np.arange(-width_km, width_km+0.01, LATERAL_STEP_KM) if width_km else np.array([0])
    coords, layers = [], []
    for k, f in enumerate(stations):
        if k == 0:
            layers.append([len(coords)]); coords.append(a)
        elif k == len(stations)-1:
            layers.append([len(coords)]); coords.append(b)
        else:
            center = spherical_interpolate(a, b, f)
            ahead = spherical_interpolate(a, b, min(1, f+0.0001))
            heading = base.edge_geometry(center, ahead)[1]
            indices=[]
            for offset in offsets:
                indices.append(len(coords))
                coords.append(offset_point(center, heading, float(offset)*1000))
            layers.append(indices)
    coords=np.asarray(coords)
    adj=[[] for _ in coords]
    # Forward edges between adjacent stations and one-station skips. Both
    # objectives use EXACTLY the same feasible directed graph.
    for i in range(len(layers)-1):
        for jump in (1,2):
            if i+jump >= len(layers): continue
            for u in layers[i]:
                for v in layers[i+jump]:
                    # Limit lateral maneuver per longitudinal advance; otherwise
                    # graph can jump between distant offsets without turn limits.
                    if base.edge_geometry(coords[u],coords[v])[0] <= (jump*STATION_STEP_KM+55)*1000:
                        adj[u].append(v)
    # Direct geodesic is a valid comparator for both algorithms.
    adj[0].append(len(coords)-1)
    return coords, adj, 0, len(coords)-1


def solve(base, coords, adj, start, goal, depart, wind, tas, mode):
    best=np.full(len(coords), np.inf); best[start]=0
    prev=np.full(len(coords), -1, dtype=int)
    heap=[(0.0,start)]; expansions=0
    while heap:
        cost,i=heapq.heappop(heap)
        if cost > best[i]+1e-8: continue
        expansions+=1
        if i == goal: break
        for j in adj[i]:
            weight = (base.edge_geometry(coords[i],coords[j])[0] if mode=='distance'
                      else base.travel(coords[i],coords[j],depart+cost,wind,tas))
            if not np.isfinite(weight): continue
            trial=cost+weight
            if trial < best[j]-1e-8:
                best[j]=trial;prev[j]=i;heapq.heappush(heap,(trial,j))
    if not np.isfinite(best[goal]): raise RuntimeError(f'No route for {mode}')
    route=[goal]
    while route[-1] != start:
        route.append(int(prev[route[-1]]))
        if route[-1]<0: raise RuntimeError('Broken predecessor chain')
    return route[::-1],expansions


def eval_route(base, coords, route, depart, wind, tas, spacing_km):
    base.MAX_LEG_KM=spacing_km
    elapsed=0.;dist=0.
    for i,j in zip(route[:-1],route[1:]):
        dist+=base.edge_geometry(coords[i],coords[j])[0]
        dt=base.travel(coords[i],coords[j],depart+elapsed,wind,tas)
        if not np.isfinite(dt): return np.inf,dist/1000
        elapsed+=dt
    return elapsed,dist/1000


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    base=import_base()
    if not GRIB.exists(): raise FileNotFoundError(GRIB)
    if not OBS.exists(): raise FileNotFoundError(OBS)
    obs=pd.read_csv(OBS).sort_values('TIME_UTC').dropna(subset=['TIME_UTC','G_LAT','G_LONG','P','TAS'])
    a=obs[['G_LAT','G_LONG']].iloc[0].to_numpy(float)
    b=obs[['G_LAT','G_LONG']].iloc[-1].to_numpy(float)
    departure=pd.Timestamp('2014-03-11',tz='UTC').timestamp()+float(obs.TIME_UTC.iloc[0])
    pressure=float(obs.P.median()); tas=float(obs.TAS.median())
    print('Segment 2798: departure UTC',pd.to_datetime(departure,unit='s',utc=True))
    print('Median pressure hPa:',pressure,'median TAS m/s:',tas)
    ds=xr.open_dataset(GRIB,engine='cfgrib',backend_kwargs={'indexpath':''})
    wind=base.Wind(ds,pressure)
    print('ERA5 bounds UTC:',pd.to_datetime(wind.times[[0,-1]],unit='s',utc=True))
    if not(wind.times[0]<=departure<wind.times[-1]):
        raise ValueError('Departure outside ERA5 time support')
    rows=[]; details=[]
    for width in WIDTHS_KM:
        t0=time.perf_counter()
        coords,adj,start,goal=build_corridor(base,a,b,width)
        if (np.any(coords[:,0]<wind.lat.min()) or np.any(coords[:,0]>wind.lat.max())
            or np.any(coords[:,1]<wind.lon.min()) or np.any(coords[:,1]>wind.lon.max())):
            raise ValueError(f'Corridor width {width} km exceeds ERA5 geographic coverage')
        result={}
        base.MAX_LEG_KM=SEARCH_INTEGRATION_KM
        for name,mode in [('distance','distance'),('wind_aware','time')]:
            route,expansions=solve(base,coords,adj,start,goal,departure,wind,tas,mode)
            xy=coords[route]
            pd.DataFrame(xy,columns=['latitude','longitude']).to_csv(
                OUT/f'step26_width_{width}km_{name}_route.csv',index=False)
            max_turn,n_sharp=base.turns(xy)
            result[name]=(route,xy)
            for spacing in VALIDATION_INTEGRATION_KM:
                seconds,km=eval_route(base,coords,route,departure,wind,tas,spacing)
                details.append(dict(width_km=width,route=name,validation_spacing_km=spacing,
                                    time_seconds=seconds,distance_km=km,waypoints=len(route),
                                    search_expansions=expansions,max_turn_deg=max_turn,
                                    turns_over_30_deg=n_sharp))
        for spacing in VALIDATION_INTEGRATION_KM:
            subset=[r for r in details if r['width_km']==width and r['validation_spacing_km']==spacing]
            d=next(r for r in subset if r['route']=='distance')
            w=next(r for r in subset if r['route']=='wind_aware')
            rows.append(dict(width_km=width,validation_spacing_km=spacing,
                             distance_time_s=d['time_seconds'],wind_aware_time_s=w['time_seconds'],
                             advantage_s=d['time_seconds']-w['time_seconds'],
                             distance_km=d['distance_km'],wind_aware_km=w['distance_km'],
                             same_waypoint_sequence=np.array_equal(result['distance'][1],result['wind_aware'][1])))
        fig,ax=plt.subplots(figsize=(9,6))
        ax.plot(obs.G_LONG,obs.G_LAT,c='0.7',lw=1.1,label='Observed mission track')
        for name,color in [('distance','black'),('wind_aware','red')]:
            xy=result[name][1]
            ax.plot(xy[:,1],xy[:,0],lw=2,c=color,label=name)
        ax.scatter([a[1],b[1]],[a[0],b[0]],c=['green','blue'],s=50,zorder=5)
        ax.set(xlabel='Longitude (degrees E)',ylabel='Latitude (degrees N)',
               title=f'Step 26 | segment 2798 | corridor +/-{width} km')
        ax.grid(alpha=.25);ax.legend();fig.tight_layout()
        fig.savefig(OUT/f'step26_width_{width}km_routes.png',dpi=180);plt.close(fig)
        print(f'Width +/-{width} km: {len(coords)} nodes, {sum(map(len,adj))} edges; '
              f'1-km advantage {rows[-1]["advantage_s"]:.6f} s; '
              f'elapsed {time.perf_counter()-t0:.1f} s',flush=True)
    pd.DataFrame(details).to_csv(OUT/'step26_route_details.csv',index=False)
    paired=pd.DataFrame(rows)
    paired.to_csv(OUT/'step26_paired_comparison.csv',index=False)
    print('\n',paired.to_string(index=False))
    print('\nSaved:',OUT)
    print('CAUTION: direct edge included in BOTH graph objectives; no mission or aircraft dynamics constraints.')
    print('CAUTION: time-dependent FIFO is not certified; 1-km evaluation is not independent wind validation.')
    print('CAUTION: ERA5 only validated along observed trajectory; no operational savings inferred.')

if __name__=='__main__':
    main()
