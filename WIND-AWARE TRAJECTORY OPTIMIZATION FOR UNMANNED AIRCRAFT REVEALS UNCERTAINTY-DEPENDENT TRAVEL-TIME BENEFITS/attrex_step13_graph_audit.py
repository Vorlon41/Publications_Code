
from pathlib import Path
import heapq
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
GRIB = ROOT/'4218a346c038961a43934d722972a5ad.grib'
OBS = ROOT/'analysis_step9'/'20140311_segment_1986_observations.csv'
OUT = ROOT/'analysis_step13'
R = 6371008.8

def meters(a,b):
    lat1,lon1=a;lat2,lon2=b
    n=np.radians(lat2-lat1)*R
    e=np.radians(lon2-lon1)*R*np.cos(np.radians((lat1+lat2)/2))
    return e,n

def experiment(step,scenario,start,end,tas,pressure,dep,uinterp,vinterp):
    margin=.7
    latmin=max(11.,min(start[0],end[0])-margin)
    latmax=min(35.,max(start[0],end[0])+margin)
    lonmin=max(142.,min(start[1],end[1])-margin)
    lonmax=min(155.,max(start[1],end[1])+margin)
    lats=np.arange(np.ceil(latmin/step)*step,np.floor(latmax/step)*step+step/2,step)
    lons=np.arange(np.ceil(lonmin/step)*step,np.floor(lonmax/step)*step+step/2,step)
    nr,nc=len(lats),len(lons)
    nodes=[(float(a),float(b)) for a in lats for b in lons]+[start,end]
    N=nr*nc; S=N; E=N+1
    def idx(i,j):return i*nc+j
    def near(p):return int(np.argmin(abs(lats-p[0]))),int(np.argmin(abs(lons-p[1])))
    # NO start-to-end shortcut; special endpoints connect only to local grid nodes.
    def adjacent(k):
        if k>=N:
            i,j=near(nodes[k]); coords=[(i+di,j+dj) for di in (-1,0,1) for dj in (-1,0,1)]
            return [idx(ii,jj) for ii,jj in coords if 0<=ii<nr and 0<=jj<nc]
        i,j=divmod(k,nc)
        z=[idx(i+di,j+dj) for di in (-1,0,1) for dj in (-1,0,1)
           if (di or dj) and 0<=i+di<nr and 0<=j+dj<nc]
        for k2 in (S,E):
            si,sj=near(nodes[k2])
            if abs(si-i)<=1 and abs(sj-j)<=1:z.append(k2)
        return z
    def wind_at(mid):
        if scenario=='zero':return 0.,0.
        if scenario=='synthetic_corridor':
            # Controlled diagnostic only: narrow eastward tailwind band north of direct route.
            return (95.0 if start[0]+.22 <= mid[0] <= start[0]+.40 else 0.),0.
        point=np.array([[dep,pressure,mid[0],mid[1]]])
        u=float(uinterp(point)[0]);v=float(vinterp(point)[0])
        if scenario=='reversed':return -u,-v
        return u,v
    def edge(a,b,with_wind):
        e,n=meters(nodes[a],nodes[b]);d=np.hypot(e,n)
        if d<.001:return 0.,0.
        if not with_wind:return d,d/tas
        mid=((nodes[a][0]+nodes[b][0])/2,(nodes[a][1]+nodes[b][1])/2)
        u,v=wind_at(mid)
        if not (np.isfinite(u) and np.isfinite(v)):return d,np.inf
        along=(u*e+v*n)/d;cross=(-u*n+v*e)/d
        if abs(cross)>=tas:return d,np.inf
        gs=np.sqrt(tas*tas-cross*cross)+along
        return (d,d/gs) if gs>0 else (d,np.inf)
    # Dijkstra is used to avoid heuristic admissibility problems.
    def solve(use_wind):
        best={S:0.};parent={};q=[(0.,S)];expanded=0
        while q:
            cost,a=heapq.heappop(q)
            if cost>best[a]+1e-8:continue
            expanded+=1
            if a==E:
                p=[E]
                while p[-1]!=S:p.append(parent[p[-1]])
                return p[::-1],expanded
            for b in adjacent(a):
                d,t=edge(a,b,use_wind);w=t if use_wind else d
                if np.isfinite(w) and cost+w<best.get(b,np.inf)-1e-8:
                    best[b]=cost+w;parent[b]=a;heapq.heappush(q,(best[b],b))
        raise RuntimeError('No route')
    def evaluate(path):
        d=t=0.
        for a,b in zip(path[:-1],path[1:]):
            dd,tt=edge(a,b,True);d+=dd;t+=tt
        return d/1000,t/60
    baseline,exp0=solve(False)
    optimized,exp1=solve(True)
    d0,t0=evaluate(baseline);d1,t1=evaluate(optimized)
    result=dict(grid_deg=step,scenario=scenario,grid_nodes=N,shortcut_edge_enabled=False,
                baseline_waypoints=len(baseline),optimized_waypoints=len(optimized),
                baseline_expanded=exp0,optimized_expanded=exp1,
                baseline_km=d0,optimized_km=d1,baseline_min=t0,optimized_min=t1,
                saving_pct=100*(t0-t1)/t0)
    fig,ax=plt.subplots(figsize=(9,4.5))
    for p,col,lab in [(baseline,'black','Distance shortest'),(optimized,'red','Wind-aware shortest time')]:
        xy=np.array([nodes[k] for k in p]);ax.plot(xy[:,1],xy[:,0],c=col,lw=1.6,label=lab)
    ax.set(xlabel='Longitude (°E)',ylabel='Latitude (°N)',title=f'{scenario} | grid {step}° | saving {result["saving_pct"]:.2f}%')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    fig.savefig(OUT/f'route_{scenario}_{str(step).replace(".","p")}.png',dpi=160)
    plt.close(fig)
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    obs=pd.read_csv(OBS).sort_values('TIME_UTC').dropna(subset=['G_LAT','G_LONG','P','TAS','TIME_UTC'])
    start=(float(obs.G_LAT.iloc[0]),float(obs.G_LONG.iloc[0]))
    end=(float(obs.G_LAT.iloc[-1]),float(obs.G_LONG.iloc[-1]))
    tas=float(obs.TAS.median());pressure=float(obs.P.median())
    depart=(pd.Timestamp('2014-03-11',tz='UTC')+pd.to_timedelta(float(obs.TIME_UTC.iloc[0]),unit='s')).timestamp()
    with xr.open_dataset(GRIB,engine='cfgrib',backend_kwargs={'indexpath':''}) as original:
        ds=original[['u','v']].sortby('isobaricInhPa').sortby('latitude').load()
    time=pd.DatetimeIndex(ds.time.values).tz_localize('UTC').asi8/1e9
    axes=(time,ds.isobaricInhPa.values,ds.latitude.values,ds.longitude.values)
    uinterp=RegularGridInterpolator(axes,ds.u.values,bounds_error=False,fill_value=np.nan)
    vinterp=RegularGridInterpolator(axes,ds.v.values,bounds_error=False,fill_value=np.nan)
    results=[]
    for step,scenario in [(0.20,'era5'),(0.10,'era5'),(0.20,'reversed'),(0.10,'reversed'),
                          (0.20,'zero'),(0.10,'zero'),(0.10,'synthetic_corridor')]:
        print('Running',scenario,'grid',step,flush=True)
        r=experiment(step,scenario,start,end,tas,pressure,depart,uinterp,vinterp)
        results.append(r)
        print(f'  baseline {r["baseline_min"]:.3f} min; optimized {r["optimized_min"]:.3f} min; saving {r["saving_pct"]:.3f}%; waypoints {r["baseline_waypoints"]}/{r["optimized_waypoints"]}',flush=True)
    summary=pd.DataFrame(results)
    summary.to_csv(OUT/'step13_graph_audit_summary.csv',index=False)
    print('\nSaved:',OUT)
    print('CAUTION: departure-time FROZEN ERA5, 8-connected grid, no direct shortcut; not dynamic A*.')
    print('CAUTION: synthetic corridor and reversed wind are diagnostics, not historical conditions.')

if __name__=='__main__':main()
