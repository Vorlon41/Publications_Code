
from pathlib import Path
import heapq
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
INPUT = ROOT / 'analysis_step2'
OUTPUT = ROOT / 'analysis_step7'
OUTPUT.mkdir(parents=True, exist_ok=True)
CASES = [
    ('low_wind', 'ATTREX-MMS-1HZ_GHawk_20150305_R1_clean.csv', 1925),
    ('strong_wind', 'ATTREX-MMS-1HZ_GHawk_20140311_R2_clean.csv', 1986),
]
N = 91  # 91x91 planning cells; no obstacles in this verification experiment
EARTH_R = 6371008.8
NEIGHBORS = [(di,dj) for di in (-1,0,1) for dj in (-1,0,1) if di or dj]

def haversine(lat1,lon1,lat2,lon2):
    a1,a2=np.radians([lat1,lat2]); dl=np.radians(lon2-lon1)
    aa=np.sin((a2-a1)/2)**2+np.cos(a1)*np.cos(a2)*np.sin(dl/2)**2
    return 2*EARTH_R*np.arcsin(np.sqrt(np.clip(aa,0,1)))

def extract_segment(df, seg_id):
    """Locate a Step-3 segment using contiguous valid-cruise blocks.

    Segment IDs depend on the earlier Step-3 implementation; we avoid guessing
    them. If an exact segment file is not provided, prompt user to export one.
    """
    # If Step 3 generated a segment export with matching identifier, use it.
    raise NotImplementedError

def run_case(label, filename, segment_id):
    path=INPUT/filename
    if not path.exists():
        print(f'SKIP {label}: missing {path}')
        return None
    df=pd.read_csv(path, usecols=['TIME_UTC','G_LAT','G_LONG','G_ALT','TAS','U','V'])
    # Segment IDs in Step 4 refer to Step 3 run numbering, whose original
    # segmentation rule is unavailable here. Instead select a deterministic,
    # transparent continuous cruise window as an algorithm smoke test.
    valid=np.isfinite(df[['TIME_UTC','G_LAT','G_LONG','G_ALT','TAS','U','V']]).all(axis=1)
    valid &= (df.G_ALT>=14000)&df.TAS.between(80,300)
    dt=df.TIME_UTC.diff()
    block=(~valid | ~valid.shift(fill_value=False) | dt.ne(1)).cumsum()
    runs=[]
    for _,g in df.loc[valid].groupby(block[valid]):
        if len(g)>=1800:
            dist=haversine(g.G_LAT.iloc[0],g.G_LONG.iloc[0],g.G_LAT.iloc[-1],g.G_LONG.iloc[-1])
            if dist>=200000:
                runs.append((dist,g))
    if not runs:
        print(f'SKIP {label}: no >=30 min continuous cruise window with >=200 km endpoint separation')
        return None
    # Select first qualifying run and first 35 min window where possible.
    g=runs[0][1].iloc[:min(2100,len(runs[0][1]))].copy()
    # If initial 35 min ends close to start, use full qualifying run.
    if haversine(g.G_LAT.iloc[0],g.G_LONG.iloc[0],g.G_LAT.iloc[-1],g.G_LONG.iloc[-1])<100000:
        g=runs[0][1]
    # Longitude unwrapping makes dateline-crossing trajectories safe.
    lons=np.degrees(np.unwrap(np.radians(g.G_LONG.to_numpy())))
    lat0=float(np.mean(g.G_LAT)); lon0=float(np.mean(lons))
    x=EARTH_R*np.cos(np.radians(lat0))*np.radians(lons-lon0)
    y=EARTH_R*np.radians(g.G_LAT.to_numpy()-lat0)
    sx,sy=float(x[0]),float(y[0]); ex,ey=float(x[-1]),float(y[-1])
    straight=np.hypot(ex-sx,ey-sy)
    if straight<100000:
        print(f'SKIP {label}: endpoint separation too short')
        return None
    # Place start and goal on central row, creating room for detours.
    ux,uy=(ex-sx)/straight,(ey-sy)/straight
    vx,vy=-uy,ux
    # coordinate transform: s along track, q perpendicular
    s=(x-sx)*ux+(y-sy)*uy
    q=(x-sx)*vx+(y-sy)*vy
    length=straight
    margin=0.35*length
    xs=np.linspace(-0.1*length,1.1*length,N)
    ys=np.linspace(-margin,margin,N)
    dx,dy=xs[1]-xs[0],ys[1]-ys[0]
    start=(int(np.argmin(abs(ys))),int(np.argmin(abs(xs-0))))
    goal=(int(np.argmin(abs(ys))),int(np.argmin(abs(xs-length))))
    va=float(np.nanmedian(g.TAS))
    median_wind=float(np.nanmedian(np.hypot(g.U,g.V)))
    # Synthetic spatially varying wind field in local (s,q) coordinates.
    # Wind scale is set from observed median solely to create distinct test
    # regimes; spatial patterns are NOT reconstructed from observations.
    S,Q=np.meshgrid(xs,ys)
    scale=min(median_wind,0.45*va)
    ws=scale*(0.35+0.85*np.sin(2*np.pi*S/length)*np.cos(np.pi*Q/margin))
    wq=scale*(0.20+0.75*np.cos(2*np.pi*S/length)*np.sin(np.pi*Q/margin))
    wind_bound=float(np.max(np.hypot(ws,wq)))
    def edge_cost(a,b,aware):
        di=b[0]-a[0]; dj=b[1]-a[1]
        dd=np.hypot(dj*dx,di*dy)
        if not aware:
            return dd/va
        ts,tq=(dj*dx/dd,di*dy/dd)
        wm_s=0.5*(ws[a]+ws[b]); wm_q=0.5*(wq[a]+wq[b])
        along=wm_s*ts+wm_q*tq
        cross=-wm_s*tq+wm_q*ts
        if abs(cross)>=va: return math.inf
        vg=np.sqrt(va*va-cross*cross)+along
        return dd/vg if vg>0 else math.inf
    def astar(aware):
        # Admissible heuristic: max ground speed <= Va + global wind bound.
        def h(node):
            dd=np.hypot((goal[1]-node[1])*dx,(goal[0]-node[0])*dy)
            return dd/(va+wind_bound) if aware else dd/va
        queue=[(h(start),0,start)]
        best={start:0.0}; parent={}; counter=0
        while queue:
            _,_,a=heapq.heappop(queue)
            if a==goal:
                route=[a]
                while route[-1] in parent: route.append(parent[route[-1]])
                return route[::-1]
            for di,dj in NEIGHBORS:
                b=(a[0]+di,a[1]+dj)
                if not(0<=b[0]<N and 0<=b[1]<N):continue
                cost=edge_cost(a,b,aware)
                new=best[a]+cost
                if new+1e-9<best.get(b,math.inf):
                    best[b]=new;parent[b]=a;counter+=1
                    heapq.heappush(queue,(new+h(b),counter,b))
        raise RuntimeError('No feasible path')
    baseline=astar(False); aware=astar(True)
    def metrics(route):
        distance=sum(np.hypot((b[1]-a[1])*dx,(b[0]-a[0])*dy) for a,b in zip(route,route[1:]))
        time=sum(edge_cost(a,b,True) for a,b in zip(route,route[1:]))
        return distance/1000,time/3600
    d0,t0=metrics(baseline); d1,t1=metrics(aware)
    fig,ax=plt.subplots(figsize=(10,6))
    im=ax.imshow(np.hypot(ws,wq),origin='lower',extent=[xs[0]/1000,xs[-1]/1000,ys[0]/1000,ys[-1]/1000],aspect='auto',cmap='viridis')
    for route,name,color in [(baseline,'Wind-unaware A*','white'),(aware,'Wind-aware A*','red')]:
        ax.plot([xs[j]/1000 for i,j in route],[ys[i]/1000 for i,j in route],color=color,lw=2,label=name)
    ax.scatter([xs[start[1]]/1000,xs[goal[1]]/1000],[ys[start[0]]/1000,ys[goal[0]]/1000],c=['lime','magenta'],s=50,zorder=5)
    ax.set(xlabel='Along-track distance (km)',ylabel='Cross-track distance (km)',title=f'{label}: algorithm verification with SYNTHETIC spatial wind')
    ax.legend(loc='upper right');fig.colorbar(im,ax=ax,label='Synthetic wind speed (m/s)')
    fig.tight_layout();fig.savefig(OUTPUT/f'{label}_synthetic_astar.png',dpi=180);plt.close(fig)
    return dict(case=label,source=filename,requested_step4_segment_id=segment_id,
        note='Automatically selected cruise window, NOT necessarily requested Step 4 segment',
        observations=len(g),median_observed_wind_mps=median_wind,assumed_airspeed_mps=va,
        endpoint_distance_km=straight/1000,baseline_distance_km=d0,wind_aware_distance_km=d1,
        baseline_time_under_same_wind_h=t0,wind_aware_time_under_same_wind_h=t1,
        simulated_time_saving_pct=100*(t0-t1)/t0)

def main():
    results=[]
    for label,filename,segid in CASES:
        result=run_case(label,filename,segid)
        if result:
            results.append(result)
            print(label, 'baseline (h):',round(result['baseline_time_under_same_wind_h'],3),
                  'wind-aware (h):',round(result['wind_aware_time_under_same_wind_h'],3),
                  'saving (%):',round(result['simulated_time_saving_pct'],2))
    if results:
        out=OUTPUT/'synthetic_astar_verification.csv'
        pd.DataFrame(results).to_csv(out,index=False)
        print('Saved:',out)
    else:
        print('No suitable continuous cruise windows found; inspect Step 3 segment exports.')
if __name__=='__main__':main()
