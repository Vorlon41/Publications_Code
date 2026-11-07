
from heapq import heappush, heappop
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

N = 101
DX = 1000.0  # meters per grid edge
VA = 50.0    # m/s; deliberately low for a transparent unit test, NOT Global Hawk performance
start, goal = (50, 5), (50, 95)
# Eastward tailwind corridor 15 km north of the direct route; zero wind elsewhere.
u = np.zeros((N,N), dtype=float)
v = np.zeros_like(u)
u[30:38, :] = 35.0
# Eight-connected graph with exact wind-triangle edge cost and midpoint sampling.
MOVES = [(dr,dc) for dr in (-1,0,1) for dc in (-1,0,1) if (dr,dc)!=(0,0)]

def edge(a,b, wind):
    dr, dc = b[0]-a[0], b[1]-a[1]
    length = DX*np.hypot(dr,dc)
    if not wind:
        return length/VA
    r,c = (a[0]+b[0])//2, (a[1]+b[1])//2
    ex, ey = dc/np.hypot(dr,dc), -dr/np.hypot(dr,dc)
    wx,wy = u[r,c],v[r,c]
    parallel = wx*ex+wy*ey
    cross = -wx*ey+wy*ex
    if abs(cross)>=VA:
        return np.inf
    speed = np.sqrt(VA**2-cross**2)+parallel
    return length/speed if speed>0 else np.inf

def astar(wind):
    # Zero heuristic is admissible and makes this a Dijkstra/A* correctness test.
    q=[(0.0,start)]; best={start:0.0}; parent={}; done=set()
    while q:
        g,a=heappop(q)
        if a in done: continue
        done.add(a)
        if a==goal:
            path=[a]
            while a in parent:
                a=parent[a];path.append(a)
            return path[::-1]
        for dr,dc in MOVES:
            b=(a[0]+dr,a[1]+dc)
            if not (0<=b[0]<N and 0<=b[1]<N): continue
            cost=edge(a,b,wind)
            new=g+cost
            if np.isfinite(cost) and new<best.get(b,np.inf):
                best[b]=new;parent[b]=a;heappush(q,(new,b))
    raise RuntimeError('No feasible path')

def metrics(path):
    distance=sum(DX*np.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(path[:-1],path[1:]))
    time=sum(edge(a,b,True) for a,b in zip(path[:-1],path[1:]))
    return distance/1000,time/60

baseline=astar(False)
aware=astar(True)
base_km,base_min=metrics(baseline)
aware_km,aware_min=metrics(aware)
print(f'Baseline: {base_km:.2f} km, {base_min:.2f} min')
print(f'Wind-aware: {aware_km:.2f} km, {aware_min:.2f} min')
print(f'Simulated saving: {100*(base_min-aware_min)/base_min:.2f}%')
assert aware_min <= base_min+1e-8, 'Wind-aware optimal route must not be worse under identical wind costs'
assert aware_min < base_min-1e-5, 'Synthetic tailwind corridor should create a detectable benefit'
print('PASS: Wind-aware route has lower travel time in controlled synthetic test')
fig,ax=plt.subplots(figsize=(10,5))
ax.imshow(np.hypot(u,v),origin='upper',extent=[0,N,N,0],cmap='viridis',vmin=0,vmax=35)
for path,label,color in [(baseline,'Wind-unaware A*','white'),(aware,'Wind-aware A*','red')]:
    p=np.array(path)
    ax.plot(p[:,1]+.5,p[:,0]+.5,color=color,lw=2,label=label)
ax.set(xlabel='Grid column (1 km)',ylabel='Grid row (1 km)',title='Controlled synthetic tailwind-corridor unit test')
ax.legend(loc='upper right');fig.tight_layout()
out=Path(__file__).resolve().parent/'attrex_step8_synthetic_unit_test.png'
fig.savefig(out,dpi=180)
print('Saved:',out)
plt.show()
