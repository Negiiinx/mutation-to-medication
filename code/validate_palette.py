"""Adjacent-pair CVD separation check in OKLab (x100), Vienot-style dichromat sim."""
import numpy as np, itertools, sys
def hex2lin(h):
    h=h.lstrip('#'); c=np.array([int(h[i:i+2],16)/255 for i in (0,2,4)])
    return np.where(c<=0.04045, c/12.92, ((c+0.055)/1.055)**2.4)
M_LMS=np.array([[0.31399022,0.63951294,0.04649755],
                [0.15537241,0.75789446,0.08670142],
                [0.01775239,0.10944209,0.87256922]])
M_LMS_INV=np.linalg.inv(M_LMS)
SIM={'protan':np.array([[0,1.05118294,-0.05116099],[0,1,0],[0,0,1]]),
     'deutan':np.array([[1,0,0],[0.9513092,0,0.04866992],[0,0,1]]),
     'tritan':np.array([[1,0,0],[0,1,0],[-0.86744736,1.86727089,0]])}
def simulate(lin,kind):
    return M_LMS_INV@(SIM[kind]@(M_LMS@lin))
def lin2oklab(c):
    c=np.clip(c,0,1)
    M1=np.array([[0.4122214708,0.5363325363,0.0514459929],
                 [0.2119034982,0.6806995451,0.1073969566],
                 [0.0883024619,0.2817188376,0.6299787005]])
    M2=np.array([[0.2104542553,0.7936177850,-0.0040720468],
                 [1.9779984951,-2.4285922050,0.4505937099],
                 [0.0259040371,0.7827717662,-0.8086757660]])
    return M2@np.cbrt(M1@c)
def dE(a,b): return float(np.linalg.norm(lin2oklab(a)-lin2oklab(b))*100)

pal=sys.argv[1].split(',')
lins=[hex2lin(h) for h in pal]
print(f"palette: {pal}\n")
print(f"{'pair':>9s} {'normal':>8s} {'protan':>8s} {'deutan':>8s} {'tritan':>8s}  verdict")
worst_norm=1e9; worst_cvd=1e9; fails=[]
for i,j in itertools.combinations(range(len(pal)),2):
    n=dE(lins[i],lins[j])
    p=dE(simulate(lins[i],'protan'),simulate(lins[j],'protan'))
    d=dE(simulate(lins[i],'deutan'),simulate(lins[j],'deutan'))
    t=dE(simulate(lins[i],'tritan'),simulate(lins[j],'tritan'))
    mn=min(p,d,t)
    v='PASS' if (n>=15 and mn>=8) else ('FLOOR' if (n>=15 and mn>=6) else 'FAIL')
    if v!='PASS': fails.append((pal[i],pal[j],n,mn,v))
    worst_norm=min(worst_norm,n); worst_cvd=min(worst_cvd,mn)
    print(f"{i+1}-{j+1:<7d} {n:8.1f} {p:8.1f} {d:8.1f} {t:8.1f}  {v}")
print(f"\nworst normal-vision dE: {worst_norm:.1f} (hard floor 15)")
print(f"worst CVD dE:           {worst_cvd:.1f} (target 8, floor 6)")
print("RESULT:", "PASS" if not fails else f"{len(fails)} problem pair(s): "+
      ", ".join(f"{a}/{b}({v})" for a,b,_,_,v in fails))
