import numpy as np, pandas as pd
def ols_cluster(y, X, clusters, names):
    """OLS with cluster-robust (CR1) standard errors."""
    X=np.asarray(X,float); y=np.asarray(y,float)
    XtX_inv=np.linalg.pinv(X.T@X); b=XtX_inv@X.T@y; e=y-X@b
    n,k=X.shape; cl=pd.Series(clusters); g=cl.nunique()
    meat=np.zeros((k,k))
    for _,idx in cl.groupby(cl).groups.items():
        ii=cl.index.get_indexer(idx); Xg=X[ii]; eg=e[ii]
        s=Xg.T@eg; meat+=np.outer(s,s)
    c=(g/(g-1))*((n-1)/(n-k))
    V=c*XtX_inv@meat@XtX_inv
    se=np.sqrt(np.diag(V))
    t=b/se
    from math import erf, sqrt
    p=[2*(1-0.5*(1+erf(abs(ti)/sqrt(2)))) for ti in t]
    return pd.DataFrame({'coef':b,'se':se,'t':t,'p':p}, index=names), g
