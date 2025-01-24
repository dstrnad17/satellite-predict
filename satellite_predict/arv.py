import numpy as np
import pandas as pd

def arv(A, P):
    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
    if isinstance(P, pd.DataFrame):
        P = P.to_numpy()

    if A.ndim == 1:
        A, P = np.expand_dims(A, axis=1), np.expand_dims(P, axis=1)

    arvs = []
    for i in range(A.shape[1]):
        var_A = np.var(A[:, i])
        arv = np.var(A[:, i] - P[:, i]) / var_A if var_A > 0 else 0
        arvs.append(arv)

    return arvs if len(arvs) > 1 else arvs[0]
