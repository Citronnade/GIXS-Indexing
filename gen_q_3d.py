import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd

def read_qs(filename):
    df = pd.read_csv(filename)
    return np.array(df['qxy'], df['qz'])

# in plane, already done
aspar = 1.3 #a*, parallel
bspar = 0.8 #b*, parallel
gamspar = np.radians(110) #gamma*, parallel

# out of plane
#dasperp = 0.05
#dbsperp = 0.2
#cs = 0.6

dasperp = 1 # range:  0.05-1ish
dbsperp = 1 # range: 0.05-1ish
cs = 1 # range: 0.2-1ish

#vectors
def gen_q(dasperp, dbsperp, cs, debug=False):
    As = np.array([aspar, 0, dasperp])
    Bs = np.array([bspar * np.cos(gamspar), bspar * np.sin(gamspar), dbsperp])
    Cs = np.array([0, 0, cs])

    #indices
    i = 0
    M = 2

    n = (M+1) * (M+1+M) * (M+1 + M)


    qxy = np.ndarray(n)
    qz = np.ndarray(n)

    for h in range(-M, M+1): #matlab has inclusive ranges but python doesn't
        for k in range(-M, M+1):
            for l in range(0, M+1):
                Q = h*As + k*Bs + l*Cs
                qxy[i] = np.sqrt(Q[0] * Q[0] + Q[1] * Q[1])
                qz[i] = Q[2]
                i = i+1
    if debug:
        #real space lattice vectors
        Vs = np.linalg.det(np.array([As,Bs,Cs]))
        A = 2 * np.pi * np.cross(Bs, Cs) / Vs
        B = 2 * np.pi * np.cross(Cs, As) / Vs
        C = 2 * np.pi * np.cross(As, Bs) / Vs
        a = np.linalg.norm(A)
        b = np.linalg.norm(B)
        c = np.linalg.norm(C)
        alf = np.degrees(np.arccos((B.T@C) / (b*c))) # @ is matrix multiplication, * is elementwise (kronecker product)
        bet = np.degrees(np.arccos((C.T@A) / (c*a)))
        gam = np.degrees(np.arccos((A.T@B) / (a*b)))

        print("a,b,c:", a,b,c)
        print("alpha, beta, gamma:",alf,bet,gam)

    return qxy, qz

def gen_optimize_q(actual_qxy, actual_qz):
    def f(x):
        dasperp, dbsperp, cs = x
        qxy, qz = gen_q(dasperp, dbsperp, cs)
        return np.linalg.norm(qxy-actual_qxy) + np.linalg.norm(qz-actual_qz)
    return f
        
#print(i)
#print(n)

qxy, qz = gen_q(dasperp, dbsperp, cs, debug=True)
result = optimize.basinhopping(gen_optimize_q(qxy, qz), np.array([1,1,1]), minimizer_kwargs={'options': {'disp': False}}, niter=10)

dasperp_op, dbsperp_op, cs_op = result.x
optimized_qxy, optimized_qz = gen_q(dasperp_op, dbsperp_op, cs_op, debug=True)
#optimized_qxy = result.x[0]
#optimized_qz = result.x[1]

plt.scatter(qxy, qz)
plt.scatter(optimized_qxy, optimized_qz, s=2)
plt.show()

