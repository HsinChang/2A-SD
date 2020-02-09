import sys
import numpy as np
import math

#product between two sparse vectors
#M list of (u,val) pairs which are nonzero entries in M
def Mir (Mi,r):
    tot=0
    for u,val in Mi:
        tot+=val*r[u]
    return tot

#product between sparse matrix and vector
def Mr (M,r):
    r1={}
    for u in M:
        r1[u]=Mir(M[u],r)
    return r1

f=open("graph","r")

#compute the L2 norm of u-v
#u,v sparse vectors represented as dictionaries
def diff (u,v):
    d={}
    tot=0
    for e in u:
        if e not in v: tot+=u[e]**2
        else: tot+=(u[e]-v[e])**2
    for e in v:
        if e in u: continue
        else: tot+=v[e]**2
    return math.sqrt(tot)
        

#input file: list of edges i->j each one represented in a line "i j"
#graph represented as a dictionary {nodes: list of successors}
S={}
for line in f:
    u,v= [ int(x) for x in line.split() ]
    if not(u in S):
        S[u]=[]
    if not(v in S):
        S[v]=[]
    S[u].append(v)

#build matrix M_G
M={}
for u in S:
        for v in S[u]:
            if not(v in M):
                M[v]=[]
            M[v].append([u,1.0/len(S[u])])

            
            
#PageRank algorithm
r={}
beta=1
eps=0.001
maxIt=1000

for u in S:
    r[u]=1.0/len(S)
for i in range(1,maxIt):
    print r
    rOld=r
    r = Mr(M,r)
    for u in r:
        r[u]=r[u]*beta+ (1-beta)/len(S)
    rNew=r

    if (diff(rOld,rNew)<eps): break