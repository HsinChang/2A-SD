# product between two sparse vectors
# M list of (u,val) pairs which are nonzero entries in M
def Mir(Mi, r):
    tot = 0
    for u, val in Mi:
        tot += val * r[u]
    return tot


# product between sparse matrix and vector
def Mr(M, r):
    r1 = {}
    for u in M:
        r1[u] = Mir(M[u], r)
    return r1


f = open("graph.txt", "r")

# build graph from input file (dictionary: nodes, list of successors)
S = {}
for line in f:
    u, v = [int(x) for x in line.split(',')]
    #v = v.rstrip('\n')
    if not (u in S):
        S[u] = []
    if not (v in S):
        S[v] = []
    S[u].append(v)

# build matrix M_G
M = {}
for u in S:
    for v in S[u]:
        if not (v in M):
            M[v] = []
        M[v].append([u, 1.0 / len(S[u])])

# PageRank algorithm
r = {}
beta = 1

for u in S:
    r[u] = 1.0 / len(S)
for i in range(1, 100):
    print(r)
    r = Mr(M, r)
    for u in r:
        r[u] = r[u] * beta + (1 - beta) / len(S)