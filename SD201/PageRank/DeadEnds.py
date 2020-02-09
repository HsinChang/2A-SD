def dead_end_check(S):
    e = False
    for u in S:
        if [u] == S[u]:
            e = True
        if S[u] == []:
            e = True
    return e

def dead_end_index(S):
    index = []
    for u in S:
        if [u] == S[u]:
            index.append(u)
        if S[u] == []:
            index.append(u)
    return index
#read the graph
f=open("testGraph.txt","r")

S={}
for line in f:
    u,v= [ x for x in line.split(',') ]
    v = v.rstrip('\n')
    if not(u in S):
        S[u]=[]
    if not(v in S):
        S[v]=[]
    S[u].append(v)

while dead_end_check(S):
    ind = dead_end_index(S)
    for i in ind:
        del S[i]
        for u in S:
            if i in S[u]:
                S[u].remove(i)
                a = 1
print(S)
