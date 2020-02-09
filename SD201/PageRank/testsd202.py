c = [('A', 'B'), ('D', 'A'), ('D', 'A', 'B'), ('C', 'D', 'A'), ('C', 'A', 'B')]
d = []
d.extend((sorted(c[0])))
leng = len(d[0])
for i in range(1, len(c)):
    e = c[i]
    if len(c[i]) == leng:
        d.extend([c[i]])
print(d)