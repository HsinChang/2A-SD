import os
import re

f1 = open('WebGraph.txt', 'r+')

directory = 'toyset/'

for filename in os.listdir(directory):
    if filename.endswith(".html"):
        path = 'toyset/' + filename
        f = open(path, encoding="utf8")
        s = f.read()
    urls = re.findall(r'<a href="(.*?)"', s)
    for edge in urls:
        item = filename + ',' + edge
        f1.write(item)
        f1.write('\n')

