import sys
import os

file_ori = open(sys.argv[1], "r")
path = file_ori.name[:-4]
os.mkdir(path)
i = 0

for line in file_ori:
	new_file = open(path + '/' + path + str(i) + '.txt', "w" )
	new_file.write(line)
	i += 1
