import sys
import string
import re

fs = open(sys.argv[1], 'r+')

lines = fs.readlines()
new_data = []
for line in lines:
    data = re.split('\s+', line)
    if len(data) > 2 and data[1] == 'BUILD':
        data[2] = str(int(data[2])+1);
        ver = ' '.join(data)
        ver += '\n'
        new_data.append(ver)
    else:
        new_data.append(line)
fs.seek(0)
fs.writelines(new_data)
fs.close()