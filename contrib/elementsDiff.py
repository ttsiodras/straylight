#!/usr/bin/env python
import sys
f2 = open(sys.argv[2])
for line1 in open(sys.argv[1]).readlines():
    line2 = f2.readline()
    for a,b in zip(line1.split(), line2.split()):
        print abs(float(a)-float(b))
