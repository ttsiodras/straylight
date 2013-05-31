#!/usr/bin/env python
import os
import sys


def main():
    if len(sys.argv) != 2:
        print "Usage:", sys.argv[0], "<input_image>"
        sys.exit(1)
    filename = "/tmp/image." + str(os.getpid()) + ".pgm"
    ppm = open(filename, "w")
    ppm.write('P5\n5271 813\n255\n')
    m = 0.
    for line in open(sys.argv[1]).readlines():
        data = line.split()
        for num in data:
            num = float(num)
            if num > m:
                m = num
    print "Maximum value detected:", m
    for line in open(sys.argv[1]).readlines():
        data = line.split()
        for num in data:
            i = int(255.*float(num)/m)
            ppm.write(chr(i))
    ppm.close()
    os.system("display " + filename)
    os.unlink(filename)

if __name__ == "__main__":
    main()
