#!/bin/bash
ulimit -s 73728 || exit 1
cd $(dirname $0)/../ || exit 1
for i in \
    "--disable-cuda --disable-eigen --disable-openmp" \
    "--disable-cuda --disable-eigen" \
    "--disable-cuda --enable-eigen" \
    "--enable-cuda " \
    "--enable-double --disable-cuda --disable-eigen --disable-openmp" \
    "--enable-double --disable-cuda --disable-eigen" \
    "--enable-double --enable-cuda " ; do 
    echo -e "\n================================================================"
    echo "Configuring with $i"
    echo "================================================================"
    ./configure $i >/dev/null || break 
    make clean >/dev/null
    make >/dev/null || break 
    mkdir -p output
    rm -f output/* 2>/dev/null
    if [ -f output.from.ESA.all.channels.stage13/stage13_1 ] ; then
        echo "(Normal runs - to check the results)"
        for j in 1 2 3 ; do
            ./src/strayLight -c $j || exit 1
            ./contrib/elementsDiff.py \
                output/stage13_$j \
                output.from.ESA.all.channels.stage13/stage13_$j | \
                stats.py  | \
                grep Overall | \
                sed 's,^.*Overall:,Channel XX\, difference per sample between IDL and C++:\n\t,' | \
                sed  "s,XX,$j,"
        done
        echo "--------------------------"
    fi
    echo "(Benchmark run - to check the speed)"
    sync
    ./src/strayLight -v -b > test-results."$i"
    cat test-results."$i" | \
        awk '/Stage 10/{a=0;} /ms$/{a+=$4;} /Stage 13/{print a;}' | \
        sed 1d | \
        stats.py | \
        grep Min | \
        sed 's,^.*Min:,\tTime (in ms) per frame:,'
done
