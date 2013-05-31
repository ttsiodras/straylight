/*
 .----------------------------------------------------------.
 |  Optimized StrayLight implementation in C++/OpenMP/CUDA  |
 |      Task 2.2 of "TASTE Maintenance and Evolutions"      |
 |           P12001 - PFL-PTE/JK/ats/412.2012i              |
 |                                                          |
 |  Contact Point:                                          |
 |           Thanassis Tsiodras, Dr.-Ing.                   |
 |           ttsiodras@semantix.gr / ttsiodras@gmail.com    |
 |           NeuroPublic S.A.                               |
 |                                                          |
 |   Licensed under the GNU Lesser General Public License,  |
 |   details here:  http://www.gnu.org/licenses/lgpl.html   |
 `----------------------------------------------------------'

*/

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <time.h>

#include "utilities.h"

DebugLevel g_debugLevel = LVL_WARN;

// debug_printf is used for logging and error reporting

long getTimeInMS()
{
    struct timespec ts;
    clock_gettime( CLOCK_REALTIME, &ts );
    return ts.tv_sec*1000UL+ts.tv_nsec/1000000UL;
}

void debug_printf(DebugLevel level, const char *fmt, ...)
{
    static char message[131072];

    if (level <= g_debugLevel) {
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(message, sizeof message, fmt, ap);
	printf(message);
	va_end(ap);
    }
    if (level == LVL_PANIC) {
        fflush(stdout);
	exit(1);
    }
}
