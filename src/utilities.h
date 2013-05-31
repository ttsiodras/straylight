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

#ifndef __DEBUG_MESSAGES_H__
#define __DEBUG_MESSAGES_H__

#include <stdarg.h>

// debug_printf is used for logging and error reporting
//
enum DebugLevel {
    LVL_PANIC=0,
    LVL_WARN,
    LVL_INFO
};

extern DebugLevel g_debugLevel;

long getTimeInMS();
void debug_printf(DebugLevel level, const char *fmt, ...);

#endif
