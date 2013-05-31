DISTCLEANFILES = *~ *.~*

SUBDIRS = src

EXTRA_DIST = \
	README COPYING INSTALL AUTHORS NEWS ChangeLog                   \
	contrib/stats.py                                                \
	contrib/checkCompilationsOptions.sh                             \
	contrib/oldMakefiles/Makefile.benchmark                         \
	contrib/oldMakefiles/Makefile.normal                            \
	contrib/elementsDiff.py                                         \
	contrib/elements.pl                                             \
	contrib/img.py                                                  \
	contrib/.gitignore                                              \
	contrib/00README                                                \
	doc/Makefile                                                    \
	doc/gnuplot.script                                              \
	doc/report.data                                                 \
	doc/finalReport.tex                                             \
	.gitignore

test:
	./contrib/checkCompilationsOptions.sh || exit 1
