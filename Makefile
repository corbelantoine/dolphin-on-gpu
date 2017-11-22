CPP=g++
CPPFLAGS=-Wall -pedantic -std=c++14
LDFLAGS=-lboost_math_c99 -lboost_thread-mt -lm -lcgal -lmpfr -lgmp

SRCS=src/app/test.cpp \
		 src/finance/asset.cpp \
		 src/finance/portfolio.cpp \
     src/helpers/date.cpp \
		 src/parsing/parse.cpp
     #  src/optimization/optimizer.cpp

OBJS=$(SRCS:.cpp=.o)

all: test

test: $(OBJS)
	$(CPP) $(CPPFLAGS) $(LDFLAGS) $(OBJS) -o test

.PHONY: clean

clean:
	$(RM) $(OBJS)

distclean: clean
	$(RM) test
