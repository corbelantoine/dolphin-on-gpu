CPP=nvcc --std=c++11 -G -g -lineinfo -Wno-deprecated-gpu-targets 

LIB=lib/cvxgen/ldl.cpp \
	lib/cvxgen/matrix_support.cpp \
	lib/cvxgen/solver.cpp \
	lib/cvxgen/util.cpp 
	
SRC=src/app/main.cu \
    src/finance/asset.cu \
    src/finance/portfolio.cu \
    src/helpers/date.cu \
    src/parsing/parse.cpp \
	src/optimization/optimizer.cu

# UNIT=src/tests/unit_test.cc

BIN=main

# UBIN=unit

CPPFLAGS=-lcuda -lcublas --device-c|-dc

all:
	$(CPP) $(LIB) $(SRC) $(CPPFLAGS) -o $(BIN)

check: all
	./$(BIN)

# unit:
# 	$(CPP) $(LIB) $(UNIT) $(CPPFLAGS) -o $(UBIN)
# 	./$(UBIN)

clean:
	$(RM) $(BIN) # $(UBIN)

.PHONY: clean
