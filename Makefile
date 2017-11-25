# Setting GPU compiler and flags
CPP=nvcc --std=c++11 -G -g -lineinfo -Wno-deprecated-gpu-targets 
CPPFLAGS=-lcuda -lcublas --device-c|-dc

# Setting library files and dir
LDIR = lib
_LIB=cvxgen/ldl.cu \
	cvxgen/matrix_support.cu \
	cvxgen/solver.cu 

LIB = $(parsubst %, $(LDIR)/%, $(_LIB))

# Setting src files and dir
SDIR = src
_SRC = finance/asset.cu \
    finance/portfolio.cu \
    helpers/date.cu \
		optimization/optimizer.cu \
		app/main.cu \
		src/parsing/parse.cpp

SRC = $(patsubst %, $(SDIR)/%, $(_SRC))

# Setting obj files and dir
ODIR = obj
_OBJ = $(_SRC:.cu=.o)
_OBJ += $(_LIB:.cu=.o)
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

BIN=main

all: $(OBJ)
	$(CPP) $(CPPFLAGS) $(LIB) $(SRC) -o $(BIN)

$(ODIR)/cvxgen/ldl.o: $(LDIR)/cvxgen/ldl.cu

$(ODIR)/cvxgen/matrix_support.o: $(LDIR)/cvxgen/matrix_support.cu

$(ODIR)/cvxgen/solver.o: $(LDIR)/cvxgen/solver.cu $(ODIR)/cvxgen/ldl.o $(ODIR)/cvxgen/matrix_support.o

$(ODIR)/helpers/date.o: $(SDIR)/helpers/date.cu

$(ODIR)/helpers/parse.o: $(SDIR)/helpers/parse.cu $(ODIR)/helpers/date.o

$(ODIR)/finance/asset.o: $(SDIR)/finance/asset.cu $(ODIR)/helpers/date.o

$(ODIR)/finance/portfolio.o: $(SDIR)/finance/portfolio.cu $(ODIR)/finance/asset.o

$(ODIR)/optimization/optimizer.o: $(SDIR)/optimization/optimizer.cu $(ODIR)/finance/portfolio.o $(ODIR)/cvxgen/solver.o 

$(ODIR)/app/main.o: $(SDIR)/app/main.cu $(ODIR)/optimization/optimizer.o 

clean:
	$(RM) $(OBJ) $(BIN) 

.PHONY: clean
