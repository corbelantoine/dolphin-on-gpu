CPP=nvcc --std=c++11 -G -g -lineinfo -Wno-deprecated-gpu-targets 
CPPFLAGS=-lcuda -lcublas --device-c|-dc

LDIR = lib
_LIB=cvxgen/ldl.cu \
	cvxgen/matrix_support.cu \
	cvxgen/solver.cu 
LIB = $(parsubst %, $(LDIR)/%, $(_LIB))

SDIR = src
_SRC = finance/asset.cu \
    finance/portfolio.cu \
    helpers/date.cu \
	optimization/optimizer.cu
SRC = $(patsubst %, $(SDIR)/%, $(_SRC))

ODIR = obj

_OBJ = $(_SRC:.cu=.o)
_OBJ += $(_LIB:.cu=.o)
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

# SRC=src/app/main.cu \
#     src/finance/asset.cu \
#     src/finance/portfolio.cu \
#     src/helpers/date.cu \
#     src/parsing/parse.cpp \
# 	src/optimization/optimizer.cu

BIN=main

all: $(BIN)
	
$(BIN): $(OBJ)
	$(CPP) $(LIB) $(SRC) $(CPPFLAGS) 
#	$(CPP) $(LIB) $(SRC) $(CPPFLAGS) -o $(BIN)

$(ODIR)/cvxgen/ldl.o: $(LDIR)/cvxgen/ldl.cu

$(ODIR)/cvxgen/matrix_support.o: $(LDIR)/cvxgen/matrix_support.cu

$(ODIR)/cvxgen/solver.o: $(LDIR)/cvxgen/solver.cu $(ODIR)/cvxgen/ldl.o $(ODIR)/cvxgen/matrix_support.o

$(ODIR)/helpers/date.o: $(SDIR)/helpers/date.cu

$(ODIR)/finance/asset.o: $(SDIR)/finance/asset.cu $(ODIR)/helpers/date.o

$(ODIR)/finance/portfolio.o: $(SDIR)/finance/portfolio.cu $(ODIR)/finance/asset.o

$(ODIR)/optimization/optimizer.o: $(SDIR)/optimization/optimizer.cu $(ODIR)/finance/portfolio.o $(ODIR)/cvxgen/solver.o 


clean:
	$(RM) $(OBJS) $(BIN) 

.PHONY: clean
