# Setting GPU compiler and flags
NVCC = nvcc 
NVCCFLAGS= --std=c++11 -G -g -lineinfo -Wno-deprecated-gpu-targets

CUDALIBS = -lcuda -lcublas 

##############################################################################

# Sources

# Setting library files and dir
LDIR = lib
_LIB = cvxgen/ldl.cu \
	   cvxgen/matrix_support.cu \
	   cvxgen/solver.cu 

LIB = $(parsubst %, $(LDIR)/%, $(_LIB))

# Setting src files and dir
SDIR = src
_SRC = finance/asset.cu \
	   finance/portfolio.cu \
	   helpers/date.cu \
	   helpers/parse.cu \
	   optimization/optimizer.cu \
	   app/main.cu 

SRC = $(patsubst %, $(SDIR)/%, $(_SRC))

##############################################################################

# Objects

ODIR = obj
_OBJ = $(_SRC:.cu=.o)
_OBJ += $(_LIB:.cu=.o)
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

##############################################################################

# link
all: $(OBJ)
	$(NVCC) $(NVCCFLAGS) $(OBJ) $(CUDALIBS) -o dolphin

##############################################################################
						# Comiple individually
		######################################################
					# Source code compiling

$(ODIR)/app/main.o: $(SDIR)/app/main.cu $(ODIR)/optimization/optimizer.o 
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(ODIR)/optimization/optimizer.o: $(SDIR)/optimization/optimizer.cu $(ODIR)/finance/portfolio.o $(ODIR)/cvxgen/solver.o 
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(ODIR)/finance/portfolio.o: $(SDIR)/finance/portfolio.cu $(ODIR)/finance/asset.o
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(ODIR)/finance/asset.o: $(SDIR)/finance/asset.cu $(ODIR)/helpers/date.o
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(ODIR)/helpers/parse.o: $(SDIR)/helpers/parse.cu $(ODIR)/helpers/date.o
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(ODIR)/helpers/date.o: $(SDIR)/helpers/date.cu
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

		######################################################
					# Cvxgen library compiling

$(ODIR)/cvxgen/ldl.o: $(LDIR)/cvxgen/ldl.cu
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(ODIR)/cvxgen/matrix_support.o: $(LDIR)/cvxgen/matrix_support.cu
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(ODIR)/cvxgen/solver.o: $(LDIR)/cvxgen/solver.cu $(ODIR)/cvxgen/ldl.o $(ODIR)/cvxgen/matrix_support.o 
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

##############################################################################

run: build
	$(EXEC) ./dolphin

clean:
	$(RM) $(OBJ) $(BIN) 

.PHONY: clean

##############################################################################
