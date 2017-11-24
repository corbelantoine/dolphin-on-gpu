/* Produced by CVXGEN, 2017-11-11 14:57:56 -0500.  */
/* CVXGEN is Copyright (C) 2006-2017 Jacob Mattingley, jem@cvxgen.com. */
/* The code in this file is Copyright (C) 2006-2017 Jacob Mattingley. */
/* CVXGEN, or solvers produced by CVXGEN, cannot be used for commercial */
/* applications without prior written permission from Jacob Mattingley. */

/* Filename: solver.h. */
/* Description: Header file with relevant definitions. */
#ifndef SOLVER_H
#define SOLVER_H
/* Uncomment the next line to remove all library dependencies. */
/*#define ZERO_LIBRARY_MODE */
#ifdef MATLAB_MEX_FILE
/* Matlab functions. MATLAB_MEX_FILE will be defined by the mex compiler. */
/* If you are not using the mex compiler, this functionality will not intrude, */
/* as it will be completely disabled at compile-time. */
#include "mex.h"
#else
#ifndef ZERO_LIBRARY_MODE
#include <stdio.h>
#endif
#endif
/* Space must be allocated somewhere (testsolver.c, csolve.c or your own */
/* program) for the global variables vars, params, work and settings. */
/* At the bottom of this file, they are externed. */
#ifndef ZERO_LIBRARY_MODE
#include <math.h>
#define pm(A, m, n) printmatrix(#A, A, m, n, 1)
#endif

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

struct Params {
  double Sigma[400];
  double lambda[1];
  double Returns[20];
};

struct Vars {
  double *Weights; /* 20 rows. */
};

struct Workspace {
  double h[40];
  double s_inv[40];
  double s_inv_z[40];
  double b[1];
  double q[20];
  double rhs[101];
  double x[101];
  double *s;
  double *z;
  double *y;
  double lhs_aff[101];
  double lhs_cc[101];
  double buffer[101];
  double buffer2[101];
  double KKT[390];
  double L[290];
  double d[101];
  double v[101];
  double d_inv[101];
  double gap;
  double optval;
  double ineq_resid_squared;
  double eq_resid_squared;
  double block_33[1];
  /* Pre-op symbols. */
  int converged;
};

struct Settings {
  double resid_tol;
  double eps;
  int max_iters;
  int refine_steps;
  int better_start;
  /* Better start obviates the need for s_init and z_init. */
  double s_init;
  double z_init;
  int verbose;
  /* Show extra details of the iterative refinement steps. */
  int verbose_refinement;
  int debug;
  /* For regularization. Minimum value of abs(D_ii) in the kkt D factor. */
  double kkt_reg;
};

// extern Vars vars;
// extern Params params;
// extern Workspace work;
// extern Settings settings;


/* Function definitions in ldl.c: */
CUDA_CALLABLE_MEMBER void ldl_solve(double *target, double *var, Workspace& work, Settings& settings);
CUDA_CALLABLE_MEMBER void ldl_factor(Workspace& work, Settings& settings);
CUDA_CALLABLE_MEMBER double check_factorization(Workspace& work);
CUDA_CALLABLE_MEMBER void matrix_multiply(double *result, double *source, Workspace& work, Settings& settings);
CUDA_CALLABLE_MEMBER double check_residual(double *target, double *multiplicand, Workspace& work, Settings& settings);
CUDA_CALLABLE_MEMBER void fill_KKT(Workspace& work, Params& params);

/* Function definitions in matrix_support.c: */
CUDA_CALLABLE_MEMBER void multbymA(double *lhs, double *rhs);
CUDA_CALLABLE_MEMBER void multbymAT(double *lhs, double *rhs);
CUDA_CALLABLE_MEMBER void multbymG(double *lhs, double *rhs);
CUDA_CALLABLE_MEMBER void multbymGT(double *lhs, double *rhs);
CUDA_CALLABLE_MEMBER void multbyP(double *lhs, double *rhs, Params& params);
CUDA_CALLABLE_MEMBER void fillq(Workspace& work, Params& params);
CUDA_CALLABLE_MEMBER void fillh(Workspace& work);
CUDA_CALLABLE_MEMBER void fillb(Workspace& work);
CUDA_CALLABLE_MEMBER void pre_ops(void);

/* Function definitions in solver.c: */
CUDA_CALLABLE_MEMBER double eval_gap(Workspace& work);
CUDA_CALLABLE_MEMBER void set_defaults(Settings& settings);
CUDA_CALLABLE_MEMBER void setup_pointers(Workspace& work, Vars& vars);
CUDA_CALLABLE_MEMBER void setup_indexing(Workspace& work, Vars& vars);
CUDA_CALLABLE_MEMBER void set_start(Workspace& work, Settings& settings);
CUDA_CALLABLE_MEMBER double eval_objv(Workspace& work, Params& params);
CUDA_CALLABLE_MEMBER void fillrhs_aff(Workspace& work, Params& params);
CUDA_CALLABLE_MEMBER void fillrhs_cc(Workspace& work);
CUDA_CALLABLE_MEMBER void refine(double *target, double *var, Workspace& work, Settings& settings);
CUDA_CALLABLE_MEMBER double calc_ineq_resid_squared(Workspace& work);
CUDA_CALLABLE_MEMBER double calc_eq_resid_squared(Workspace& work);
CUDA_CALLABLE_MEMBER void better_start(Workspace& work, Settings& settings, Params& params);
CUDA_CALLABLE_MEMBER void fillrhs_start(Workspace& work);
CUDA_CALLABLE_MEMBER long solve(Workspace& work, Settings& settings, Params& params, Vars& vars);

/* Function definitions in util.c: */
void tic(void);
float toc(void);
float tocq(void);
void printmatrix(char *name, double *A, int m, int n, int sparse);
double unif(double lower, double upper);
float ran1(long*idum, int reset);
float randn_internal(long *idum, int reset);
double randn(void);
void reset_rand(void);

#endif
