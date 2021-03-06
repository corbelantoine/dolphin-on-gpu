/* Produced by CVXGEN, 2017-11-20 12:18:49 -0500.  */
/* CVXGEN is Copyright (C) 2006-2017 Jacob Mattingley, jem@cvxgen.com. */
/* The code in this file is Copyright (C) 2006-2017 Jacob Mattingley. */
/* CVXGEN, or solvers produced by CVXGEN, cannot be used for commercial */
/* applications without prior written permission from Jacob Mattingley. */

/* Filename: matrix_support.c. */
/* Description: Support functions for matrix multiplication and vector filling. */
#include "solver.cuh"

CUDA_CALLABLE_MEMBER
void multbymA(double *lhs, double *rhs) {
  lhs[0] = -rhs[0]*(1)-rhs[1]*(1)-rhs[2]*(1)-rhs[3]*(1)-rhs[4]*(1)-rhs[5]*(1)-rhs[6]*(1)-rhs[7]*(1)-rhs[8]*(1)-rhs[9]*(1)-rhs[10]*(1)-rhs[11]*(1)-rhs[12]*(1)-rhs[13]*(1)-rhs[14]*(1)-rhs[15]*(1)-rhs[16]*(1)-rhs[17]*(1)-rhs[18]*(1)-rhs[19]*(1);
}

CUDA_CALLABLE_MEMBER
void multbymAT(double *lhs, double *rhs) {
  for (int i = 0; i < 20; ++i)
    lhs[i] = -rhs[0]*(1);
}

CUDA_CALLABLE_MEMBER
void multbymG(double *lhs, double *rhs) {
  for (int i = 0; i < 20; ++i) {
    lhs[i] = -rhs[i];
    lhs[i + 20] = rhs[i];
  }
}

CUDA_CALLABLE_MEMBER
void multbymGT(double *lhs, double *rhs) {
  for (int i = 0; i < 20; ++i)
    lhs[i] = -rhs[i] + rhs[i + 20];
}

CUDA_CALLABLE_MEMBER
void multbyP(double *lhs, double *rhs, Params& params) {
  for (int i = 0; i < 20; ++i) {
    lhs[i] = 0;
    for (int j = 0; j < 20; ++j)
      lhs[i] += rhs[j] * 2 * params.Sigma[j * 20 + i];
  }
}

CUDA_CALLABLE_MEMBER
void fillq(Workspace& work, Params& params) {
  for (int i = 0; i < 20; ++i)
    work.q[i] = -params.lambda[i] * params.Returns[i];
}

CUDA_CALLABLE_MEMBER
void fillh(Workspace& work) {
  for (int i = 0; i < 20; ++i) {
    work.h[i] = 0.2;
    work.h[i + 20] = -0.01;
  }
}

CUDA_CALLABLE_MEMBER
void fillb(Workspace& work) {
  work.b[0] = 1;
}

CUDA_CALLABLE_MEMBER
void pre_ops(void) {
}
