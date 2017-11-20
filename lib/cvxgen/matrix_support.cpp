/* Produced by CVXGEN, 2017-11-11 14:57:56 -0500.  */
/* CVXGEN is Copyright (C) 2006-2017 Jacob Mattingley, jem@cvxgen.com. */
/* The code in this file is Copyright (C) 2006-2017 Jacob Mattingley. */
/* CVXGEN, or solvers produced by CVXGEN, cannot be used for commercial */
/* applications without prior written permission from Jacob Mattingley. */

/* Filename: matrix_support.c. */
/* Description: Support functions for matrix multiplication and vector filling. */
#include "solver.hpp"

void multbymA(double *lhs, double *rhs) {
  lhs[0] = -rhs[0]*(1)-rhs[1]*(1)-rhs[2]*(1)-rhs[3]*(1)-rhs[4]*(1)-rhs[5]*(1)-rhs[6]*(1)-rhs[7]*(1)-rhs[8]*(1)-rhs[9]*(1)-rhs[10]*(1)-rhs[11]*(1)-rhs[12]*(1)-rhs[13]*(1)-rhs[14]*(1)-rhs[15]*(1)-rhs[16]*(1)-rhs[17]*(1)-rhs[18]*(1)-rhs[19]*(1);
}

void multbymAT(double *lhs, double *rhs) {
  for (int i = 0; i < 20; ++i)
    lhs[i] = -rhs[0]*(1);
}

void multbymG(double *lhs, double *rhs) {
  for (int i = 0; i < 20; ++i)
    lhs[i] = -rhs[i] * (-1);
}

void multbymGT(double *lhs, double *rhs) {
  for (int i = 0; i < 20; ++i)
    lhs[i] = -rhs[i] * (-1);
}

void multbyP(double *lhs, double *rhs, Params& params) {
  for (int i = 0; i < 20; ++i) {
    lhs[i] = 0;
    for (int j = 0; j < 20; ++j)
      lhs[i] += rhs[j] * 2 * params.Sigma[j * 20 + i];
  }
}

void fillq(Workspace& work, Params& params) {
  for (int i = 0; i < 20; ++i)
    work.q[i] = -params.lambda[i]*params.Returns[i];
}

void fillh(Workspace& work) {
  for (int i = 0; i < 20; ++i)
    work.h[i] = 0.1;
}

void fillb(Workspace& work) {
  work.b[0] = 1;
}

void pre_ops(void) {
}
