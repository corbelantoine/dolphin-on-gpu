/* Produced by CVXGEN, 2017-11-11 14:57:56 -0500.  */
/* CVXGEN is Copyright (C) 2006-2017 Jacob Mattingley, jem@cvxgen.com. */
/* The code in this file is Copyright (C) 2006-2017 Jacob Mattingley. */
/* CVXGEN, or solvers produced by CVXGEN, cannot be used for commercial */
/* applications without prior written permission from Jacob Mattingley. */

/* Filename: matrix_support.c. */
/* Description: Support functions for matrix multiplication and vector filling. */
#include "solver.h"
void multbymA(double *lhs, double *rhs) {
  lhs[0] = -rhs[0]*(1)-rhs[1]*(1)-rhs[2]*(1)-rhs[3]*(1)-rhs[4]*(1)-rhs[5]*(1)-rhs[6]*(1)-rhs[7]*(1)-rhs[8]*(1)-rhs[9]*(1)-rhs[10]*(1)-rhs[11]*(1)-rhs[12]*(1)-rhs[13]*(1)-rhs[14]*(1)-rhs[15]*(1)-rhs[16]*(1)-rhs[17]*(1)-rhs[18]*(1)-rhs[19]*(1);
}
void multbymAT(double *lhs, double *rhs) {
  lhs[0] = -rhs[0]*(1);
  lhs[1] = -rhs[0]*(1);
  lhs[2] = -rhs[0]*(1);
  lhs[3] = -rhs[0]*(1);
  lhs[4] = -rhs[0]*(1);
  lhs[5] = -rhs[0]*(1);
  lhs[6] = -rhs[0]*(1);
  lhs[7] = -rhs[0]*(1);
  lhs[8] = -rhs[0]*(1);
  lhs[9] = -rhs[0]*(1);
  lhs[10] = -rhs[0]*(1);
  lhs[11] = -rhs[0]*(1);
  lhs[12] = -rhs[0]*(1);
  lhs[13] = -rhs[0]*(1);
  lhs[14] = -rhs[0]*(1);
  lhs[15] = -rhs[0]*(1);
  lhs[16] = -rhs[0]*(1);
  lhs[17] = -rhs[0]*(1);
  lhs[18] = -rhs[0]*(1);
  lhs[19] = -rhs[0]*(1);
}
void multbymG(double *lhs, double *rhs) {
  lhs[0] = -rhs[0]*(-1);
  lhs[1] = -rhs[1]*(-1);
  lhs[2] = -rhs[2]*(-1);
  lhs[3] = -rhs[3]*(-1);
  lhs[4] = -rhs[4]*(-1);
  lhs[5] = -rhs[5]*(-1);
  lhs[6] = -rhs[6]*(-1);
  lhs[7] = -rhs[7]*(-1);
  lhs[8] = -rhs[8]*(-1);
  lhs[9] = -rhs[9]*(-1);
  lhs[10] = -rhs[10]*(-1);
  lhs[11] = -rhs[11]*(-1);
  lhs[12] = -rhs[12]*(-1);
  lhs[13] = -rhs[13]*(-1);
  lhs[14] = -rhs[14]*(-1);
  lhs[15] = -rhs[15]*(-1);
  lhs[16] = -rhs[16]*(-1);
  lhs[17] = -rhs[17]*(-1);
  lhs[18] = -rhs[18]*(-1);
  lhs[19] = -rhs[19]*(-1);
}
void multbymGT(double *lhs, double *rhs) {
  lhs[0] = -rhs[0]*(-1);
  lhs[1] = -rhs[1]*(-1);
  lhs[2] = -rhs[2]*(-1);
  lhs[3] = -rhs[3]*(-1);
  lhs[4] = -rhs[4]*(-1);
  lhs[5] = -rhs[5]*(-1);
  lhs[6] = -rhs[6]*(-1);
  lhs[7] = -rhs[7]*(-1);
  lhs[8] = -rhs[8]*(-1);
  lhs[9] = -rhs[9]*(-1);
  lhs[10] = -rhs[10]*(-1);
  lhs[11] = -rhs[11]*(-1);
  lhs[12] = -rhs[12]*(-1);
  lhs[13] = -rhs[13]*(-1);
  lhs[14] = -rhs[14]*(-1);
  lhs[15] = -rhs[15]*(-1);
  lhs[16] = -rhs[16]*(-1);
  lhs[17] = -rhs[17]*(-1);
  lhs[18] = -rhs[18]*(-1);
  lhs[19] = -rhs[19]*(-1);
}
void multbyP(double *lhs, double *rhs) {
  /* TODO use the fact that P is symmetric? */
  /* TODO check doubling / half factor etc. */
  lhs[0] = rhs[0]*(2*params.Sigma[0])+rhs[1]*(2*params.Sigma[20])+rhs[2]*(2*params.Sigma[40])+rhs[3]*(2*params.Sigma[60])+rhs[4]*(2*params.Sigma[80])+rhs[5]*(2*params.Sigma[100])+rhs[6]*(2*params.Sigma[120])+rhs[7]*(2*params.Sigma[140])+rhs[8]*(2*params.Sigma[160])+rhs[9]*(2*params.Sigma[180])+rhs[10]*(2*params.Sigma[200])+rhs[11]*(2*params.Sigma[220])+rhs[12]*(2*params.Sigma[240])+rhs[13]*(2*params.Sigma[260])+rhs[14]*(2*params.Sigma[280])+rhs[15]*(2*params.Sigma[300])+rhs[16]*(2*params.Sigma[320])+rhs[17]*(2*params.Sigma[340])+rhs[18]*(2*params.Sigma[360])+rhs[19]*(2*params.Sigma[380]);
  lhs[1] = rhs[0]*(2*params.Sigma[1])+rhs[1]*(2*params.Sigma[21])+rhs[2]*(2*params.Sigma[41])+rhs[3]*(2*params.Sigma[61])+rhs[4]*(2*params.Sigma[81])+rhs[5]*(2*params.Sigma[101])+rhs[6]*(2*params.Sigma[121])+rhs[7]*(2*params.Sigma[141])+rhs[8]*(2*params.Sigma[161])+rhs[9]*(2*params.Sigma[181])+rhs[10]*(2*params.Sigma[201])+rhs[11]*(2*params.Sigma[221])+rhs[12]*(2*params.Sigma[241])+rhs[13]*(2*params.Sigma[261])+rhs[14]*(2*params.Sigma[281])+rhs[15]*(2*params.Sigma[301])+rhs[16]*(2*params.Sigma[321])+rhs[17]*(2*params.Sigma[341])+rhs[18]*(2*params.Sigma[361])+rhs[19]*(2*params.Sigma[381]);
  lhs[2] = rhs[0]*(2*params.Sigma[2])+rhs[1]*(2*params.Sigma[22])+rhs[2]*(2*params.Sigma[42])+rhs[3]*(2*params.Sigma[62])+rhs[4]*(2*params.Sigma[82])+rhs[5]*(2*params.Sigma[102])+rhs[6]*(2*params.Sigma[122])+rhs[7]*(2*params.Sigma[142])+rhs[8]*(2*params.Sigma[162])+rhs[9]*(2*params.Sigma[182])+rhs[10]*(2*params.Sigma[202])+rhs[11]*(2*params.Sigma[222])+rhs[12]*(2*params.Sigma[242])+rhs[13]*(2*params.Sigma[262])+rhs[14]*(2*params.Sigma[282])+rhs[15]*(2*params.Sigma[302])+rhs[16]*(2*params.Sigma[322])+rhs[17]*(2*params.Sigma[342])+rhs[18]*(2*params.Sigma[362])+rhs[19]*(2*params.Sigma[382]);
  lhs[3] = rhs[0]*(2*params.Sigma[3])+rhs[1]*(2*params.Sigma[23])+rhs[2]*(2*params.Sigma[43])+rhs[3]*(2*params.Sigma[63])+rhs[4]*(2*params.Sigma[83])+rhs[5]*(2*params.Sigma[103])+rhs[6]*(2*params.Sigma[123])+rhs[7]*(2*params.Sigma[143])+rhs[8]*(2*params.Sigma[163])+rhs[9]*(2*params.Sigma[183])+rhs[10]*(2*params.Sigma[203])+rhs[11]*(2*params.Sigma[223])+rhs[12]*(2*params.Sigma[243])+rhs[13]*(2*params.Sigma[263])+rhs[14]*(2*params.Sigma[283])+rhs[15]*(2*params.Sigma[303])+rhs[16]*(2*params.Sigma[323])+rhs[17]*(2*params.Sigma[343])+rhs[18]*(2*params.Sigma[363])+rhs[19]*(2*params.Sigma[383]);
  lhs[4] = rhs[0]*(2*params.Sigma[4])+rhs[1]*(2*params.Sigma[24])+rhs[2]*(2*params.Sigma[44])+rhs[3]*(2*params.Sigma[64])+rhs[4]*(2*params.Sigma[84])+rhs[5]*(2*params.Sigma[104])+rhs[6]*(2*params.Sigma[124])+rhs[7]*(2*params.Sigma[144])+rhs[8]*(2*params.Sigma[164])+rhs[9]*(2*params.Sigma[184])+rhs[10]*(2*params.Sigma[204])+rhs[11]*(2*params.Sigma[224])+rhs[12]*(2*params.Sigma[244])+rhs[13]*(2*params.Sigma[264])+rhs[14]*(2*params.Sigma[284])+rhs[15]*(2*params.Sigma[304])+rhs[16]*(2*params.Sigma[324])+rhs[17]*(2*params.Sigma[344])+rhs[18]*(2*params.Sigma[364])+rhs[19]*(2*params.Sigma[384]);
  lhs[5] = rhs[0]*(2*params.Sigma[5])+rhs[1]*(2*params.Sigma[25])+rhs[2]*(2*params.Sigma[45])+rhs[3]*(2*params.Sigma[65])+rhs[4]*(2*params.Sigma[85])+rhs[5]*(2*params.Sigma[105])+rhs[6]*(2*params.Sigma[125])+rhs[7]*(2*params.Sigma[145])+rhs[8]*(2*params.Sigma[165])+rhs[9]*(2*params.Sigma[185])+rhs[10]*(2*params.Sigma[205])+rhs[11]*(2*params.Sigma[225])+rhs[12]*(2*params.Sigma[245])+rhs[13]*(2*params.Sigma[265])+rhs[14]*(2*params.Sigma[285])+rhs[15]*(2*params.Sigma[305])+rhs[16]*(2*params.Sigma[325])+rhs[17]*(2*params.Sigma[345])+rhs[18]*(2*params.Sigma[365])+rhs[19]*(2*params.Sigma[385]);
  lhs[6] = rhs[0]*(2*params.Sigma[6])+rhs[1]*(2*params.Sigma[26])+rhs[2]*(2*params.Sigma[46])+rhs[3]*(2*params.Sigma[66])+rhs[4]*(2*params.Sigma[86])+rhs[5]*(2*params.Sigma[106])+rhs[6]*(2*params.Sigma[126])+rhs[7]*(2*params.Sigma[146])+rhs[8]*(2*params.Sigma[166])+rhs[9]*(2*params.Sigma[186])+rhs[10]*(2*params.Sigma[206])+rhs[11]*(2*params.Sigma[226])+rhs[12]*(2*params.Sigma[246])+rhs[13]*(2*params.Sigma[266])+rhs[14]*(2*params.Sigma[286])+rhs[15]*(2*params.Sigma[306])+rhs[16]*(2*params.Sigma[326])+rhs[17]*(2*params.Sigma[346])+rhs[18]*(2*params.Sigma[366])+rhs[19]*(2*params.Sigma[386]);
  lhs[7] = rhs[0]*(2*params.Sigma[7])+rhs[1]*(2*params.Sigma[27])+rhs[2]*(2*params.Sigma[47])+rhs[3]*(2*params.Sigma[67])+rhs[4]*(2*params.Sigma[87])+rhs[5]*(2*params.Sigma[107])+rhs[6]*(2*params.Sigma[127])+rhs[7]*(2*params.Sigma[147])+rhs[8]*(2*params.Sigma[167])+rhs[9]*(2*params.Sigma[187])+rhs[10]*(2*params.Sigma[207])+rhs[11]*(2*params.Sigma[227])+rhs[12]*(2*params.Sigma[247])+rhs[13]*(2*params.Sigma[267])+rhs[14]*(2*params.Sigma[287])+rhs[15]*(2*params.Sigma[307])+rhs[16]*(2*params.Sigma[327])+rhs[17]*(2*params.Sigma[347])+rhs[18]*(2*params.Sigma[367])+rhs[19]*(2*params.Sigma[387]);
  lhs[8] = rhs[0]*(2*params.Sigma[8])+rhs[1]*(2*params.Sigma[28])+rhs[2]*(2*params.Sigma[48])+rhs[3]*(2*params.Sigma[68])+rhs[4]*(2*params.Sigma[88])+rhs[5]*(2*params.Sigma[108])+rhs[6]*(2*params.Sigma[128])+rhs[7]*(2*params.Sigma[148])+rhs[8]*(2*params.Sigma[168])+rhs[9]*(2*params.Sigma[188])+rhs[10]*(2*params.Sigma[208])+rhs[11]*(2*params.Sigma[228])+rhs[12]*(2*params.Sigma[248])+rhs[13]*(2*params.Sigma[268])+rhs[14]*(2*params.Sigma[288])+rhs[15]*(2*params.Sigma[308])+rhs[16]*(2*params.Sigma[328])+rhs[17]*(2*params.Sigma[348])+rhs[18]*(2*params.Sigma[368])+rhs[19]*(2*params.Sigma[388]);
  lhs[9] = rhs[0]*(2*params.Sigma[9])+rhs[1]*(2*params.Sigma[29])+rhs[2]*(2*params.Sigma[49])+rhs[3]*(2*params.Sigma[69])+rhs[4]*(2*params.Sigma[89])+rhs[5]*(2*params.Sigma[109])+rhs[6]*(2*params.Sigma[129])+rhs[7]*(2*params.Sigma[149])+rhs[8]*(2*params.Sigma[169])+rhs[9]*(2*params.Sigma[189])+rhs[10]*(2*params.Sigma[209])+rhs[11]*(2*params.Sigma[229])+rhs[12]*(2*params.Sigma[249])+rhs[13]*(2*params.Sigma[269])+rhs[14]*(2*params.Sigma[289])+rhs[15]*(2*params.Sigma[309])+rhs[16]*(2*params.Sigma[329])+rhs[17]*(2*params.Sigma[349])+rhs[18]*(2*params.Sigma[369])+rhs[19]*(2*params.Sigma[389]);
  lhs[10] = rhs[0]*(2*params.Sigma[10])+rhs[1]*(2*params.Sigma[30])+rhs[2]*(2*params.Sigma[50])+rhs[3]*(2*params.Sigma[70])+rhs[4]*(2*params.Sigma[90])+rhs[5]*(2*params.Sigma[110])+rhs[6]*(2*params.Sigma[130])+rhs[7]*(2*params.Sigma[150])+rhs[8]*(2*params.Sigma[170])+rhs[9]*(2*params.Sigma[190])+rhs[10]*(2*params.Sigma[210])+rhs[11]*(2*params.Sigma[230])+rhs[12]*(2*params.Sigma[250])+rhs[13]*(2*params.Sigma[270])+rhs[14]*(2*params.Sigma[290])+rhs[15]*(2*params.Sigma[310])+rhs[16]*(2*params.Sigma[330])+rhs[17]*(2*params.Sigma[350])+rhs[18]*(2*params.Sigma[370])+rhs[19]*(2*params.Sigma[390]);
  lhs[11] = rhs[0]*(2*params.Sigma[11])+rhs[1]*(2*params.Sigma[31])+rhs[2]*(2*params.Sigma[51])+rhs[3]*(2*params.Sigma[71])+rhs[4]*(2*params.Sigma[91])+rhs[5]*(2*params.Sigma[111])+rhs[6]*(2*params.Sigma[131])+rhs[7]*(2*params.Sigma[151])+rhs[8]*(2*params.Sigma[171])+rhs[9]*(2*params.Sigma[191])+rhs[10]*(2*params.Sigma[211])+rhs[11]*(2*params.Sigma[231])+rhs[12]*(2*params.Sigma[251])+rhs[13]*(2*params.Sigma[271])+rhs[14]*(2*params.Sigma[291])+rhs[15]*(2*params.Sigma[311])+rhs[16]*(2*params.Sigma[331])+rhs[17]*(2*params.Sigma[351])+rhs[18]*(2*params.Sigma[371])+rhs[19]*(2*params.Sigma[391]);
  lhs[12] = rhs[0]*(2*params.Sigma[12])+rhs[1]*(2*params.Sigma[32])+rhs[2]*(2*params.Sigma[52])+rhs[3]*(2*params.Sigma[72])+rhs[4]*(2*params.Sigma[92])+rhs[5]*(2*params.Sigma[112])+rhs[6]*(2*params.Sigma[132])+rhs[7]*(2*params.Sigma[152])+rhs[8]*(2*params.Sigma[172])+rhs[9]*(2*params.Sigma[192])+rhs[10]*(2*params.Sigma[212])+rhs[11]*(2*params.Sigma[232])+rhs[12]*(2*params.Sigma[252])+rhs[13]*(2*params.Sigma[272])+rhs[14]*(2*params.Sigma[292])+rhs[15]*(2*params.Sigma[312])+rhs[16]*(2*params.Sigma[332])+rhs[17]*(2*params.Sigma[352])+rhs[18]*(2*params.Sigma[372])+rhs[19]*(2*params.Sigma[392]);
  lhs[13] = rhs[0]*(2*params.Sigma[13])+rhs[1]*(2*params.Sigma[33])+rhs[2]*(2*params.Sigma[53])+rhs[3]*(2*params.Sigma[73])+rhs[4]*(2*params.Sigma[93])+rhs[5]*(2*params.Sigma[113])+rhs[6]*(2*params.Sigma[133])+rhs[7]*(2*params.Sigma[153])+rhs[8]*(2*params.Sigma[173])+rhs[9]*(2*params.Sigma[193])+rhs[10]*(2*params.Sigma[213])+rhs[11]*(2*params.Sigma[233])+rhs[12]*(2*params.Sigma[253])+rhs[13]*(2*params.Sigma[273])+rhs[14]*(2*params.Sigma[293])+rhs[15]*(2*params.Sigma[313])+rhs[16]*(2*params.Sigma[333])+rhs[17]*(2*params.Sigma[353])+rhs[18]*(2*params.Sigma[373])+rhs[19]*(2*params.Sigma[393]);
  lhs[14] = rhs[0]*(2*params.Sigma[14])+rhs[1]*(2*params.Sigma[34])+rhs[2]*(2*params.Sigma[54])+rhs[3]*(2*params.Sigma[74])+rhs[4]*(2*params.Sigma[94])+rhs[5]*(2*params.Sigma[114])+rhs[6]*(2*params.Sigma[134])+rhs[7]*(2*params.Sigma[154])+rhs[8]*(2*params.Sigma[174])+rhs[9]*(2*params.Sigma[194])+rhs[10]*(2*params.Sigma[214])+rhs[11]*(2*params.Sigma[234])+rhs[12]*(2*params.Sigma[254])+rhs[13]*(2*params.Sigma[274])+rhs[14]*(2*params.Sigma[294])+rhs[15]*(2*params.Sigma[314])+rhs[16]*(2*params.Sigma[334])+rhs[17]*(2*params.Sigma[354])+rhs[18]*(2*params.Sigma[374])+rhs[19]*(2*params.Sigma[394]);
  lhs[15] = rhs[0]*(2*params.Sigma[15])+rhs[1]*(2*params.Sigma[35])+rhs[2]*(2*params.Sigma[55])+rhs[3]*(2*params.Sigma[75])+rhs[4]*(2*params.Sigma[95])+rhs[5]*(2*params.Sigma[115])+rhs[6]*(2*params.Sigma[135])+rhs[7]*(2*params.Sigma[155])+rhs[8]*(2*params.Sigma[175])+rhs[9]*(2*params.Sigma[195])+rhs[10]*(2*params.Sigma[215])+rhs[11]*(2*params.Sigma[235])+rhs[12]*(2*params.Sigma[255])+rhs[13]*(2*params.Sigma[275])+rhs[14]*(2*params.Sigma[295])+rhs[15]*(2*params.Sigma[315])+rhs[16]*(2*params.Sigma[335])+rhs[17]*(2*params.Sigma[355])+rhs[18]*(2*params.Sigma[375])+rhs[19]*(2*params.Sigma[395]);
  lhs[16] = rhs[0]*(2*params.Sigma[16])+rhs[1]*(2*params.Sigma[36])+rhs[2]*(2*params.Sigma[56])+rhs[3]*(2*params.Sigma[76])+rhs[4]*(2*params.Sigma[96])+rhs[5]*(2*params.Sigma[116])+rhs[6]*(2*params.Sigma[136])+rhs[7]*(2*params.Sigma[156])+rhs[8]*(2*params.Sigma[176])+rhs[9]*(2*params.Sigma[196])+rhs[10]*(2*params.Sigma[216])+rhs[11]*(2*params.Sigma[236])+rhs[12]*(2*params.Sigma[256])+rhs[13]*(2*params.Sigma[276])+rhs[14]*(2*params.Sigma[296])+rhs[15]*(2*params.Sigma[316])+rhs[16]*(2*params.Sigma[336])+rhs[17]*(2*params.Sigma[356])+rhs[18]*(2*params.Sigma[376])+rhs[19]*(2*params.Sigma[396]);
  lhs[17] = rhs[0]*(2*params.Sigma[17])+rhs[1]*(2*params.Sigma[37])+rhs[2]*(2*params.Sigma[57])+rhs[3]*(2*params.Sigma[77])+rhs[4]*(2*params.Sigma[97])+rhs[5]*(2*params.Sigma[117])+rhs[6]*(2*params.Sigma[137])+rhs[7]*(2*params.Sigma[157])+rhs[8]*(2*params.Sigma[177])+rhs[9]*(2*params.Sigma[197])+rhs[10]*(2*params.Sigma[217])+rhs[11]*(2*params.Sigma[237])+rhs[12]*(2*params.Sigma[257])+rhs[13]*(2*params.Sigma[277])+rhs[14]*(2*params.Sigma[297])+rhs[15]*(2*params.Sigma[317])+rhs[16]*(2*params.Sigma[337])+rhs[17]*(2*params.Sigma[357])+rhs[18]*(2*params.Sigma[377])+rhs[19]*(2*params.Sigma[397]);
  lhs[18] = rhs[0]*(2*params.Sigma[18])+rhs[1]*(2*params.Sigma[38])+rhs[2]*(2*params.Sigma[58])+rhs[3]*(2*params.Sigma[78])+rhs[4]*(2*params.Sigma[98])+rhs[5]*(2*params.Sigma[118])+rhs[6]*(2*params.Sigma[138])+rhs[7]*(2*params.Sigma[158])+rhs[8]*(2*params.Sigma[178])+rhs[9]*(2*params.Sigma[198])+rhs[10]*(2*params.Sigma[218])+rhs[11]*(2*params.Sigma[238])+rhs[12]*(2*params.Sigma[258])+rhs[13]*(2*params.Sigma[278])+rhs[14]*(2*params.Sigma[298])+rhs[15]*(2*params.Sigma[318])+rhs[16]*(2*params.Sigma[338])+rhs[17]*(2*params.Sigma[358])+rhs[18]*(2*params.Sigma[378])+rhs[19]*(2*params.Sigma[398]);
  lhs[19] = rhs[0]*(2*params.Sigma[19])+rhs[1]*(2*params.Sigma[39])+rhs[2]*(2*params.Sigma[59])+rhs[3]*(2*params.Sigma[79])+rhs[4]*(2*params.Sigma[99])+rhs[5]*(2*params.Sigma[119])+rhs[6]*(2*params.Sigma[139])+rhs[7]*(2*params.Sigma[159])+rhs[8]*(2*params.Sigma[179])+rhs[9]*(2*params.Sigma[199])+rhs[10]*(2*params.Sigma[219])+rhs[11]*(2*params.Sigma[239])+rhs[12]*(2*params.Sigma[259])+rhs[13]*(2*params.Sigma[279])+rhs[14]*(2*params.Sigma[299])+rhs[15]*(2*params.Sigma[319])+rhs[16]*(2*params.Sigma[339])+rhs[17]*(2*params.Sigma[359])+rhs[18]*(2*params.Sigma[379])+rhs[19]*(2*params.Sigma[399]);
}
void fillq(void) {
  work.q[0] = -params.lambda[0]*params.Returns[0];
  work.q[1] = -params.lambda[0]*params.Returns[1];
  work.q[2] = -params.lambda[0]*params.Returns[2];
  work.q[3] = -params.lambda[0]*params.Returns[3];
  work.q[4] = -params.lambda[0]*params.Returns[4];
  work.q[5] = -params.lambda[0]*params.Returns[5];
  work.q[6] = -params.lambda[0]*params.Returns[6];
  work.q[7] = -params.lambda[0]*params.Returns[7];
  work.q[8] = -params.lambda[0]*params.Returns[8];
  work.q[9] = -params.lambda[0]*params.Returns[9];
  work.q[10] = -params.lambda[0]*params.Returns[10];
  work.q[11] = -params.lambda[0]*params.Returns[11];
  work.q[12] = -params.lambda[0]*params.Returns[12];
  work.q[13] = -params.lambda[0]*params.Returns[13];
  work.q[14] = -params.lambda[0]*params.Returns[14];
  work.q[15] = -params.lambda[0]*params.Returns[15];
  work.q[16] = -params.lambda[0]*params.Returns[16];
  work.q[17] = -params.lambda[0]*params.Returns[17];
  work.q[18] = -params.lambda[0]*params.Returns[18];
  work.q[19] = -params.lambda[0]*params.Returns[19];
}
void fillh(void) {
  work.h[0] = 0;
  work.h[1] = 0;
  work.h[2] = 0;
  work.h[3] = 0;
  work.h[4] = 0;
  work.h[5] = 0;
  work.h[6] = 0;
  work.h[7] = 0;
  work.h[8] = 0;
  work.h[9] = 0;
  work.h[10] = 0;
  work.h[11] = 0;
  work.h[12] = 0;
  work.h[13] = 0;
  work.h[14] = 0;
  work.h[15] = 0;
  work.h[16] = 0;
  work.h[17] = 0;
  work.h[18] = 0;
  work.h[19] = 0;
}
void fillb(void) {
  work.b[0] = 1;
}
void pre_ops(void) {
}
