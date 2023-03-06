#ifndef SPARSE_MAT_MULT
#define SPARSE_MAT_MULT

#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>


#include "sparse_matrix.h"

#ifdef OPEN_MP_PROJECT
#include "omp.h"

#elif defined(OPEN_ACC_PROJECT)

#else
// single threaded
#endif


void SparseDenseMatMult(SparseMatrix *adjM, flt32 *initPgRnkV, flt32 *finPgRnkV, uns32 N);
void SparseMatMult(SparseMatrix *M);
flt32 matirxErrorandCopyV(flt32 *initPgRnkV, flt32 *finPgRnkV, uns32 N);


#endif