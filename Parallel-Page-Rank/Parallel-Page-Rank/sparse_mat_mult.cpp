#include "sparse_mat_mult.h"



#ifdef OPEN_MP_PROJECT

void SparseDenseMatMult(SparseMatrix *adjM, flt32 *initPgRnkV, flt32 *finPgRnkV, uns32 N)
{
    
    //Idx M: Loop throgh adjM->ptrs array
    #pragma omp parallel for
    for(uns32 i = 0; i < N; i++){
        //Idx V
        for(uns32 j = adjM->ptrs[i]; j < adjM->ptrs[i+1]; j++){
            finPgRnkV[i] += adjM->vals[j] *initPgRnkV[adjM->inds[j]];
        }       
    }
}


void SparseMatMult(SparseMatrix * M) {
    SparseMatrix * A = M;
    SparseMatrix * B = M;
    idx_t numThreads = omp_get_max_threads();
    idx_t internalNumThreads = 2;
    accumulatorPerThread accumulator[numThreads];
    for(idx_t i=0; i<numThreads; ++i) {
        accumulator[i].accum = (val_t*)calloc(B->ncols, sizeof(val_t));
        accumulator[i].index = (idx_t*)calloc(B->ncols, sizeof(idx_t));
    }
    ansPerRow* rowAns = (ansPerRow*) calloc(A->nrows, sizeof(ansPerRow));
    #pragma omp parallel for schedule(dynamic)
    for(idx_t i=0; i<A->nrows; ++i) {
        int tid = omp_get_thread_num();
        #pragma omp parallel for num_threads(internalNumThreads)
        for(idx_t j=A->ptrs[i]; j<A->ptrs[i+1]; ++j) {
            for(idx_t k=B->ptrs[A->inds[j]]; k<B->ptrs[(A->inds[j])+1]; ++k) {
                float cur_val = A->vals[j]*B->vals[k];
                if(accumulator[tid].accum[B->inds[k]] == 0.0 && cur_val != 0.0) {
                    accumulator[tid].index[accumulator[tid].indexFilled] = B->inds[k];
                    (accumulator[tid].indexFilled)++;
                }
                accumulator[tid].accum[B->inds[k]] += cur_val;
            }
        }
        rowAns[i].size = accumulator[tid].indexFilled;
        if(rowAns[i].size != 0) {
            rowAns[i].val = (val_t*)malloc((rowAns[i].size)*sizeof(val_t));
            rowAns[i].idx = (idx_t*)malloc((rowAns[i].size)*sizeof(idx_t));
        }
        #pragma omp parallel for num_threads(internalNumThreads)
        for(idx_t j=0; j<rowAns[i].size; ++j) {
            rowAns[i].val[j] = accumulator[tid].accum[accumulator[tid].index[j]];
            accumulator[tid].accum[accumulator[tid].index[j]] = 0.0;
            rowAns[i].idx[j] = accumulator[tid].index[j];
            accumulator[tid].index[j] = 0;
        }
        accumulator[tid].indexFilled = 0;
    }
    #pragma omp barrier
    
    for(idx_t i=1; i<M->nrows; ++i)
        M->ptrs[i] = M->ptrs[i-1]+rowAns[i-1].size;
    M->reserve(A->nrows, B->ncols, M->ptrs[M->nrows-1]+rowAns[M->nrows-1].size);
    M->ptrs[M->nrows] = M->ptrs[M->nrows-1]+rowAns[M->nrows-1].size;

    #pragma omp parallel for schedule(dynamic)
    for(idx_t i=0; i<M->nrows; ++i) {
        for(idx_t j=0; j<rowAns[i].size; j++) {
            M->vals[M->ptrs[i]+j] = rowAns[i].val[j];
            M->inds[M->ptrs[i]+j] = rowAns[i].idx[j];
        }
        if(rowAns[i].val)
            free(rowAns[i].val);
        rowAns[i].val = nullptr;
        if(rowAns[i].idx)
            free(rowAns[i].idx);
        rowAns[i].idx = nullptr;
    }
    #pragma omp barrier
    if(rowAns)
        free(rowAns);
}


flt32 matirxErrorandCopyV(flt32 *initPgRnkV, flt32 *finPgRnkV, uns32 N){

    flt32 total_error = 0;
    #pragma omp parallel for reduction(+:total_error)
    for(uns32 idx = 0; idx < N; idx++){
        total_error += abs(finPgRnkV[idx] - initPgRnkV[idx]);
        initPgRnkV[idx] = finPgRnkV[idx];
        finPgRnkV[idx] = 0;
    }
    return total_error;

}

#elif defined(OPEN_ACC_PROJECT)

void SparseDenseMatMult(SparseMatrix *adjM, flt32 *initPgRnkV, flt32 *finPgRnkV, uns32 N)
{
    //Idx M: Loop throgh adjM->ptrs array
    #pragma acc kernels loop copy(adjM, initPgRnkV, finPgRnkV)
    for(uns32 i = 0; i < N; i++){
        //Idx V
        for(uns32 j = adjM->ptrs[i]; j < adjM->ptrs[i+1]; j++){
            finPgRnkV[i] += adjM->vals[j] *initPgRnkV[adjM->inds[j]];
        }        
    }   
}

#else
// single threaded

void SparseDenseMatMult(SparseMatrix *adjM, flt32 *initPgRnkV, flt32 *finPgRnkV, uns32 N)
{
    
    //Idx M: Loop throgh adjM->ptrs array
    for(uns32 i = 0; i < N; i++){
        //Idx V
        for(uns32 j = adjM->ptrs[i]; j < adjM->ptrs[i+1]; j++){
            finPgRnkV[i] += adjM->vals[j] *initPgRnkV[adjM->inds[j]];
        }        
    }
}


#endif
