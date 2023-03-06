#include "sparse_matrix.h"


void SparseMatrix::reserve(const uns32 r, const uns32 c, const uns32 nnz) {
    ncols = c;
    if(r != nrows){
        if(ptrs){
            ptrs = (uns32*) realloc(ptrs, sizeof(uns32) * (r+1));
        } else {
            ptrs = (uns32*) malloc(sizeof(uns32) * (r+1));
            ptrs[0] = 0;
        }
        if(!ptrs){
            throw std::runtime_error("Could not allocate ptrs array.");
        }
        nrows = r;
    }
    if(nnz > ptrs[nrows]){
        if(inds){
            inds = (uns32*) realloc(inds, sizeof(uns32) * nnz);
        } else {
            inds = (uns32*) malloc(sizeof(uns32) * nnz);
        }
        if(!inds){
            throw std::runtime_error("Could not allocate inds array.");
        }
        if(vals){
            vals = (flt32*) realloc(vals, sizeof(flt32) * nnz);
        } else {
            vals = (flt32*) malloc(sizeof(flt32) * nnz);
        }
        if(!vals){
            throw std::runtime_error("Could not allocate vals array.");
        }
    }
}

void SparseMatrix::allocate(const uns32 r, const uns32 c) {
    reserve(r, c, r*c);    // filled matrix - r*c non-zeros
}


void SparseMatrix::printMatrix(uns32 rows) {
    if (!rows) rows = nrows;
    printf("=====================================\n");
    printf("Row Pointers: \n");
    printf("  [ ");
    for (uns32 i = 0; i < rows+1; ++i) {
        printf("%ld ", ptrs[i]);
    }
    printf("]\n");

    printf("Column Indices: \n");
    printf("  [ ");
    for (uns32 i = 0; i < ptrs[rows]; ++i) {
        printf("%ld ", inds[i]);
    }
    printf("]\n");

    printf("Values: \n");
    printf("  [ ");
    for (uns32 i = 0; i < ptrs[rows]; ++i) {
        printf("%f ", vals[i]);
    }
    printf("]\n");

    // flt32 *currRow = (flt32* ) malloc(sizeof(flt32)*ncols);
    // printf("Dense Matrix Format:\n");
    // for (uns32 i = 0; i < nrows; ++i) {
        
    //     memset(currRow, 0x0, sizeof(flt32)*ncols);

    //     for (uns32 j = ptrs[i]; j < ptrs[i+1]; ++j) {
    //         currRow[inds[j]] = vals[j];
    //     }
    //     printf("   ");
    //     for (uns32 k = 0; k < ncols; ++k) {
    //         printf("%f ", currRow[k]);
    //     }
    //     printf("\n");
    // }
    // free(currRow);

    printf("=====================================\n");
}