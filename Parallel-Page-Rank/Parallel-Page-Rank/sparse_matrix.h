#ifndef SPARSE_MATRIX
#define SPARSE_MATRIX

#include <cerrno>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <string.h>

#include "global_types.h"

using namespace std;

using idx_t = uns32;
using val_t = flt32;
using ptr_t = uns32;
 
struct accumulatorPerThread {
    val_t* accum;
    idx_t* index;
    idx_t indexFilled;
    accumulatorPerThread() : accum(nullptr), index(nullptr), indexFilled(0) {}
    ~accumulatorPerThread() {
        if(accum)
            free(accum);
        if(index)
            free(index);
        indexFilled = 0;
        accum = nullptr;
        index = nullptr;
    }
};

struct ansPerRow {
    val_t* val;
    idx_t* idx;
    idx_t size;
    ansPerRow() : val(nullptr), idx(nullptr), size(0) {}
    ~ansPerRow() {
        if(val)
            free(val);
        if(idx)
            free(idx);
        size = 0;
        val = nullptr;
        idx = nullptr;
    }
};

class SparseMatrix {
private:

public:
    uns32 nrows;    // number of rows
    uns32 ncols;    // number of columns
    uns32 * inds;   // column indices
    flt32 * vals;   // values
    uns32 * ptrs;   // pointers (start of row in ind/val)

    SparseMatrix(){
        nrows = ncols = 0;
        inds = NULL;
        vals = NULL;
        ptrs = NULL;
    }
    
    void allocate(const uns32 r, const uns32 c); // full size for now
    void reserve(const uns32 r, const uns32 c, const uns32 nnz);
    string info(const string name="") const
    {
        return (name.empty() ? "CSR" : name) + "<" + to_string(nrows) + ", " + to_string(ncols) + ", " +
            (ptrs ? to_string(ptrs[nrows]) : "0") + ">";
    }
    void printMatrix(uns32 rows=0);
    void tearDown() {
        if (inds) free(inds);
        if (vals) free(vals);
        if (ptrs) free(ptrs);
    }

};


#endif

