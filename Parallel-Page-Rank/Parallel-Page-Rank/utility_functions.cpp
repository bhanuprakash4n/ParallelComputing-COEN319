#include "utility_functions.h"


void test_matrix(SparseMatrix * mat) {
    uns32 nrows = mat->nrows;
    uns32 ncols = mat->ncols;
    assert(mat->ptrs);
    auto nnz = mat->ptrs[nrows];
    for(uns32 i = 0; i < nrows; ++i){
        assert(mat->ptrs[i] <= nnz);
    }
    for(uns32 j=0; j < nnz; ++j){
        assert(mat->inds[j] < ncols);
    }
}




int getToFrom(string line, uns32 &from, uns32 &to) {
    size_t splitIdx = line.find("\t");
    if (splitIdx == string::npos) return FAILURE;

    from = atoi(line.substr(0, splitIdx).c_str())-1;    // input is not zero-indexed
    to = atoi(line.substr(splitIdx).c_str())-1;

    return SUCCESS;
}



uns32 getNnzPerNode(NodeInfo *nodes, uns32 *oLinks, uns32 N, ifstream &fileS) {

    if (!fileS.is_open()) return 0;

    string currLine, from, to;
    uns32 fromNode, toNode, total;
    total = 0;
    while (fileS.good()) {
        // if (total % 1000000 == 0) {
        //     cout << "Line " << total << endl;
        // }

        fromNode = 0;
        toNode = 0;
        getline(fileS, currLine);

        // get from and to indices
        if (getToFrom(currLine, fromNode, toNode) == FAILURE || 
            fromNode >= N || toNode >= N) continue;

        // set NodeInfo[to]
        ++total;
        nodes[toNode].push_back(fromNode);
        ++oLinks[fromNode];
    }
    return total;
}





#ifdef OPEN_MP_PROJECT

void createNodeMatrix(SparseMatrix *adjM, NodeInfo *nodes, uns32 *oLinks, uns32 N) {
    // master thread to figure out how to split up C based on nnzPerRow
    for (uns32 z = 1; z <= N; ++z) {
        adjM->ptrs[z] = adjM->ptrs[z-1] + nodes[z-1].size();
    }


    #pragma omp parallel for schedule(dynamic)
    for (uns32 i = 0; i < N; ++i) {
        uns32 currIdx = 0;
        for (uns32 j = adjM->ptrs[i]; j < adjM->ptrs[i+1]; ++j) {
            uns32 colID = nodes[i][currIdx++];
            adjM->inds[j] = colID;
            adjM->vals[j] = 1.0 / oLinks[colID];
        }
    }

}

void computeInverseIndex(SparseMatrix *A) {
   val_t* oldVal = A->vals;
   idx_t* oldInd = A->inds;
   ptr_t* oldPtr = A->ptrs;
  
   A->vals = (val_t*)malloc(oldPtr[A->nrows]*sizeof(val_t));
   A->inds = (idx_t*)malloc(oldPtr[A->nrows]*sizeof(idx_t));
   A->ptrs = (ptr_t*)calloc(A->ncols+1, sizeof(ptr_t));
  
   for(idx_t i=0; i<oldPtr[A->nrows]; ++(A->ptrs[oldInd[i]]), ++i);
   for(idx_t i=1; i<=A->ncols; ++i)
       A->ptrs[i] += A->ptrs[i-1];
   for(idx_t i=0; i<A->nrows; ++i) {
       for(idx_t j=oldPtr[i]; j<oldPtr[i+1]; ++j) {
           A->vals[--(A->ptrs[oldInd[j]])] = oldVal[j];
           A->inds[A->ptrs[oldInd[j]]] = i;
       }
   }
   int temp = A->nrows;
   A->nrows = A->ncols;
   A->ncols = temp;
   if(oldInd)
       free(oldInd);
   if(oldVal)
       free(oldVal);
}

void initializePageRankVector(flt32 *pgRnkV, uns32 N) {
    flt32 baseValue = 1.0 / N;

    #pragma omp parallel for schedule(dynamic)
    for (uns32 i = 0; i < N; ++i)
        pgRnkV[i] = baseValue;
}

void calculatePageRank(SparseMatrix *adjM, flt32 *initPgRnkV, flt32 *finPgRnkV, uns32 N) {
    flt32 min_error = 0.0001;
    flt32 cur_error = min_error;
    for(int i =0; i < 10000; i++){      
        SparseDenseMatMult(adjM,initPgRnkV,finPgRnkV,N);
        cur_error = matirxErrorandCopyV(initPgRnkV, finPgRnkV, N);
        if (i % 100 == 0)
            cout<<"i:"<<i<<" Error: "<<cur_error<<endl;
        if (cur_error < min_error){
            cout<<"i:"<<i<<" Final Error: "<< cur_error <<endl;
            break;
        }
    }  
}

void calculatePageRank2(SparseMatrix *adjM, flt32 *initPgRnkV, flt32 *finPgRnkV, uns32 N) {
    flt32 min_error = 0.0001;
    flt32 cur_error = min_error;
    for(int i =0; i < 10000; i++){      
        SparseDenseMatMult(adjM,initPgRnkV,finPgRnkV,N);
        cur_error = matirxErrorandCopyV(initPgRnkV, finPgRnkV, N);
        cout<<"i:"<<i<<" Error: "<<cur_error<<endl;
        if (cur_error < min_error){
            cout<<"i:"<<i<<" Final Error: "<< cur_error <<endl;
            break;
        }
        SparseMatMult(adjM);
        cout << "SparseMatMult done, nnz = " << adjM->ptrs[adjM->nrows] << endl;
    }  
}

#elif defined(OPEN_ACC_PROJECT)

void createNodeMatrix(SparseMatrix *adjM, NodeInfo *nodes, uns32 *oLinks, uns32 N) {
    // master thread to figure out how to split up C based on nnzPerRow
    for (uns32 z = 1; z <= N; ++z) {
        adjM->ptrs[z] = adjM->ptrs[z-1] + nodes[z-1].size();
    }

    #pragma acc kernels loop copyin(nodes, oLinks, N) copyout(adjM)
    for (uns32 i = 0; i < N; ++i) {
        uns32 currIdx = 0;
        for (uns32 j = adjM->ptrs[i]; j < adjM->ptrs[i+1]; ++j) {
            uns32 colID = nodes[i][currIdx++];
            adjM->inds[j] = colID;
            adjM->vals[j] = 1.0 / oLinks[colID];
        }
    }
}

void initializePageRankVector(flt32 *pgRnkV, uns32 N) {
    flt32 baseValue = 1.0 / N;

    #pragma acc kernels loop copyin(baseValue) copyout(pgRnkV)
    for (uns32 i = 0; i < N; ++i)
        pgRnkV[i] = baseValue;
}

void calculatePageRank(SparseMatrix *adjM, flt32 *initPgRnkV, flt32 *finPgRnkV, uns32 N) {

    flt32 min_error = 0.0001;
    flt32 cur_error = min_error;

    #pragma acc data copy(adjM, initPgRnkV, finPgRnkV)
    for(int i =0; i < 10000; i++){

        #pragma acc parallel
        {
            #pragma acc loop
            for(uns32 i = 0; i < N; i++){
                //Idx V
                for(uns32 j = adjM->ptrs[i]; j < adjM->ptrs[i+1]; j++){
                    finPgRnkV[i] += adjM->vals[j] *initPgRnkV[adjM->inds[j]];
                }       
            }

            //Compute Absolute Error, 
            //Set initPgRnkV to finPgRnkV, 
            //and finPgRnkV to zero vector
            cur_error = 0;
            #pragma acc loop reduction(+:cur_error)
            for(uns32 idx = 0; idx < N; idx++){
                cur_error += abs(finPgRnkV[idx] - initPgRnkV[idx]);
                initPgRnkV[idx] = finPgRnkV[idx];
                finPgRnkV[idx] = 0;
            }
            
            if (i % 10 == 0)
                //cout<<"i:"<<i<<" Error: "<<cur_error<<endl;
            if (cur_error < min_error){
                //cout<<"i:"<<i<<" Final Error: "<< cur_error <<endl;
                i = 10000;
            }
        }
    }  
    
    



}

#else

void createNodeMatrix(SparseMatrix *adjM, NodeInfo *nodes, uns32 *oLinks, uns32 N) {
    uns32 currNnz = 0;
    for (uns32 i = 0; i < N; ++i) {
        adjM->ptrs[i] = currNnz;
        uns32 nnz = nodes[i].size();
        for (uns32 j = 0; j < nnz; ++j) {
            uns32 colID = nodes[i][j];
            adjM->inds[currNnz] = colID;
            adjM->vals[currNnz++] = 1.0 / oLinks[colID];
        }
    }
    adjM->ptrs[N] = currNnz;
}

void initializePageRankVector(flt32 *pgRnkV, uns32 N) {
    flt32 baseValue = 1.0 / N;
    for (uns32 i = 0; i < N; ++i)
        pgRnkV[i] = baseValue;
}

void calculatePageRank(SparseMatrix *adjM, flt32 *initPgRnkV, flt32 *finPgRnkV, uns32 N) {
    
    SparseDenseMatMult(adjM,initPgRnkV,finPgRnkV,N);



}


#endif

