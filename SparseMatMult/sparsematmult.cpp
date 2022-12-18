#include <iostream>
#include <omp.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <cstring>      /* strcasecmp */
#include <cstdint>
#include <assert.h>
#include <vector>       // std::vector
#include <algorithm>    // std::random_shuffle
#include <random>
#include <stdexcept>
#include <iomanip>

using namespace std;

using idx_t = std::uint32_t;
using val_t = float;
using ptr_t = std::uintptr_t;

/**
 * CSR structure to store search results
 */
typedef struct csr_t {
  idx_t nrows; // number of rows
  idx_t ncols; // number of rows
  idx_t * ind; // column ids
  val_t * val; // values
  ptr_t * ptr; // pointers (start of row in ind/val)

  csr_t()
  {
    nrows = ncols = 0;
    ind = nullptr;
    val = nullptr;
    ptr = nullptr;
  }

  /**
   * Reserve space for more rows or non-zeros. Structure may only grow, not shrink.
   * @param nrows Number of rows
   * @param nnz   Number of non-zeros
   */
  void reserve(const idx_t nrows, const ptr_t nnz)
  {
    if(nrows > this->nrows){
      if(ptr){
        ptr = (ptr_t*) realloc(ptr, sizeof(ptr_t) * (nrows+1));
      } else {
        ptr = (ptr_t*) malloc(sizeof(ptr_t) * (nrows+1));
        ptr[0] = 0;
      }
      if(!ptr){
        throw std::runtime_error("Could not allocate ptr array.");
      }
    }
    if(nnz > ptr[this->nrows]){
      if(ind){
        ind = (idx_t*) realloc(ind, sizeof(idx_t) * nnz);
      } else {
        ind = (idx_t*) malloc(sizeof(idx_t) * nnz);
      }
      if(!ind){
        throw std::runtime_error("Could not allocate ind array.");
      }
      if(val){
        val = (val_t*) realloc(val, sizeof(val_t) * nnz);
      } else {
        val = (val_t*) malloc(sizeof(val_t) * nnz);
      }
      if(!val){
        throw std::runtime_error("Could not allocate val array.");
      }
    }
    this->nrows = nrows;
  }

  /**
   * Reserve space for more rows or non-zeros. Structure may only grow, not shrink.
   * @param nrows Number of rows
   * @param ncols Number of columns
   * @param factor   Sparsity factor
   */
  static csr_t * random(const idx_t nrows, const idx_t ncols, const double factor)
  {
    ptr_t nnz = (ptr_t) (factor * nrows * ncols);
    if(nnz >= nrows * ncols / 2.0){
      throw std::runtime_error("Asking for too many non-zeros. Matrix is not sparse.");
    }
    auto mat = new csr_t();
    mat->reserve(nrows, nnz);
    mat->ncols = ncols;

    /* fill in ptr array; generate random row sizes */
    unsigned int seed = (unsigned long) mat;
    long double sum = 0;
    for(idx_t i=1; i <= mat->nrows; ++i){
      mat->ptr[i] = rand_r(&seed) % ncols;
      sum += mat->ptr[i];
    }
    for(idx_t i=0; i < mat->nrows; ++i){
      double percent = mat->ptr[i+1] / sum;
      mat->ptr[i+1] = mat->ptr[i] + (ptr_t)(percent * nnz);
      if(mat->ptr[i+1] > nnz){
        mat->ptr[i+1] = nnz;
      }
    }
    if(nnz - mat->ptr[mat->nrows-1] <= ncols){
      mat->ptr[mat->nrows] = nnz;
    }

    /* fill in indices and values with random numbers */
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      unsigned int seed = (unsigned long) mat * (1+tid);
      std::vector<int> perm;
      for(idx_t i=0; i < ncols; ++i){
        perm.push_back(i);
      }
      std::random_device seeder;
      std::mt19937 engine(seeder());

      #pragma omp for
      for(idx_t i=0; i < nrows; ++i){
        std::shuffle(perm.begin(), perm.end(), engine);
        for(ptr_t j=mat->ptr[i]; j < mat->ptr[i+1]; ++j){
          mat->ind[j] = perm[j - mat->ptr[i]];
          mat->val[j] = ((double) rand_r(&seed)/rand_r(&seed));
        }
      }
    }

    return mat;
  }

  string info(const string name="") const
  {
    return (name.empty() ? "CSR" : name) + "<" + to_string(nrows) + ", " + to_string(ncols) + ", " +
      (ptr ? to_string(ptr[nrows]) : "0") + ">";
  }

  void print(char name) const
  { 
    if(nrows*ncols > 100)
      return;
    cout << "Print matrix " << name << "? (y/n): ";
    char ch;
    cin >> ch;
    if(ch != 'y' && ch != 'Y')
      return;
    for(ptr_t i=0; i<ptr[nrows]; ++i)
        cout << std::fixed << std::setprecision(2) << val[i] << " ";
    cout << endl;

    for(ptr_t i=0; i<ptr[nrows]; ++i)
        cout << std::fixed << std::setprecision(2) << ind[i] << " ";
    cout << endl;

    for(idx_t i=0; i<=nrows; ++i)
        cout << ptr[i] << " ";
    cout << endl << endl;

    val_t tempMatrix[nrows][ncols];
    memset(tempMatrix, 0, nrows*ncols*sizeof(val_t));
    for(idx_t i=0; i<nrows; ++i)
      for(idx_t j=ptr[i]; j<ptr[i+1]; ++j)
        tempMatrix[i][ind[j]] = val[j];
    
    for(idx_t i=0; i<nrows; ++i) {
      for(idx_t j=0; j<ncols; ++j)
        cout << tempMatrix[i][j] << " ";
      cout << endl;
    }
    cout << endl;
  }

  ~csr_t()
  {
    if(ind){
      free(ind);
    }
    if(val){
      free(val);
    }
    if(ptr){
      free(ptr);
    }
  }
} csr_t;

/**
 * Ensure the matrix is valid
 * @param mat Matrix to test
 */
void test_matrix(csr_t * mat){
  auto nrows = mat->nrows;
  auto ncols = mat->ncols;
  assert(mat->ptr);
  auto nnz = mat->ptr[nrows];
  for(idx_t i=0; i < nrows; ++i){
    assert(mat->ptr[i] <= nnz);
  }
  for(ptr_t j=0; j < nnz; ++j){
    assert(mat->ind[j] < ncols);
  }
}

struct accumulatorPerThread {
  val_t* accum;
  idx_t* index;
  idx_t indexFilled;
  accumulatorPerThread() : accum(nullptr), index(nullptr), indexFilled(0) {}
};

struct ansPerRow {
  val_t* val;
  idx_t* idx;
  idx_t size;
  ansPerRow() : val(nullptr), idx(nullptr), size(0) {}
};

void computeInverseIndex(csr_t *A) {
  val_t* oldVal = A->val;
  idx_t* oldInd = A->ind;
  ptr_t* oldPtr = A->ptr;

  A->val = (val_t*)malloc(oldPtr[A->nrows]*sizeof(val_t));
  A->ind = (idx_t*)malloc(oldPtr[A->nrows]*sizeof(idx_t));
  A->ptr = (ptr_t*)calloc(A->ncols+1, sizeof(ptr_t));

  for(idx_t i=0; i<oldPtr[A->nrows]; ++(A->ptr[oldInd[i]]), ++i);
  for(idx_t i=1; i<=A->ncols; ++i)
    A->ptr[i] += A->ptr[i-1];
  for(idx_t i=0; i<A->nrows; ++i) {
    for(idx_t j=oldPtr[i]; j<oldPtr[i+1]; ++j) {
      A->val[--(A->ptr[oldInd[j]])] = oldVal[j];
      A->ind[A->ptr[oldInd[j]]] = i;
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

/**
 * Multiply A and B (transposed given) and write output in C.
 * Note that C has no data allocations (i.e., ptr, ind, and val pointers are null).
 * Use `csr_t::reserve` to increase C's allocations as necessary.
 * @param A  Matrix A.
 * @param B The transpose of matrix B.
 * @param C  Output matrix
 */
void sparsematmult(csr_t * A, csr_t * B, csr_t *C)
{
  computeInverseIndex(B);
  idx_t numThreads = omp_get_max_threads(); 
  accumulatorPerThread accumulator[numThreads];
  for(idx_t i=0; i<numThreads; ++i) {
    accumulator[i].accum = (val_t*)calloc(B->ncols, sizeof(val_t));
    accumulator[i].index = (idx_t*)calloc(B->ncols, sizeof(idx_t));
  }
  ansPerRow rowAns[A->nrows];
  #pragma omp parallel for schedule(dynamic)
  for(idx_t i=0; i<A->nrows; ++i) {
    int tid = omp_get_thread_num();
    for(idx_t j=A->ptr[i]; j<A->ptr[i+1]; ++j) {
      for(idx_t k=B->ptr[A->ind[j]]; k<B->ptr[A->ind[j]+1]; ++k) {
        if(accumulator[tid].accum[B->ind[k]] == 0.0) {
          accumulator[tid].index[accumulator[tid].indexFilled] = B->ind[k];
          (accumulator[tid].indexFilled)++;
        }
        accumulator[tid].accum[B->ind[k]] += A->val[j]*B->val[k];
      }
    }
    rowAns[i].size = accumulator[tid].indexFilled;
    if(rowAns[i].size != 0) {
      rowAns[i].val = (val_t*)malloc((rowAns[i].size)*sizeof(val_t));
      rowAns[i].idx = (idx_t*)malloc((rowAns[i].size)*sizeof(idx_t));
    }
    for(idx_t j=0; j<rowAns[i].size; ++j) {
      rowAns[i].val[j] = accumulator[tid].accum[accumulator[tid].index[j]];
      accumulator[tid].accum[accumulator[tid].index[j]] = 0.0;
      rowAns[i].idx[j] = accumulator[tid].index[j];
      accumulator[tid].index[j] = 0;
    }
    accumulator[tid].indexFilled = 0;
  }

  C->reserve(A->nrows, 0);
  C->ncols = B->ncols;

  for(idx_t i=1; i<C->nrows; ++i)
    C->ptr[i] = C->ptr[i-1]+rowAns[i-1].size;
  C->ptr[C->nrows] = 0;
  C->reserve(A->nrows, C->ptr[C->nrows-1]+rowAns[C->nrows-1].size);
  C->ptr[C->nrows] = C->ptr[C->nrows-1]+rowAns[C->nrows-1].size;

  #pragma omp parallel for schedule(dynamic)
  for(idx_t i=0; i<C->nrows; ++i) {
    for(idx_t j=0; j<rowAns[i].size; j++) {
      C->val[C->ptr[i]+j] = rowAns[i].val[j];
      C->ind[C->ptr[i]+j] = rowAns[i].idx[j];
    }
    if(rowAns[i].val)
      free(rowAns[i].val);
    rowAns[i].val = nullptr;
    if(rowAns[i].idx)
      free(rowAns[i].idx);
    rowAns[i].idx = nullptr;
  }
}


void myPrint(csr_t * A, csr_t * B, csr_t *C) {
  A->print('A');
  cout << endl;
  B->print('B');
  cout << endl;
  C->print('C');
  cout << endl;
}

int main(int argc, char *argv[])
{
  if(argc < 4){
    cerr << "Invalid options." << endl << "<program> <A_nrows> <A_ncols> <B_ncols> <fill_factor> [-t <num_threads>]" << endl;
    exit(1);
  }
  int nrows = atoi(argv[1]);
  int ncols = atoi(argv[2]);
  int ncols2 = atoi(argv[3]);
  double factor = atof(argv[4]);
  int nthreads = 1;
  if(argc == 7 && strcasecmp(argv[5], "-t") == 0){
    nthreads = atoi(argv[6]);
    omp_set_num_threads(nthreads);
  }
  cout << "A_nrows: " << nrows << endl;
  cout << "A_ncols: " << ncols << endl;
  cout << "B_ncols: " << ncols2 << endl;
  cout << "factor: " << factor << endl;
  cout << "nthreads: " << nthreads << endl;

  /* initialize random seed: */
  srand (time(NULL));

  auto A = csr_t::random(nrows, ncols, factor);
  auto B = csr_t::random(ncols2, ncols, factor); // Note B is already transposed.
  test_matrix(A);
  test_matrix(B);
  auto C = new csr_t(); // Note that C has no data allocations so far.

  cout << A->info("A") << endl;
  cout << B->info("B") << endl;

  auto t1 = omp_get_wtime();
  sparsematmult(A, B, C);
  auto t2 = omp_get_wtime();
  test_matrix(B);

  cout << C->info("C") << endl;
  test_matrix(C);

  cout << "Execution time: " << (t2-t1) << endl;

  // myPrint(A, B, C);

  delete A;
  delete B;
  delete C;

  return 0;
}
