/* assert */
#include <assert.h>

/* errno */
#include <errno.h>

/* fopen, fscanf, fprintf, fclose */
#include <stdio.h>

/* EXIT_SUCCESS, EXIT_FAILURE, malloc, free */
#include <stdlib.h>

#include <time.h>

#include <omp.h>

int nthreads = 16;

static int create_mat(size_t const nrows, size_t const ncols, double ** const matp)
{
    double * mat=NULL;
    if (!(mat = (double*) malloc(nrows*ncols*sizeof(*mat)))) {
        goto cleanup;
    }

    /** Initialize matrix with random values **/
    for(size_t i = 0; i < nrows; i++){
        for (size_t j = 0; j < ncols; j++){
            mat[(i * ncols) + j] = (double)(rand() % 1000) / 353.0;
        }
    }
    /** End random initialization **/

    *matp = mat;

    return 0;

    cleanup:
    free(mat);
    return -1;
}

static int mult_mat_collapse(size_t const n, size_t const m, size_t const p,
                    double const * const A, double const * const B,
                    double ** const Cp)
{
  size_t i, j, k;
  double * C = NULL;

  if (!(C = (double*) malloc(n*p*sizeof(*C)))) {
    goto cleanup;
  }
#pragma omp parallel for collapse(2) schedule(static) default(none) shared(C) private(i,j,k) num_threads(nthreads)
  for (i=0; i<n; ++i) {
    for (j=0; j<p; ++j) {
      C[i*p+j] = 0.0;
      for (k=0; k<m; ++k) {
        C[i*p+j] += A[i*m+k] * B[k*p+j];
      }
    }
  }

  *Cp = C;

  return 0;

  cleanup:
  free(C);

  /*failure:*/
  return -1;
}

static int mult_mat_tiling(size_t const n, size_t const m, size_t const p,
                    double const * const A, double const * const B,
                    double ** const Cp)
{
    size_t i, j, k;
    double * C = NULL;
    const size_t b = 25;

    if (!(C = (double*) malloc(n*p*sizeof(*C)))) {
        goto cleanup;
    }

    #pragma omp parallel for collapse(2)
    for(size_t i=0; i<n; i++) {
        for(size_t j=0; j<p;j++) {
            C[i*p+j] = 0.0;
        }
    }

    #pragma omp parallel for collapse(2) schedule(static) default(none) shared(C) private(i,j,k) num_threads(nthreads)
    for (i=0; i<n/b; ++i) {
        for (j=0; j<p/b; ++j) {            
            for (k=0; k<m/b; ++k) {
                for(size_t x=0;x<b;x++) {
                    for(size_t y=0;y<b;y++) {
                        for(size_t z=0;z<b;z++) {
                            C[(i*b+x)*p +j*b+y] += A[(i*b+x)*m + k*b+z] * B[(k*b+z)*p + j*b+y];
                        }
                    }
                }
            }
        }
    }

    *Cp = C;

    return 0;

    cleanup:
    free(C);

    /*failure:*/
    return -1;
}

static bool verify_mat(size_t const n, size_t const m,
                        double const * const A, double const * const B) {
    for(size_t i=0; i<n; i++) {
        for(size_t j=0; j<m;j++) {
            if(A[i*m+j] != B[i*m+j])
                return false;
        }
    }
    return true;
}

int main(int argc, char * argv[])
{
  // size_t stored an unsigned integer
  size_t nrows, ncols, ncols2;
  double * A=NULL, * B=NULL, * C_collapse=NULL, * C_tiling=NULL;
  double time_taken, start, end;

  if (argc < 4) {
    fprintf(stderr, "usage: matmult nrows ncols ncols2 nthreads\n");
    goto failure;
  }

  nrows = atoi(argv[1]);
  ncols = atoi(argv[2]);
  ncols2 = atoi(argv[3]);
  if(argc == 5) {
    nthreads = atoi(argv[4]);
  }

  if (create_mat(nrows, ncols, &A)) {
    perror("error");
    goto failure;
  }

  if (create_mat(ncols, ncols2, &B)) {
    perror("error");
    goto failure;
  }

  start = omp_get_wtime();
  if (mult_mat_tiling(nrows, ncols, ncols2, A, B, &C_tiling)) {
    perror("error");
    goto failure;
  }
  end = omp_get_wtime();
  time_taken = end - start;
  printf("Time taken with tiling is : %lf\n", time_taken);

  start = omp_get_wtime();
  if (mult_mat_collapse(nrows, ncols, ncols2, A, B, &C_collapse)) {
    perror("error");
    goto failure;
  }
  end = omp_get_wtime();
  time_taken = end - start;
  printf("Time taken with collapse is : %lf \n", time_taken);

  if(verify_mat(nrows, ncols2, C_collapse, C_tiling)) {
    printf("Matrix Multiply correct\n");
  } else {
    printf("ERROR: Matrix Multiply incorrect\n");
  }

  free(A);
  free(B);
  free(C_collapse);
  free(C_tiling);

  return EXIT_SUCCESS;

  failure:
  if(A){
    free(A);
  }
  if(B){
    free(B);
  }
  if(C_collapse){
    free(C_collapse);
  }
  if(C_tiling){
    free(C_tiling);
  }
  return EXIT_FAILURE;
}
