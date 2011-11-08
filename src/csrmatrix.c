#include "csrmatrix.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <mkl_dss.h>

#define ASSURE_UPPER(n1, n2) if (n2 < n1) { \
n1 = n1 ^ n2; \
n2 = n1 ^ n2; \
n1 = n1 ^ n2; \
}

struct csr_matrix_t {
   int nnz;            //number of non zero elements a.k.a. size of value and columns
   int n;              //matrix dimension NxN
   double *values;     //value array
   int *columns;       //column of each entry in value
   int *rowIndex;      //array of size n+1 for each row i values and columns range from rowIndex[i] - rowIndex[i+1]
   int *link_offset;   //direct offset of ith link int values array
   _MKL_DSS_HANDLE_t handle; //handle for mkl dss library
};

// declarations internal api
static int get_nnz(int njuncs, int nlinks, Slink *links);
static void set_rowIndex(csr_matrix matrix, int nlinks, Slink *links);
static void set_columns(csr_matrix matrix, int nlinks, Slink *links);
static void set_link_offsets(csr_matrix matrix, int nlinks, Slink *links);
static void init_pardiso(csr_matrix matrix);
static int is_sorted(csr_matrix matrix);
static int is_upper(csr_matrix matrix);
static void sort(csr_matrix matrix);

//external API
csr_matrix csr_matrix_new(int njuncs, int nlinks, Slink *links) {
   csr_matrix matrix;

   matrix = malloc(sizeof(struct csr_matrix_t));
   matrix->n = njuncs;
   matrix->nnz = get_nnz(njuncs, nlinks, links);
   matrix->columns = malloc(matrix->nnz*sizeof(int));
   matrix->values = malloc(matrix->nnz*sizeof(double));
   matrix->rowIndex = malloc((matrix->n+1)*sizeof(int));
   matrix->link_offset = malloc(nlinks*sizeof(int));


   set_rowIndex(matrix, nlinks, links);
   set_columns(matrix, nlinks, links);

   sort(matrix);
   set_link_offsets(matrix, nlinks, links);
   assert(is_sorted(matrix));
   assert(is_upper(matrix));
   init_pardiso(matrix);

   return matrix;
}

void csr_matrix_free(csr_matrix matrix) {
   int error, opt;
   opt = MKL_DSS_DEFAULTS;
   error = dss_delete(matrix->handle, opt);
   assert(error == MKL_DSS_SUCCESS);
   free(matrix->link_offset);
   free(matrix->rowIndex);
   free(matrix->values);
   free(matrix->columns);
   free(matrix);
   matrix = 0;
}

void csr_matrix_zero(csr_matrix matrix) {
   int i;
   for (i = 0; i < matrix->nnz; i++) {
      matrix->values[i] = 0.0;
   }
}

void csr_matrix_diagonal_add(csr_matrix matrix, int i, double v) {
   assert(matrix->rowIndex[i]-1 < matrix->nnz);
   matrix->values[matrix->rowIndex[i]-1] += v;
}

extern int Nlinks;

void csr_matrix_link_add(csr_matrix matrix, int i, double v) {
   assert(i >= 0 && i < Nlinks);
   assert(matrix->link_offset[i]-1 < matrix->nnz);
   assert(matrix->link_offset[i]-1 >= 0);
   matrix->values[matrix->link_offset[i]] += v;
}

void csr_matrix_solve(csr_matrix matrix, double *F, double *X) {
   int opt, error;
   opt = MKL_DSS_POSITIVE_DEFINITE;
   error = dss_factor_real(matrix->handle, opt, matrix->values);
   opt = MKL_DSS_DEFAULTS;
   int nrhs = 1;
   error = dss_solve_real(matrix->handle, opt, F, nrhs, X);
   assert(error == MKL_DSS_SUCCESS);
}

void csr_matrix_dump(csr_matrix matrix) {
   int i, j;
   printf("\n");
   for (i = 0; i < matrix->n; i++) {
      int start = matrix->rowIndex[i]-1;
      int stop = matrix->rowIndex[i+1]-1;
      for (j = start; j < stop; j++) {
         printf("(%d,%d) = %f\n", i, matrix->columns[j]-1, matrix->values[j]);
         fflush(stdout);
      }
   }
   fflush(stdout);
}

// BEGIN INTERNAL API

static int get_nnz(int njuncs, int nlinks, Slink *links) {
   int i;
   int nnz = njuncs;
   for (i = 1; i <= nlinks; i++) {
      nnz += (links[i].N1 <= njuncs) && (links[i].N2 <= njuncs);
   }
   return nnz;
}

static void set_rowIndex(csr_matrix matrix, int nlinks, Slink *links) {
   int njuncs = matrix->n;
   int i;
//#pragma omp parallel for private(i)
   for (i = 1; i < matrix->n+1; i++) {
      matrix->rowIndex[i] = 1;
   }
   matrix->rowIndex[0] = 0;

//#pragma omp parallel for private(i)
   for (i = 1; i <= nlinks; i++) {
      int n1 = links[i].N1;
      int n2 = links[i].N2;
      if (n1 <= njuncs && n2 <= njuncs) {
         ASSURE_UPPER(n1, n2);
         matrix->rowIndex[n1]++;
      }
   }

   matrix->rowIndex[0] = 1;

   for (i = 0; i < matrix->n; i++) {
      assert(i+1 < matrix->n+1);
      matrix->rowIndex[i+1] += matrix->rowIndex[i];
   }
}

static void set_columns(csr_matrix matrix, int nlinks, Slink *links) {
   int i;
   int njuncs = matrix->n;
   int *tmp = malloc(matrix->n*sizeof(int));

//#pragma omp parallel for private(i)
   for (i = 0; i < matrix->n; i++) {
      tmp[i] = 0;
   }
   //memset(tmp, 0, sizeof(int) * matrix->n);

//#pragma omp parallel for private(i)
   for (i = 0; i < njuncs; i++) {
      matrix->columns[matrix->rowIndex[i]-1] = i+1; //set diagonal indices
   }

//#pragma omp parallel for private(i)
   for (i = 1; i <= nlinks; i++) {
      int n1 = links[i].N1;
      int n2 = links[i].N2;
      if (n1 <= njuncs && n2 <= njuncs) {
         ASSURE_UPPER(n1, n2);

         int offset = matrix->rowIndex[n1-1]+tmp[n1-1];
         assert(offset > 0);
         assert(offset < matrix->nnz);
         matrix->columns[offset] = n2;
         //matrix->link_offset[i-1] = offset+1;
//#pragma omp atomic
         tmp[n1-1]++;
      }
   }
   free(tmp);
}

static void set_link_offsets(csr_matrix matrix, int nlinks, Slink *links) {
   int i;
   int njuncs = matrix->n;

#pragma omp parallel for private(i)
   for (i = 1; i <= nlinks; i++) {
      int n1 = links[i].N1;
      int n2 = links[i].N2;
      if (n1 <= njuncs && n2 <= njuncs) {
         ASSURE_UPPER(n1, n2);
         int start = matrix->rowIndex[n1-1];
         int stop = matrix->rowIndex[n1];
         int k;
         int found = FALSE;
         for (k = start; k < stop; k++) {
            if (matrix->columns[k-1] == n2) {
               found = TRUE;
               matrix->link_offset[i-1] = k-1;
               break;
            }
         }
         assert(found);
      }
   }
}

static void init_pardiso(csr_matrix matrix) {
   _INTEGER_t error;
   int opt = MKL_DSS_DEFAULTS;
   //int opt = MKL_DSS_MSG_LVL_INFO + MKL_DSS_TERM_LVL_ERROR;
   error = dss_create(matrix->handle, opt);
   assert(error == MKL_DSS_SUCCESS);

   int dss_opt = MKL_DSS_SYMMETRIC;

   error = dss_define_structure(matrix->handle, dss_opt,
                                matrix->rowIndex, matrix->n, matrix->n,
                                matrix->columns, matrix->nnz);

   assert(error == MKL_DSS_SUCCESS);

   error = dss_reorder(matrix->handle, opt, 0);
   assert(error == MKL_DSS_SUCCESS);
}

static int is_sorted(csr_matrix matrix) {
   int i;
   int sorted = TRUE;
   for (i = 0; i < matrix->n; i++) {
      int start = matrix->rowIndex[i]-1;
      int stop = matrix->rowIndex[i+1]-1;
      int k;
      for (k = start; k < stop-1; k++) {
         if (matrix->columns[k] > matrix->columns[k+1]) {
            printf("row %d is not sorted\n", i);
            sorted = FALSE;
         }
      }
   }
   return sorted;
}

static int icompare (const void *a, const void *b) {
   int a_int = *((int*) a);
   int b_int = *((int*) b);
   return a_int - b_int;
}


static void sort(csr_matrix matrix) {
   int i;
//#pragma omp parallel for private(i)
   for (i = 0; i < matrix->n; i++) {
      int start = matrix->rowIndex[i]-1;
      int stop = matrix->rowIndex[i+1]-1;
      int k;
      for (k = start; k < stop; k++) {
         assert(matrix->columns[k] > 0);
      }
      qsort(&matrix->columns[start], stop-start, sizeof(int), icompare);
      for (k = start; k < stop; k++) {
         assert(matrix->columns[k] > 0);
      }
   }
}

static int is_upper(csr_matrix matrix) {
   int i;
   int upper = TRUE;
   for (i = 0; i < matrix->n; i++) {
      int start = matrix->rowIndex[i]-1;
      int stop = matrix->rowIndex[i+1]-1;
      int k;
      for (k = start; k < stop; k++) {
         int column = matrix->columns[k]-1;
         if (column < i) {
            upper = FALSE;
         }
      }
   }
   return upper;
}
