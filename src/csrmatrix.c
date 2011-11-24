#include "csrmatrix.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

/* PARDISO prototype. */
void pardisoinit (void   *, int    *,   int *, int *, double *, int *);
void pardiso     (void   *, int    *,   int *, int *,    int *, int *,
                  double *, int    *,    int *, int *,   int *, int *,
                     int *, double *, double *, int *, double *);
void pardiso_chkmatrix  (int *, int *, double *, int *, int *, int *);
void pardiso_chkvec     (int *, int *, double *, int *);
void pardiso_printstats (int *, int *, double *, int *, int *, int *,
                           double *, int *);

#define ASSURE_UPPER(n1, n2) if (n2 < n1) { \
n1 = n1 ^ n2; \
n2 = n1 ^ n2; \
n1 = n1 ^ n2; \
}

struct csr_matrix_t {
   int      nnz;           //number of non zero elements a.k.a. size of value and columns
   int      n;             //matrix dimension NxN
   double  *values;        //value array
   int     *columns;       //column of each entry in value
   int     *rowIndex;      //array of size n+1 for each row i values and columns range from rowIndex[i] - rowIndex[i+1]
   int     *link_offset;   //direct offset of ith link int values array

//pardiso stuff
   void    *handle[64];    //handle for pardiso
   int      iparm[64];
   double   dparm[64];
   int      mtype;     //real positiv symmetric
   int      maxfct;
   int      mnum;
   int      nrhs;
   int      msglvl;
};

// declarations internal api
static int get_nnz(int njuncs, int nlinks, Slink *links, int *paralinks);
static void set_rowIndex(csr_matrix matrix, int nlinks, Slink *links, int *paralinks);
static void set_columns(csr_matrix matrix, int nlinks, Slink *links, int *paralinks);
static void set_link_offsets(csr_matrix matrix, int nlinks, Slink *links, int *paralinks);
static void init_pardiso(csr_matrix matrix);
static int is_sorted(csr_matrix matrix);
static int is_upper(csr_matrix matrix);
static void sort(csr_matrix matrix);
static void set_paralinks(int nlinks, Slink *links, int *paralinks);

//external API
csr_matrix csr_matrix_new(int njuncs, int nlinks, Slink *links) {
   csr_matrix matrix;

   matrix = malloc(sizeof(struct csr_matrix_t));
   matrix->n = njuncs;
   //printf("\nmatrix size: %d\n", matrix->n);
   //printf("\nnumber of equations: %d\n", nlinks);

   int *paralinks = malloc(nlinks  * sizeof(int));
   set_paralinks(nlinks, links, paralinks);

   matrix->nnz = get_nnz(njuncs, nlinks, links, paralinks);
   matrix->columns = malloc(matrix->nnz*sizeof(int));
   matrix->values = malloc(matrix->nnz*sizeof(double));
   matrix->rowIndex = malloc((matrix->n+1)*sizeof(int));
   matrix->link_offset = malloc(nlinks*sizeof(int));


   set_rowIndex(matrix, nlinks, links, paralinks);
   set_columns(matrix, nlinks, links, paralinks);

   sort(matrix);
   set_link_offsets(matrix, nlinks, links, paralinks);
   assert(is_sorted(matrix));
   assert(is_upper(matrix));
   init_pardiso(matrix);

   free(paralinks);

   return matrix;
}

void csr_matrix_free(csr_matrix matrix) {
   int error;
   int phase = 0;

   pardiso (matrix->handle, &matrix->maxfct, &matrix->mnum, &matrix->mtype, &phase,
            &matrix->n, matrix->values, matrix->rowIndex, matrix->columns, 0/*perm*/, &matrix->nrhs,
            matrix->iparm, &matrix->msglvl, 0, 0, &error,  matrix->dparm);
   if (error != 0) {
      printf("error releasing pardios memory\n");
   }

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
   int error;
   int phase = 23;
   pardiso (matrix->handle, &matrix->maxfct, &matrix->mnum, &matrix->mtype, &phase,
            &matrix->n, matrix->values, matrix->rowIndex, matrix->columns, 0/*perm*/, &matrix->nrhs,
            matrix->iparm, &matrix->msglvl, F, X, &error,  matrix->dparm);

   if (error != 0) {
      printf("\nERROR during solution: %d", error);
      exit(3);
   }
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

void csr_matrix_dump_csr(csr_matrix matrix, const double *b, const double *x, const char *fname) {
    FILE *dump_file;
    int i;

    dump_file = fopen(fname, "w+");

    fprintf(dump_file, "%d %d\n", matrix->n, matrix->nnz);

    for (i = 0; i < matrix->n+1; i++) {
        fprintf(dump_file, "%d\n", matrix->rowIndex[i]);
    }
    fprintf(dump_file, "\n");

    for (i = 0; i < matrix->nnz; i++) {
        fprintf(dump_file, "%d %f\n", matrix->columns[i], matrix->values[i]);
    }

    fprintf(dump_file, "\n");

    for (i = 0; i < matrix->n; i++) {
        fprintf(dump_file, "%f %f\n", b[i], x[i]);
    }

    fprintf(dump_file, "\n");
    fclose(dump_file);
}

void csr_matrix_dump_mm(csr_matrix matrix, const char *fname) {
    FILE *dump_file;
    int i, j;

    dump_file = fopen(fname, "w+");

    fprintf(dump_file, "%%%%MatrixMarket matrix coordinate real symmetric\n");
    fprintf(dump_file, "%d %d %d\n", matrix->n, matrix->n, matrix->nnz);

    for (i = 0; i < matrix->n; i++) {
       int start = matrix->rowIndex[i]-1;
       int stop = matrix->rowIndex[i+1]-1;
       for (j = start; j < stop; j++) {
          fprintf(dump_file, "%d %d %f\n", i+1, matrix->columns[j], matrix->values[j]);
       }
    }
}

// BEGIN INTERNAL API

static int get_nnz(int njuncs, int nlinks, Slink *links, int *paralinks) {
   int i;
   int nnz = njuncs;
   for (i = 1; i <= nlinks; i++) {
      nnz += (links[i].N1 <= njuncs) && (links[i].N2 <= njuncs) && !paralinks[i-1];
   }
   return nnz;
}

static void set_rowIndex(csr_matrix matrix, int nlinks, Slink *links, int *paralinks) {
   int njuncs = matrix->n;
   int i;
//#pragma omp parallel for private(i)
   for (i = 1; i < matrix->n+1; i++) {
      matrix->rowIndex[i] = 1;
   }
   matrix->rowIndex[0] = 0;

   //count number of nnz per row
   for (i = 1; i <= nlinks; i++) {
      int n1 = links[i].N1;
      int n2 = links[i].N2;
      if (n1 <= njuncs && n2 <= njuncs && !paralinks[i-1]) {
         ASSURE_UPPER(n1, n2);
         matrix->rowIndex[n1]++;
      }
   }

   matrix->rowIndex[0] = 1;

   //set start and end
   for (i = 0; i < matrix->n; i++) {
      assert(i+1 < matrix->n+1);
      matrix->rowIndex[i+1] += matrix->rowIndex[i];
   }
}

static void set_columns(csr_matrix matrix, int nlinks, Slink *links, int *paralinks) {
   int i;
   int njuncs = matrix->n;
   int *tmp = malloc(matrix->n*sizeof(int));

   for (i = 0; i < matrix->n; i++) {
      tmp[i] = 0;
   }

   for (i = 0; i < njuncs; i++) {
      matrix->columns[matrix->rowIndex[i]-1] = i+1; //set diagonal indices
   }

   for (i = 1; i <= nlinks; i++) {
      int n1 = links[i].N1;
      int n2 = links[i].N2;
      if (n1 <= njuncs && n2 <= njuncs && !paralinks[i-1]) {
         ASSURE_UPPER(n1, n2);

         int offset = matrix->rowIndex[n1-1]+tmp[n1-1];
         assert(offset > 0);
         assert(offset < matrix->nnz);
         matrix->columns[offset] = n2;
         tmp[n1-1]++;
      }
   }
   free(tmp);
}

static void set_link_offsets(csr_matrix matrix, int nlinks, Slink *links, int *paralinks) {
   int i;
   int njuncs = matrix->n;

   for (i = 1; i <= nlinks; i++) {
      int n1 = links[i].N1;
      int n2 = links[i].N2;
      if (n1 <= njuncs && n2 <= njuncs) {
         ASSURE_UPPER(n1, n2);
         if (paralinks[i-1]) {
            int pn1 = links[paralinks[i-1]].N1;
            int pn2 = links[paralinks[i-1]].N2;
            ASSURE_UPPER(pn1, pn2);
            assert(n1 == pn1);
            assert(n2 == pn2);

            matrix->link_offset[i-1] = matrix->link_offset[paralinks[i-1]-1];
            continue;
         }
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
   int error = 0;
   int solver = 0; /* use sparse direct solver */
   matrix->mtype = 2;
   matrix->maxfct = 1;
   matrix->mnum = 1;
   matrix->nrhs = 1;
   matrix->msglvl = 0;

   int num_procs;

   char *var = getenv("OMP_NUM_THREADS");
   if (var != NULL) {
      sscanf(var , "%d" ,&num_procs);
   } else {
      printf("Set environment OMP_NUM_THREADS");
      exit(1);
   }


   matrix->iparm[2] = num_procs;
   matrix->iparm[7] = 0; //no iterative refinement


   pardisoinit (matrix->handle,  &matrix->mtype, &solver, matrix->iparm, matrix->dparm, &error);

   if (error != 0) {
      if (error == -10 )
         printf("No license file found \n");
      if (error == -11 )
         printf("License is expired \n");
      if (error == -12 )
         printf("Wrong username or hostname \n");
   } else {
      printf("[PARDISO]: License check was successful ... \n");
   }

   pardiso_chkmatrix  (&matrix->mtype, &matrix->n, matrix->values, matrix->rowIndex, matrix->columns, &error);

   if (error != 0) {
      printf("\nERROR in consistency of matrix: %d\n", error);
      exit(1);
   }

   fflush(stdout);

   int phase = 11;

   pardiso (matrix->handle, &matrix->maxfct, &matrix->mnum, &matrix->mtype, &phase,
            &matrix->n, matrix->values, matrix->rowIndex, matrix->columns, 0, &matrix->nrhs,
            matrix->iparm, &matrix->msglvl, 0, 0, &error, matrix->dparm);

   if (error != 0) {
      printf("\nERROR during symbolic factorization: %d", error);
      exit(1);
   }
   printf("\nReordering completed ... ");
   printf("\nNumber of nonzeros in factors  = %d", matrix->iparm[17]);
   printf("\nNumber of factorization MFLOPS = %d", matrix->iparm[18]);
}

static int is_sorted(csr_matrix matrix) {
   int i;
   int sorted = TRUE;
   printf("\n");
   for (i = 0; i < matrix->n; i++) {
      int start = matrix->rowIndex[i]-1;
      int stop = matrix->rowIndex[i+1]-1;
      int k;
      for (k = start; k < stop-1; k++) {
         if (matrix->columns[k] >= matrix->columns[k+1]) {
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

static void set_paralinks(int nlinks, Slink *links, int *paralinks) {
   int i, k;

   //mark all not parallel
   for (i = 0; i < nlinks; i++) {
      paralinks[i] = 0;
   }
   for (i = 1; i <= nlinks; i++) {
      int i_n1 = links[i].N1;
      int i_n2 = links[i].N2;
      ASSURE_UPPER(i_n1, i_n2);
      for (k = i+1; k <= nlinks; k++) {
         int k_n1 = links[k].N1;
         int k_n2 = links[k].N2;
         ASSURE_UPPER(k_n1, k_n2);
         if (i_n1 == k_n1 && i_n2 == k_n2) {
            paralinks[k-1] = i; //set to the parallel link
            //printf("link %d (%d -> %d) is parallel to link %d (%d -> %d)\n", i, i_n1, i_n2, k, k_n1, k_n2);
         }
      }
   }
}
