#ifndef CSRMATRIX_H
#define CSRMATRIX_H

#include "types.h"

typedef struct csr_matrix_t *csr_matrix;

/**
  * allocate and initialize csr matrix based on the links list
  */
csr_matrix csr_matrix_new(int njuncs, int nlinks, Slink *links);

/**
  * free all csr matrix related memory
  */
void csr_matrix_free(csr_matrix matrix);

/**
  * zero all values of the matrix
  */
void csr_matrix_zero(csr_matrix matrix);

/**
  * add value v to ith diagonal entry
  */
void csr_matrix_diagonal_add(csr_matrix matrix, int i, double v);

/**
  * set ith diagonal entry
  */
//void csr_matrix_diagonal_set(csr_matrix matrix, int i, double v);

/**
  * add value v to ith link coeficient
  */
void csr_matrix_link_add(csr_matrix matrix, int i, double v);

/**
  * set ith link coeficient to value v
  */
//void csr_matrix_link_set(csr_matrix matrix, int i, double v);

/**
  * solve the linear system defined by matrix and F replacing values in F
  */
void csr_matrix_solve(csr_matrix matrix, double *F, double *X);


/**
  * dumps the matrix in (row, column, value) triples
  */
void csr_matrix_dump(csr_matrix matrix);

/**
  * dumps the matrix into a csr loadable format
  */
void csr_matrix_dump_csr(csr_matrix matrix, const double *b, const double *x, const char *fname);

/**
  * dump matrix in MatrixMarket format.
  */
void csr_matrix_dump_mm(csr_matrix matrix, const char *fname);
#endif // CSRMATRIX_H
