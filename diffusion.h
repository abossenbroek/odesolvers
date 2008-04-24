#ifndef _DIFFUSION_H_
#define _DIFFUSION_H_

#include <mpi.h>

#define NUM_PARAMS 7

typedef struct {
	double dx;
	double dt;
	double D;
	int	 ntotal;
	int	 ttotal;
	int	 x;
	int	 y;
} pparams;

int pparams_blength[NUM_PARAMS] = {1, 1, 1, 1, 1, 1, 1};

MPI_Datatype pparams_type[NUM_PARAMS] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, 
	MPI_INT, MPI_INT, MPI_INT, MPI_INT};

enum { X_COORD,
	Y_COORD };
#endif //  _DIFFUSION_H_

