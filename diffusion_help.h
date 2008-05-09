#ifndef _DIFFUSION_HELP_H
#define _DIFFUSION_HELP_H
#include "mpi.h"

/* By defining NO_SSE the user can disable SSE support in the program. By
 * defining DOUBLE the user can turn on double precision for the grid. */
#ifndef NO_SSE
#	ifdef DOUBLE
#		include <emmintrin.h>
#		define SIMD_CAPACITY 2
typedef __m128d grid_simd_type;
#	else
#		include <xmmintrin.h>
#		define SIMD_CAPACITY 4
typedef __m128 grid_simd_type;
#	endif /* DOUBLE */
#endif /* NO_SSE */

#	ifdef DOUBLE
#		define MPI_GRID_TYPE MPI_DOUBLE
typedef double grid_type;
#	else
#		define MPI_GRID_TYPE MPI_FLOAT
typedef float grid_type;
#	endif /* DOUBLE */

#define FIRST_FREE_TAG 9

enum { X_COORD,
	Y_COORD,
	X_UP_TAG,
   X_DOWN_TAG,
	Y_LEFT_TAG,
	Y_RIGHT_TAG, 
	TIME_COMP_TAG,
	TIME_COMM_TAG,
	TIME_INIT_TAG,
	PRINT_COMM = 1000
};

void send_grid(grid_type **grid, size_t *grains, int *offset, int rank,
		MPI_Comm comm, double *time_comm, int base_tag);

void recv_grid(grid_type **grid, size_t *grains, int *offset, int time,
		int rank, int nnodes, MPI_Comm comm, double *time_comm, int base_tag, 
		void (*handler)(int, int, int, size_t*, int*, grid_type*, void *), void *handlerargs);
	
void print_elem(int time, int rank, int x, size_t *grains, int *offset, grid_type *column,
		void *fd);

#endif

/* vim: set spell spelllang=en:cindent:tw=80:et*/

