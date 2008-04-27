/* 
 * This program solves the pde for the 2D diffusion equation using MPI-v2 and
 * SSE2.
 *
 * Copyright (C) 2008  Anton Bossenbroek <abossenb@science.uva.nl>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <mpi.h>
#include <math.h>
#include <sysexits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include <unistd.h>
#include <string.h>

#include "diffusion.h"

#if MPI_VERSION < 2
#	error "Need MPI version 2 for this program"
#endif

#ifndef NO_SEE
#	include "xmmintrin.h"
#endif

void usage(void);

int getparams(int argc, char *argv[], pparams *params, FILE **gridfile,
		FILE **statusfile, MPI_Datatype *solve_params_dt, int rank);

void send_grid(float **grid, size_t *grains, int *offset, int rank, MPI_Comm
		comm);

int main(int argc, char *argv[])
{
	MPI_Comm comm = MPI_COMM_WORLD;  /* Communicator. */
	MPI_Datatype pparams_mpi;		   /* Contains all the parameters. */
	int nnodes = 5;						/* Total number of nodes. */
	int gsize[2] = {0};					/* Grid size. */
	int periods[2] = {false, false};
	int rank = 0;
	int coord[2];
	/* We are interested in the diffusion process in two directions. */
	const int dims = 2;
	int status;
	int offset[2];
	size_t grains[2];
	int coord_lneigh[2];
	int coord_rneigh[2];
	int coord_uneigh[2];
	int coord_dneigh[2];
	int rank_lneigh;
	int rank_rneigh;
	int rank_uneigh;
	int rank_dneigh;
	MPI_Status xdown_status;
	MPI_Status xup_status;
	MPI_Status yleft_status;
	MPI_Status yright_status;
	
#ifndef NO_SSE
	__m128	sse_ratio;
	__m128	sse_ratio1;
	__m128	curr_grid;
	__m128	currr_grid;
	__m128	currl_grid;
	__m128	curru_grid;
	__m128	currd_grid;
	__m128	ngrid_sse;

#endif


	float **grid = NULL;
	float **ngrid = NULL;
	float *xdown;
	float *xup;

	float ratio;

	/* Arguments. */
	pparams params;

	FILE	*profilefile = NULL;
	FILE	*statusfile = NULL;

	size_t i;
	size_t x, y;
#ifndef NO_SSE
	size_t y_remain;
#endif
	long time;

	MPI_Init(&argc, &argv);
   MPI_Comm_rank(comm, &rank);
	/* Parse the parameters. The function only parses parameters if this
	 * processor has rank zero. */
	if ((status = getparams(argc, argv, &params, &profilefile, &statusfile,
					&pparams_mpi, rank)) != EX_OK)
		MPI_Abort(comm, status);

	/* Send all the parameters to the remaining nodes in the comm. */
	MPI_Bcast(&params, 1, pparams_mpi, 0, MPI_COMM_WORLD);

	/* Determine the number of nodes in the communicator. */
	MPI_Comm_size(comm, &nnodes);

	/* Check whether the number of nodes can be used to form a two dimensional
	 * grid equal height and length. */
	if (rank == 0 && nnodes / params.l != params.h) {
		usage(); 
	}
	/* The grid should be the same on each node. */
	if (rank == 0 && params.ntotal % (params.l * params.h) != 0) {
		usage();
	}
	/* Compute the grid form. */
	gsize[X_COORD] = params.l;
	gsize[Y_COORD] = params.h;

	/* Create a get information of a Cartesian grid topology. */
	if (MPI_Cart_create(comm, dims, gsize, periods, true, &comm) != 
			MPI_SUCCESS) 
		MPI_Abort(comm, EX_UNAVAILABLE);

	/* Translate the current rank to the coordinate in the Cartesian 
	 * topology. */
	MPI_Cart_coords(comm, rank, dims, coord);


	/* Using the coordinate of the current node we can determine the amount of
    * points this node has to compute and the offset of the points. */
	for (i = 0; i < dims; i++) {
		grains[i] = params.ntotal / gsize[i] + (params.ntotal % gsize[i] +
				gsize[i] - coord[i] - 1) / gsize[i];

		if (grains[i] > params.ntotal / gsize[i])
			offset[i] = (params.ntotal / gsize[i] + 1) * coord[i];
		else
			offset[i] = params.ntotal / gsize[i] * coord[i] + params.ntotal % gsize[i];
	}
	
	/* With the current dimensions arrays which represent the grid can be
	 * allocated. Two more entries are used to store neighbouring points. 
	 *
	 * Grids are composed as follows:
	 *
	 * |  |  |  |   |   |       |            |
	 * |  |  |  |   |   |       |            |
	 * |  |  |  |   |   |       |            |
	 * |  |  |  |   |   |       |            |
	 * |  |  |  |   |   |       |            |
	 * 0  1  2  ..  ..  ..  grains[x]    grains[x] + 1
	 * |  |  |  |   |   |       |            |
	 * |  |  |  |   |   |       |            |
	 * |  |  |  |   |   |       |            |
	 * |  |  |  |   |   |       |            |
	 * |  |  |  |   |   |       |            |
	 *
	 *
	 */
	if ((grid = calloc(grains[X_COORD] + 2, sizeof(float *))) == NULL ||
			(ngrid = calloc(grains[X_COORD] + 2, sizeof(float *))) == NULL)
		MPI_Abort(comm, EX_OSERR);

	for (i = 0; i < grains[X_COORD] + 2; i++)
		if ((grid[i] = calloc(grains[Y_COORD] + 2, sizeof(float))) == NULL ||
				(ngrid[i] = calloc(grains[Y_COORD] + 2, sizeof(float))) == NULL)
			MPI_Abort(comm, EX_OSERR);

	/* Create temporary storage to prevent iterating through the entire grid. */
	if ((xdown = calloc(grains[X_COORD], sizeof(float))) == NULL ||
			(xup = calloc(grains[X_COORD], sizeof(float))) == NULL)
		MPI_Abort(comm, EX_OSERR);

	if ((ratio = params.dt * params.D * 4 / (params.dx * params.dx)) > 1)
		ratio = 1;

#ifndef NO_SSE
	sse_ratio = _mm_set_ps1(ratio);
	/* This variable is used to reduce the number of computations when computing
	 * the finite difference scheme. */
	sse_ratio1 = _mm_set_ps1(1.0 - 4.0 * ratio);
	y_remain = grains[Y_COORD] % 4;
#endif

	/* All the coordinates are translated to ranks by first computing the
	 * coordinate of the appropriate neighbours. Then the coordinates are
	 * used to determine the rank. These ranks can be used for the
	 * communication. */
	coord_lneigh[Y_COORD] = (coord[Y_COORD] + gsize[Y_COORD] - 1) % gsize[Y_COORD];
	coord_rneigh[Y_COORD] = (coord[Y_COORD] + 1) % gsize[Y_COORD];
	coord_lneigh[X_COORD] = coord[X_COORD];
	coord_rneigh[X_COORD] = coord[X_COORD];

	coord_dneigh[X_COORD] = (coord[X_COORD] + gsize[X_COORD] - 1) % gsize[X_COORD];
	coord_uneigh[X_COORD] = (coord[X_COORD] + 1) % gsize[X_COORD];
	coord_dneigh[Y_COORD] = coord[Y_COORD];
	coord_uneigh[Y_COORD] = coord[Y_COORD];

	MPI_Cart_rank(comm, coord_lneigh, &rank_lneigh);
	MPI_Cart_rank(comm, coord_rneigh, &rank_rneigh);
	MPI_Cart_rank(comm, coord_dneigh, &rank_dneigh);
	MPI_Cart_rank(comm, coord_uneigh, &rank_uneigh);

	for (time = 0; time < params.ttotal; time++) {
		/* Create two new arrays to prevent bad memory access. */
		for (i = 0; i < grains[X_COORD]; i++) {
			xup[i] = grid[i][grains[Y_COORD]];
			xdown[i] = grid[i][1];
		}

		MPI_Send((void *)xup, grains[X_COORD], MPI_FLOAT, rank_uneigh, X_UP_TAG, comm);
		MPI_Send((void *)xdown, grains[X_COORD], MPI_FLOAT, rank_dneigh, X_DOWN_TAG, comm);
		MPI_Send((void *)(grid[grains[X_COORD] + 1] + 1), grains[Y_COORD], MPI_FLOAT,
				rank_rneigh, Y_RIGHT_TAG, comm);
		MPI_Send((void *)(grid[1] + 1), grains[Y_COORD], MPI_FLOAT,
				rank_lneigh, Y_LEFT_TAG, comm);

		MPI_Recv((void *)xup, grains[X_COORD], MPI_FLOAT, rank_dneigh, X_UP_TAG,
				comm, &xup_status);
		MPI_Recv((void *)xdown, grains[X_COORD], MPI_FLOAT, rank_uneigh,
				X_DOWN_TAG, comm, &xdown_status);
		MPI_Recv((void *)(grid[grains[X_COORD] + 1] + 1), grains[Y_COORD], MPI_FLOAT,
				rank_lneigh, Y_RIGHT_TAG, comm, &yright_status);
		MPI_Recv((void *)(grid[1] + 1), grains[Y_COORD], MPI_FLOAT,
				rank_rneigh, Y_LEFT_TAG, comm, &yleft_status);

		/* The freshly received xup and xdown have to be put in the grid. */
		for (i = 0; i < grains[X_COORD]; i++) {
			grid[i][grains[Y_COORD]] = xup[i];
			grid[i][1] = xdown[i];
		}

		/* Ensure that all the processes are at the same point. */
		MPI_Barrier(comm);

		for (x = 1; x < grains[X_COORD] + 1; x++) {
#ifdef NO_SSE
			for (y = 1; y < grains[Y_COORD] + 1; y++) {
				/* Do the finite difference computation. */
				ngrid[x][y] = grid[x][y] + ratio * (grid[x][y + 1] + grid[x][y - 1]
						+ grid[x + 1][y] + grid[x - 1][y] - 4 * grid[x][y]);
			}
#else
			for (y = 1; y < grains[Y_COORD] + 1 - y_remain; y += 4) {
				/* Load all the necessary values to  SSE variables. */
				/* r3 = (x, y + 3)
				 * r2 = (x, y + 2)
				 * r1 = (x, y + 1)
				 * r0 = (x, y)
				 */
				curr_grid = _mm_loadu_ps(grid[x] + y);
				currr_grid = _mm_loadu_ps(grid[x + 1] + y);
				currl_grid = _mm_loadu_ps(grid[x - 1] + y);
				curru_grid = _mm_loadu_ps(grid[x] + y + 1);
				currd_grid = _mm_loadu_ps(grid[x] + y - 1);

				/* Perform arithmetic in an order which should reduce the number of
				 * bubbles in the processor pipeline. */
				currr_grid = _mm_add_ps(currr_grid, currl_grid);
				curru_grid = _mm_add_ps(curru_grid, currd_grid);
				ngrid_sse = _mm_mul_ps(curr_grid, sse_ratio1);
				currr_grid = _mm_add_ps(currr_grid, curru_grid);
				currr_grid = _mm_mul_ps(currr_grid, sse_ratio);
				ngrid_sse = _mm_add_ps(currr_grid, ngrid_sse);
				_mm_storeu_ps(ngrid[x] + y, ngrid_sse);

			}

			for (; y < grains[Y_COORD] + 1; y++) {
				ngrid[x][y] = grid[x][y] + ratio * (grid[x][y + 1] + grid[x][y - 1]
						+ grid[x + 1][y] + grid[x - 1][y] - 4 * grid[x][y]);
			}
#endif /* NO_SSE */
		}

		/* Copy the new grid to the current grid. */
		for (x = 1; x < grains[X_COORD] + 1; x++) 
			memcpy((void *)(grid[x] + 1), (void *)(ngrid[x] + 1), grains[Y_COORD]
					* sizeof(float));
		
	}

	/* Free the memory used for the grid. */
	for (i = 0; i < grains[X_COORD] + 2; i++) {
		free(grid[i]);
		free(ngrid[i]);
	}

	free(grid);
	free(ngrid);
	
	if (rank == 0) {
		fclose(statusfile);
		fclose(profilefile);
	}


	MPI_Finalize();
	return EX_OK;
}

void
send_grid(float **grid, size_t *grains, int *offset, int rank, 
		MPI_Comm comm)
{
	MPI_Request dummy;

	if (rank != 0) {
		MPI_Isend((void *)grains, 2, MPI_INT, 0, GRAIN_COMM, comm, &dummy);
		MPI_Isend((void *)offset, 2, MPI_INT, 0, OFFSET_COMM, comm, &dummy);

		for (size_t i = 1; i < grains[X_COORD]; i++) 
			MPI_Isend((void *)(grid + i), grains[Y_COORD], MPI_FLOAT, 0, GRID_COMM +
					i - 1, comm, &dummy);
	}
}

void 
print_grid(FILE *fd, float **grid, size_t *grains, int *offset, int time, int rank,
      int nnodes, MPI_Comm comm) {
   /* Only rank 0 has to perform this function. */
   if (rank > 0)
      return;

	size_t x = 0, y = 0;
   float *recv_buff;
   size_t recv_grains[2];
   int recv_offset[2];
   MPI_Status status;

   /* When load balancing the node with rank 0 will always have the largest
    * amount of grains to compute. Therefore the number of grains which are
    * to be computed by the root can also be used as buffer size. */
   if ((recv_buff = calloc(grains[X_COORD], sizeof(float))) == NULL) {
      MPI_Abort(MPI_COMM_WORLD, EX_OSERR);
   }
   if (fd == NULL)
      MPI_Abort(MPI_COMM_WORLD, EX_SOFTWARE);

   /* Print all the values computed by the node with rank 0 */
   for (x = 0; x < grains[X_COORD]; ++x) 
		for (y = 0; y < grains[Y_COORD]; ++y) 
			fprintf(fd, "%i %i %i %i %f\n", time, 0, (int)x, (int)y, grid[x + 1][y + 1]);

   /* Print all the values computed by the nodes with rank > 0. These 
    * values have to be received from the other nodes. */
   for (int proc = 1; proc < nnodes; proc++) {
      /* Perform blocking receives from the non blocking sends. */
      MPI_Recv((void *)recv_grains, 2, MPI_INT, proc, GRAIN_COMM, comm,
            &status);
      MPI_Recv((void *)recv_offset, 2, MPI_INT, proc, OFFSET_COMM, comm,
            &status);
		for (x = 0; x < recv_grains[X_COORD]; ++x) {
			MPI_Recv((void *)recv_buff, recv_grains[Y_COORD], MPI_FLOAT, proc,
					GRID_COMM + x, comm, &status);
			/* Print the buffer to the file. */
			for (y = 0; y < recv_grains[Y_COORD]; ++y)
				fprintf(fd, "%i %i %i %i %f\n", time, 0, (int)x + offset[X_COORD],
						(int)y + offset[Y_COORD], recv_buff[y]);

		}

   }

   fflush(fd);

   free(recv_buff);
}


int
getparams(int argc, char *argv[], pparams *params, FILE **gridfile, 
		FILE **statusfile, MPI_Datatype *pparams_dt, int rank)
{
	MPI_Aint pparams_displ[NUM_PARAMS];
	int	arg;

	/* Compute the displacements necessary to create a new MPI datatype. */
	pparams_displ[0] = (size_t)&(params->dx) - (size_t)params;
	pparams_displ[1] = (size_t)&(params->dt) - (size_t)params;
	pparams_displ[2] = (size_t)&(params->D) - (size_t)params;
	pparams_displ[3] = (size_t)&(params->ntotal) - (size_t)params;
	pparams_displ[4] = (size_t)&(params->ttotal) - (size_t)params;
	pparams_displ[5] = (size_t)&(params->l) - (size_t)params;
	pparams_displ[6] = (size_t)&(params->h) - (size_t)params;

	/* Create new MPI datatype. */
	MPI_Type_create_struct(NUM_PARAMS, 
			pparams_blength,
			pparams_displ,
			pparams_type,
			pparams_dt);

	MPI_Type_commit(pparams_dt);

	/* Only rank 0 has to parse the parameters. */
	if (rank > 0)
		return EX_OK;

	params->dx = -1;
	params->dt = -1;
	params->D = -1;
	params->l = 0;
	params->h = 0;
	*gridfile = NULL;

	while ((arg = getopt(argc, argv, "x:D:t:f:s:h:l:")) != -1) {
		switch (arg) {
			case 'x':
				params->dx = strtof(optarg, NULL);
				break;
			case 'D':
				params->D = strtof(optarg, NULL);
				break;
			case 't':
				params->dt = strtof(optarg, NULL);
				break;
			case 'f':
				if ((*gridfile = fopen(optarg, "w")) == NULL) 
					return EX_CANTCREAT;
				break;
			case 's':
				if ((*statusfile = fopen(optarg, "w+")) == NULL) 
					return EX_CANTCREAT;
				break;
			case 'l':
				params->l = (int)strtol(optarg, NULL, 10);
				break;
			case 'h':
				params->h = (int)strtol(optarg, NULL, 10);
				break;
			default:
				usage();
		}
	}
	argc -= optind;
	argv += optind;
	
	/* Although this could be computed every time, we prefer storing the values.
	 */
	params->ntotal = (int)(1 / params->dx);
	params->ttotal = (int)(1 / params->dt);

	/* Do some sanity check. */
	if (params->ntotal < 1 || params->ntotal < 1 || params->D < 0 || *gridfile
			== NULL || params->l == 0 || params->h == 0)
		usage();

	return EX_OK;
}

void
usage(void)
{
	fprintf(stderr, "diffusion -D <diffusion> -t <delta t> -x <delta x> \n");
	fprintf(stderr, "          -f <file> -s <file> -l <length>   \n");
	fprintf(stderr, "          -h <height>\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "Note that since a two dimensional grid is used the");
	fprintf(stderr, "number of nodes should always be h * l\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "-D <diffusion> the diffusion coefficient\n");
	fprintf(stderr, "-t <delta t>   the discretization step of the time\n");
	fprintf(stderr, "-x <delta x>   the discretization of the distance\n");
	fprintf(stderr, "-f <file>      file where the density profile is stored\n");
	fprintf(stderr, "-s <file>      file where the mpi status is store\n");
	fprintf(stderr, "-l <l decomp>  length of decomposition\n");
	fprintf(stderr, "-h <h decomp>  height of decomposition\n");

	MPI_Abort(MPI_COMM_WORLD, EX_USAGE);
	exit(EX_USAGE);
}
/* vim: set spell spelllang=en:cindent:tw=80:et*/

