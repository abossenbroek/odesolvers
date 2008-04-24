#include <mpi.h>
#include <math.h>
#include <sysexits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include <unistd.h>

#include "diffusion.h"

void usage(void);

int getparams(int argc, char *argv[], pparams *params, FILE **wavefile,
		FILE **statusfile, MPI_Datatype *solve_params_dt, int rank);

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
	int grains[2];
	 

	/* Arguments. */
	pparams params;

	FILE	*profilefile = NULL;
	FILE	*statusfile = NULL;


	size_t i;


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
	if (rank == 0 && nnodes / params.x != params.y) {
		usage(); 
	}

	/* Compute the grid form. */
	gsize[X_COORD] = params.x;
	gsize[Y_COORD] = params.y;

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


	warnx("rank %i (%i, %i) grain (%i, %i) offset (%i, %i)", rank, coord[X_COORD], coord[Y_COORD],
			grains[0], grains[1], offset[0], offset[1]);
	
	
	if (rank == 0) {
		fclose(statusfile);
		fclose(profilefile);
	}


	MPI_Finalize();
	return EX_OK;
}

int
getparams(int argc, char *argv[], pparams *params, FILE **wavefile, 
		FILE **statusfile, MPI_Datatype *pparams_dt, int rank)
{
	MPI_Aint pparams_displ[NUM_PARAMS];
	int	arg;

	pparams_displ[0] = (size_t)&(params->dx) - (size_t)params;
	pparams_displ[1] = (size_t)&(params->dt) - (size_t)params;
	pparams_displ[2] = (size_t)&(params->D) - (size_t)params;
	pparams_displ[3] = (size_t)&(params->ntotal) - (size_t)params;
	pparams_displ[4] = (size_t)&(params->ttotal) - (size_t)params;
	pparams_displ[5] = (size_t)&(params->x) - (size_t)params;
	pparams_displ[6] = (size_t)&(params->y) - (size_t)params;

	MPI_Type_create_struct(NUM_PARAMS, 
			pparams_blength,
			pparams_displ,
			pparams_type,
			pparams_dt);

	MPI_Type_commit(pparams_dt);

	if (rank > 0)
		return EX_OK;

	params->dx = -1;
	params->dt = -1;
	params->D = -1;
	params->x = 0;
	params->y = 0;
	*wavefile = NULL;

	while ((arg = getopt(argc, argv, "x:D:t:f:s:h:l:")) != -1) {
		switch (arg) {
			case 'x':
				params->dx = strtod(optarg, NULL);
				break;
			case 'D':
				params->D = strtod(optarg, NULL);
				break;
			case 't':
				params->dt = strtod(optarg, NULL);
				break;
			case 'f':
				if ((*wavefile = fopen(optarg, "w")) == NULL) 
					return EX_CANTCREAT;
				break;
			case 's':
				if ((*statusfile = fopen(optarg, "w+")) == NULL) 
					return EX_CANTCREAT;
				break;
			case 'l':
				params->x = (int)strtol(optarg, NULL, 10);
				break;
			case 'h':
				params->y = (int)strtol(optarg, NULL, 10);
				break;
			default:
				usage();
		}
	}
	argc -= optind;
	argv += optind;
	
	params->ntotal = (int)(1 / params->dx);
	params->ttotal = (int)(1 / params->dt);

	/* Do some sanity check. */
	if (params->ntotal < 1 || params->ntotal < 1 || params->D < 0 || *wavefile
			== NULL || params->x == 0 || params->y == 0)
		usage();



	return EX_OK;
}

void
usage(void)
{
	fprintf(stderr, "diffusion -D <diffusion> -t <delta t> -x <delta x> ");
	fprintf(stderr, " -f <file>\n");
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

