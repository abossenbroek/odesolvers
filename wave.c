/* 
 * This program solves the pde for the wave equation using MPI-v2 and SSE2.
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
#include <sysexits.h>
#include <string.h>
#include <getopt.h>
#include <err.h>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#define __USE_BSD
#include <math.h>

#include "wave.h"

#ifndef NO_SSE2
   /* Needed for SSE2 */
#  ifndef __SSE2__
#     error	"SSE2 needs to be enabled. Use -msse -msse2 as gcc flags."
#  endif
#  include <emmintrin.h>
#endif

#if MPI_VERSION < 2
#	error "Need MPI version 2 for this program"
#endif


void usage(void);

/*
 * This function uses a blocking receive to receive all the information from
 * all the nodes with rank > 0 in the communicator. It saves the information
 * in the file fd.
 */
void print_wave(FILE *fd, double *pde, int grains, int offset, int time, int
      rank, int nnodes, MPI_Comm comm, double *time_comm);

/*
 * This function sends all the information of the node using a non blocking
 * send. It is only executed on nodes with a rank higher than zero.
 */
void send_wave(double *pde, int grains, int offset, int rank,
      MPI_Request *grain_send, MPI_Request *offset_send, MPI_Request
      *pde_send, MPI_Comm comm, double *time_comm);

int getparams(int argc, char *argv[], solve_params *params, FILE **wavefile,
		FILE **satusfile, MPI_Datatype *solve_params_dt, int rank);

int 
main(int argc, char *argv[])
{
   /* MPI variables. */
   MPI_Comm comm = MPI_COMM_WORLD;
   int nnodes = 5;
   int periods = false;
   int rank;
   MPI_Request grain_snd;
   MPI_Request offset_snd;
   MPI_Request pde_snd;
   MPI_Status  neigh_sts[2];
   MPI_Datatype solve_params_mpi;
	MPI_Status	time_sts;
   int status;
	int xcoord_lneigh;
	int xcoord_rneigh;
	int rank_lneigh;
	int rank_rneigh;
#ifndef NO_SSE2
   int dbl_iter, dbl_iter_r;
   __m128d pde_n_simd; 
   __m128d pde_c_simd_xl;
   __m128d pde_c_simd_x;
   __m128d pde_c_simd_xu;
   __m128d c1_simd;
   __m128d c2_simd;
   __m128d tmp_simd;
#endif /* NO_SSE2 */

   double time_start_comm = 0;
   double time_start_comp = 0;
   double time_start_init = 0;
   double time_end_init = 0;
   double time_end_comm = 0;
   double time_end_comp = 0;
	double time_start_total = 0;
	double time_end_total = 0;
	double time_recv_buf;

   /* Define the dimension of the Cartesian grid. As we want to investigate the
    * wave equation in one dimension. */
   const int dims = 1;
   int xcoord = 0;

   /* Arguments. */
   solve_params params;
   FILE	*wavefile = NULL;
	FILE  *statusfile = NULL;
   /* Initialise the values to be able to detect errors in the parameter
    * specification. */
   params.location = -1;
   params.delta = -1;
   params.periods = -1;
   params.freq = 0;
   params.method = NONE;
   params.ntotal = -1;
   params.tau = -1.0;

   /* Local information. */
   int	 grains;
   int	 offset;
   double *pde_o;				/* Old values. */
   double *pde_c;				/* Current values. */
   double *pde_n;				/* New values. */
   double c;
   int    start;
   int    end;

   size_t i;


   MPI_Init(&argc, &argv);
   time_start_init = MPI_Wtime();
	time_start_total = MPI_Wtime();
   /* Find the rank of this processors. */
	MPI_Comm_rank(comm, &rank);

	if ((status = getparams(argc, argv, &params, &wavefile, &statusfile,
					&solve_params_mpi, rank)) != EX_OK)
      MPI_Abort(comm, status);

   /* Send all the parameters to the remaining nodes in the comm. */
   MPI_Bcast(&params, 1, solve_params_mpi, 0, MPI_COMM_WORLD);

   /* Determine the number of nodes in the communicator. */
   MPI_Comm_size(comm, &nnodes);

   /* Create a get information of a Cartesian grid topology. */
   if (MPI_Cart_create(comm, dims, &nnodes, &periods, true, &comm) != 
         MPI_SUCCESS) 
      MPI_Abort(comm, EX_UNAVAILABLE);

   /* Translate the current rank to the coordinate in the Cartesian topology.
   */
   MPI_Cart_coords(comm, rank, dims, &xcoord);

   /* Using the coordinate of the current node we can determine the amount of
    * points this node has to compute. */
   grains = params.ntotal / nnodes + (params.ntotal % nnodes + nnodes - xcoord
         - 1) / nnodes;

   /* Compute the offset necessary to plot the results and initialize the pde
    * solver. */
   if (grains > params.ntotal / nnodes)
      offset = (params.ntotal / nnodes + 1) * xcoord;
   else
      offset = params.ntotal / nnodes * xcoord + params.ntotal % nnodes;

   /* Allocate memory for the points. */
   if (((pde_o = calloc(grains + 2, sizeof(double))) == NULL) ||
         ((pde_c = calloc(grains + 2, sizeof(double))) == NULL) ||
         ((pde_n = calloc(grains + 2, sizeof(double))) == NULL)) {
      MPI_Abort(comm, EX_OSERR);
      errx(EX_OSERR, "Could not allocate any memory for pde values.");
   }

   /* Using the offset and the grains to initialize the pde_c. */
   double pos;
   switch (params.method) {
      case SINE:
         for (i = offset; i < (offset + grains); i++) {
            pos = (double)i / (double)(params.ntotal - 1);
            /* Use a double sine to initialize the value of the string. */
            pde_c[i - offset + 1] = sin(M_PI * pos) + sin(M_PI * pos * 2);
            //pde_c[i - offset + 1] = sin(2 * M_PI * pos);
            pde_o[i - offset + 1] = pde_c[i - offset + 1];
         }
         break;

      case PLUCKED:
         for (i = offset; i < (offset + grains); i++) {
            pos = (double)i / (double)(params.ntotal - 1);
            /* Compute the initial value of the string when plucked. */
            if (i < (int)(params.location * (double)params.ntotal))
               pde_c[i - offset + 1] = params.height / params.location * pos;
            else 
               pde_c[i - offset + 1] = -params.height / (1.0 - params.location) * pos +
                  params.height / (1.0 - params.location);
            pde_o[i - offset + 1] = pde_c[i - offset + 1];
         }
         break;
      default:
         MPI_Abort(comm, EX_SOFTWARE);
         errx(EX_SOFTWARE, "Illegal initialization method %i", params.method);
   }

   /* Factor to use in finite difference stencil:
    * factor = delta_time^2 * tau^2 / delta_distance^2 
    */
   if (params.tau > 0)
      c = params.delta * params.delta * params.tau * params.tau;
   else
      c = 1;

#ifndef NO_SSE2
   /* Store two c factors needed for the finite difference approximation. */
   c1_simd = _mm_set1_pd(c);
   c2_simd = _mm_set1_pd(2.0 - 2.0 * c);
#endif /* NO_SSE2 */

   /* Since the end points are rigid the boundaries are not computed. */
   start = 1;
   end = 1;
   if (xcoord == 0)
      start++;
   if (xcoord == nnodes - 1)
      end--;


#ifndef NO_SSE2
   dbl_iter = ((grains  - start + end)) / 2;
   dbl_iter_r = (grains  - start + end) % 2;
#endif /* NO_SSE2 */

	/* Send and receive neighbours using non blocking communication. */
	xcoord_lneigh = (nnodes + xcoord - 1) % nnodes;
	xcoord_rneigh = (xcoord + 1) % nnodes;

	MPI_Cart_rank(comm, &xcoord_lneigh, &rank_lneigh);
	MPI_Cart_rank(comm, &xcoord_rneigh, &rank_rneigh);

   time_end_init = MPI_Wtime() - time_start_init;

   for (int time = 0; time < (int)(params.stime / params.delta); time++) {
      time_start_comm = MPI_Wtime();
     		
		MPI_Send((void *)&(pde_c[1]), 1, MPI_DOUBLE, rank_lneigh, LVAL_COMM, comm);
		MPI_Send((void *)&(pde_c[grains]), 1, MPI_DOUBLE, rank_rneigh, RVAL_COMM, comm);
		MPI_Recv((void *)&(pde_c[grains + 1]), 1, MPI_DOUBLE, rank_rneigh,
				LVAL_COMM, comm, &(neigh_sts[0]));
		MPI_Recv((void *)&(pde_c[0]), 1, MPI_DOUBLE, rank_lneigh,
				RVAL_COMM, comm, &(neigh_sts[1]));
      
		time_end_comm += MPI_Wtime() - time_start_comm;
      /* Send all the wave information to the root. This send is non blocking. */
      if ((time % params.freq) == 0)
         send_wave(pde_c, grains, offset, rank, &grain_snd, &offset_snd,
               &pde_snd, comm, &time_end_comm);
		

      /* Ensure that before the actual computation starts all the processes are
       * at this point in the program. */
      MPI_Barrier(comm);

      time_start_comp = MPI_Wtime();

#ifdef NO_SSE2
      for (i = start; i < grains + end; i++)  {
         /* Perform an approximation of the wave equation using finite
          * difference. */
			pde_n[i] = (2 - 2 * c) * pde_c[i] + c * (pde_c[i - 1] + pde_c[i + 1])
				- pde_o[i];
      }
#else
      size_t j = start;
      for (i = 0; i < dbl_iter; ++i) {
         /* Load all the values from the current pde approximation into the
          * SSE2 registers. 
          * _mm_set_pd(double w, double x) loads w to r0 and x to r1 
          *
          * The computations are the same as in the previous loop. */
			pde_c_simd_xl = _mm_loadu_pd(pde_c + j - 1);
			pde_c_simd_xu = _mm_loadu_pd(pde_c + j + 1);
         /* Load the old values immediately to the new values. */
         pde_n_simd = _mm_set_pd(-pde_o[j + 1], -pde_o[j]);
         tmp_simd = _mm_add_pd(pde_c_simd_xl, pde_c_simd_xu);
         
         /* Permit the last operation to come out of the processor pipeline by
          * first loading the current pde values. */
         pde_c_simd_x = _mm_set_pd(pde_c[j + 1], pde_c[j]);
         tmp_simd = _mm_mul_pd(c1_simd, tmp_simd);
         pde_c_simd_x = _mm_mul_pd(c2_simd, pde_c_simd_x);
         pde_n_simd = _mm_add_pd(tmp_simd, pde_n_simd);
         pde_n_simd = _mm_add_pd(pde_c_simd_x, pde_n_simd);
         /* Store the computed values to the array representing the new pde
          * estimation. */
         _mm_storeu_pd(pde_n + j, pde_n_simd);
         j += 2;
      }

      for (i = 0; i < dbl_iter_r; ++i)  {
         /* Perform an approximation of the wave equation using finite
          * difference. */
			pde_n[j] = (2 - 2 * c) * pde_c[j] + c * (pde_c[j - 1] + pde_c[j + 1])
				- pde_o[j];
			++j;
      }
#endif /* NO_SSE2 */

      time_end_comp += MPI_Wtime() - time_start_comp;

      /* Aggregate all the information on the wave at the root. */
      if ((time % params.freq) == 0)
         print_wave(wavefile, pde_c, grains, offset, time, rank, nnodes, comm,
               &time_end_comm);

      /* Copy the current pde. */
		memcpy((void *)(pde_o + 1), (void *)(pde_c + 1), grains * sizeof(double));
		memcpy((void *)(pde_c + 1), (void *)(pde_n + 1), grains * sizeof(double));
   }

	if (rank != 0) {
		MPI_Send(&time_end_comm, 1, MPI_DOUBLE, 0, TIME_COMM_TAG, MPI_COMM_WORLD);
		MPI_Send(&time_end_comp, 1, MPI_DOUBLE, 0, TIME_COMP_TAG, MPI_COMM_WORLD);
		MPI_Send(&time_end_init, 1, MPI_DOUBLE, 0, TIME_INIT_TAG, MPI_COMM_WORLD);
	} 

	if (rank == 0) {
		for (i = 1; i < nnodes; ++i) {
			MPI_Recv(&time_recv_buf, 1, MPI_DOUBLE, i, TIME_COMM_TAG, MPI_COMM_WORLD, &time_sts);
			time_end_comm += time_recv_buf;

			MPI_Recv(&time_recv_buf, 1, MPI_DOUBLE, i, TIME_COMP_TAG, MPI_COMM_WORLD, &time_sts);
			time_end_comp += time_recv_buf;

			MPI_Recv(&time_recv_buf, 1, MPI_DOUBLE, i, TIME_INIT_TAG, MPI_COMM_WORLD, &time_sts);
			time_end_init += time_recv_buf;
		}
		if (statusfile != NULL) {
			time_end_total = MPI_Wtime() - time_start_total;
			fprintf(statusfile, "%i %lf %lf %lf %lf\n", nnodes, time_end_total,
					time_end_comp, time_end_init, time_end_comm);
			fclose(statusfile);
		}
		fclose(wavefile);
	}
   free(pde_o);
   free(pde_c);
   free(pde_n);

   MPI_Finalize();
   return EX_OK;
}

void
usage(void)
{
   fprintf(stderr, "string [-l <location> | -p <periods>] -h <height> \n");
   fprintf(stderr, "       -d <delta> -f <freq> -t <time> -n <ntotal>\n");
   fprintf(stderr, "       -w <file> -s <file>\n");
   fprintf(stderr, "-l <location> pluck string at location <location>\n");
   fprintf(stderr, "-h <height>   the height of the pluck or sine\n");
   fprintf(stderr, "-p <periods>  set the periods of the sinus on <periods>\n");
   fprintf(stderr, "-d <delta>    grain size of time discretization\n");
   fprintf(stderr, "-f <freq>     plot every <freq>\n");
   fprintf(stderr, "-t <time>     seconds to simulation\n");
   fprintf(stderr, "-n <ntotal>   grain size of length discretization\n");
   fprintf(stderr, "-w <file>     file to store wave information\n");
   fprintf(stderr, "-s <file>     status file where to store mpi times\n");
   MPI_Abort(MPI_COMM_WORLD, EX_USAGE);
   exit(EX_USAGE);
}

void
send_wave(double *pde, int grains, int offset, int rank, 
      MPI_Request *grain_send, MPI_Request *offset_send, MPI_Request *pde_send,
      MPI_Comm comm, double *time_comm)
{
   double time_comm_start;

   if (rank != 0) {
		time_comm_start = MPI_Wtime();
      MPI_Isend((void *)&grains, 1, MPI_INT, 0, GRAIN_COMM, comm,
            grain_send);
      MPI_Isend((void *)&offset, 1, MPI_INT, 0, OFFSET_COMM, comm,
            offset_send);
      /* Increase the address of the buffer by one to prevent sending the
       * boundaries used to compute the stencil. */
      MPI_Isend((void *)(pde + 1), grains, MPI_DOUBLE, 0, 
            PDE_COMM, comm, pde_send);
		*time_comm += MPI_Wtime() - time_comm_start;
   }
}

void 
print_wave(FILE *fd, double *pde, int grains, int offset, int time, int rank,
      int nnodes, MPI_Comm comm, double *time_comm) {
   /* Only rank 0 has to perform this function. */
   if (rank > 0)
      return;

   int i = 0;
   double *recv_buff;
   int recv_grains;
   int recv_offset;
   MPI_Status status;
   double time_comm_start;

   /* When load balancing the node with rank 0 will always have the largest
    * amount of grains to compute. Therefore the number of grains which are
    * to be computed by the root can also be used as buffer size. */
   if ((recv_buff = calloc(grains, sizeof(double))) == NULL) {
      MPI_Abort(MPI_COMM_WORLD, EX_OSERR);
      errx(EX_OSERR, "No memory for receive buffer.");
   }
   if (fd == NULL)
      MPI_Abort(MPI_COMM_WORLD, EX_SOFTWARE);

   /* Print all the values computed by the node with rank 0 */
   for (i = 0; i < grains; i++) 
      fprintf(fd, "%i %i %i %lf\n", time, 0, i, pde[i + 1]);

   /* Print all the values computed by the nodes with rank > 0. These 
    * values have to be received from the other nodes. */
   for (int proc = 1; proc < nnodes; proc++) {
      time_comm_start = MPI_Wtime();
      /* Perform blocking receives from the non blocking sends. */
      MPI_Recv((void *)&recv_grains, 1, MPI_INT, proc, GRAIN_COMM, comm,
            &status);
      MPI_Recv((void *)&recv_offset, 1, MPI_INT, proc, OFFSET_COMM, comm,
            &status);
      MPI_Recv((void *)recv_buff, recv_grains, MPI_DOUBLE, proc, PDE_COMM, comm,
            &status);
      *time_comm += MPI_Wtime() - time_comm_start;

      /* Print the buffer to the file. */
      for (i = 0; i < recv_grains; i++)
         fprintf(fd, "%i %i %i %lf\n", time, proc, i + recv_offset, recv_buff[i]);
   }

   fflush(fd);

   free(recv_buff);
}

int
getparams(int argc, char *argv[], solve_params *params, FILE **wavefile,
		FILE **statusfile, MPI_Datatype *solve_params_dt, int rank)
{
   MPI_Aint solve_params_displ[NUM_PARAMS];
   int	arg;

   solve_params_displ[0] = (size_t)&(params->method) - (size_t)params;
   solve_params_displ[1] = (size_t)&(params->periods) - (size_t)params;
   solve_params_displ[2] = (size_t)&(params->location) - (size_t)params;
   solve_params_displ[3] = (size_t)&(params->stime) - (size_t)params;
   solve_params_displ[4] = (size_t)&(params->delta) - (size_t)params;
   solve_params_displ[5] = (size_t)&(params->ntotal) - (size_t)params;
   solve_params_displ[6] = (size_t)&(params->freq) - (size_t)params;
   solve_params_displ[7] = (size_t)&(params->height) - (size_t)params;
   solve_params_displ[8] = (size_t)&(params->tau) - (size_t)params;

   MPI_Type_create_struct(NUM_PARAMS, 
         solve_params_blength,
         solve_params_displ,
         solve_params_type,
         solve_params_dt);

   MPI_Type_commit(solve_params_dt);

   if (rank > 0)
      return EX_OK;

   while ((arg = getopt(argc, argv, "p:d:l:f:t:n:h:w:c:s:")) != -1) {
      switch (arg) {
         case 'l':
            params->location = strtod(optarg, NULL);
            params->method = PLUCKED;
            break;
         case 'p':
            params->periods = (int)strtol(optarg, NULL, 10);
            params->method = SINE;
            break;
         case 'd':
            params->delta = strtod(optarg, NULL);
            break;
         case 'f':
            params->freq = (int)strtol(optarg, NULL, 10);
            break;
         case 't':
            params->stime = strtod(optarg, NULL);
            break;
         case 'n':
            params->ntotal = (int)strtol(optarg, NULL, 10);
            break;
         case 'h':
            params->height = strtod(optarg, NULL);
            break;
         case 'w':
            if ((*wavefile = fopen(optarg, "w")) == NULL) 
               return EX_CANTCREAT;
            break;
         case 'c':
            params->tau = strtod(optarg, NULL);
            break;
			case 's':
            if ((*statusfile = fopen(optarg, "a")) == NULL) 
               return EX_CANTCREAT;
            break;
         default:
            usage();
      }
   }
   argc -= optind;
   argv += optind;

   /* Perform sanity check on the variables-> */
   if (params->delta < 0 || 
         (params->location > 0 && params->periods > 0) || 
         (params->location < 0 && params->periods < 0) ||
         (params->ntotal < 2) ||
			(params->stime / params->delta < 1)) {
      fprintf(stderr, "Illegal parameters->\n");
      usage();
   }

   return EX_OK;
}

/* vim: set spell spelllang=en:cindent:tw=79:et*/

