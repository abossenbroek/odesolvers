/* 
 * This program solves the pde for the 2D diffusion equation using MPI-v2 and
 * SSE 1/2.
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
#include <assert.h>
#include <stdbool.h>

#include "tidiffusion.h"
#include "diffusion_help.h"

#if MPI_VERSION < 2
#   error "Need MPI version 2 for this program"
#endif

#ifndef NO_SEE
#   ifdef DOUBLE
#      include <emmintrin.h>
#   else
#      include <xmmintrin.h>
#   endif /* DOUBLE */
#endif /* NO_SSE */

void usage(void);

int getparams(int argc, char *argv[], pparams *params, FILE **gridfile,
      FILE **statusfile, MPI_Datatype *solve_params_dt, int rank);

int 
main(int argc, char *argv[])
{
   MPI_Comm comm = MPI_COMM_WORLD;  /* Communicator. */
   MPI_Datatype pparams_mpi;         /* Contains all the parameters. */
   MPI_Status   time_sts;
   int nnodes = 5;                  /* Total number of nodes. */
   int gsize[2] = {0};               /* Grid size. */
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

   double time_start_comm = 0;
   double time_start_comp = 0;
   double time_start_init = 0;
   double time_end_init = 0;
   double time_end_comm = 0;
   double time_end_comp = 0;
   double time_start_total = 0;
   double time_end_total = 0;
   double time_recv_buf;

   size_t yend;
   size_t ystart;
   
#ifndef NO_SSE
   grid_simd_type   sse_ratio;
   grid_simd_type   curr_grid;
   grid_simd_type   currr_grid;
   grid_simd_type   currl_grid;
   grid_simd_type   curru_grid;
   grid_simd_type   currd_grid;
   grid_simd_type   ngrid_sse;
#endif /* NO_SSE */

   grid_type **grid = NULL;
   grid_type **ngrid = NULL;
   grid_type *xdown = NULL;
   grid_type *xup = NULL;


   /* Arguments. */
   pparams params;

   FILE   *profilefile = NULL;
   FILE   *statusfile = NULL;

   size_t i;
   size_t x, y;
#ifndef NO_SSE
   size_t j;
   size_t y_qdl;
   size_t y_qdl_r;
#endif /* NO_SSE */
   long time = 0;
   bool is_steady = false; 
   bool steady_recv = false; 
   int proc;
   MPI_Request steady_comm;
   MPI_Status steady_status;

   MPI_Init(&argc, &argv);
   time_start_init = MPI_Wtime();
   time_start_total = MPI_Wtime();
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
   for (i = 0; i < (size_t)dims; i++) {
      grains[i] = params.ntotal / gsize[i] + (params.ntotal % gsize[i] +
            gsize[i] - coord[i] - 1) / gsize[i];

      if (grains[i] > (size_t)params.ntotal / gsize[i])
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
    */
   if ((grid = calloc(grains[X_COORD] + 2, sizeof(grid_type *))) == NULL ||
         (ngrid = calloc(grains[X_COORD] + 2, sizeof(grid_type *))) == NULL)
      MPI_Abort(comm, EX_OSERR);

   for (i = 0; i < grains[X_COORD] + 2; i++)
      if ((grid[i] = calloc(grains[Y_COORD] + 2, sizeof(grid_type))) == NULL ||
            (ngrid[i] = calloc(grains[Y_COORD] + 2, sizeof(grid_type))) == NULL)
         MPI_Abort(comm, EX_OSERR);

   /* Create temporary storage to prevent iterating through the entire grid. */
   if ((xdown = calloc(grains[X_COORD], sizeof(grid_type))) == NULL ||
         (xup = calloc(grains[X_COORD], sizeof(grid_type))) == NULL)
      MPI_Abort(comm, EX_OSERR);


#ifndef NO_SSE
#   ifdef DOUBLE
   sse_ratio = _mm_set1_pd(0.25);
#   else
   sse_ratio = _mm_set_ps1(0.25);

#   endif /* DOUBLE */
#endif /* NO_SSE */

   /* All the coordinates are translated to ranks by first computing the
    * coordinate of the appropriate neighbours. Then the coordinates are
    * used to determine the rank. These ranks can be used for the
    * communication. */
   coord_lneigh[X_COORD] = (coord[X_COORD] + gsize[X_COORD] - 1) % gsize[X_COORD];
   coord_rneigh[X_COORD] = (coord[X_COORD] + 1) % gsize[X_COORD];
   coord_lneigh[Y_COORD] = coord[Y_COORD];
   coord_rneigh[Y_COORD] = coord[Y_COORD];

   coord_dneigh[Y_COORD] = (coord[Y_COORD] + gsize[Y_COORD] - 1) % gsize[Y_COORD];
   coord_uneigh[Y_COORD] = (coord[Y_COORD] + 1) % gsize[Y_COORD];
   coord_dneigh[X_COORD] = coord[X_COORD];
   coord_uneigh[X_COORD] = coord[X_COORD];

   MPI_Cart_rank(comm, coord_lneigh, &rank_lneigh);
   MPI_Cart_rank(comm, coord_rneigh, &rank_rneigh);
   MPI_Cart_rank(comm, coord_dneigh, &rank_dneigh);
   MPI_Cart_rank(comm, coord_uneigh, &rank_uneigh);

   /* Compute by how much the loop iterators have to be adjusted. */
   yend = 1;
   ystart = 1;
   if (coord[Y_COORD] == (gsize[Y_COORD] - 1)) {
      yend--;
      for (x = 1; x < grains[X_COORD] + 1; ++x) 
         grid[x][grains[Y_COORD]] = 1;
   } 

   if (grains[Y_COORD] - yend - ystart < 1)
      MPI_Abort(MPI_COMM_WORLD, EX_USAGE);

   if (coord[Y_COORD] == 0)
      ystart++;

#ifndef NO_SSE
   /* Compute the loop start and end for the SSE instructions. */
   y_qdl =  (grains[Y_COORD] - ystart + yend) / SIMD_CAPACITY;
   y_qdl_r = (grains[Y_COORD] - ystart + yend) % SIMD_CAPACITY;
#endif /* NO_SSE */
   time_end_init = MPI_Wtime() - time_start_init;

   for (time = 0; !is_steady && time < params.ttotal; time++)
   {
      /* Create two new arrays to prevent bad memory access. */
      for (i = 0; i < grains[X_COORD]; i++) {
         xup[i] = grid[i + 1][grains[Y_COORD]];
         xdown[i] = grid[i + 1][1];
      }

      time_start_comm = MPI_Wtime();
      MPI_Send((void *)xdown, grains[X_COORD], MPI_GRID_TYPE, rank_dneigh, X_DOWN_TAG, comm);
      MPI_Send((void *)xup, grains[X_COORD], MPI_GRID_TYPE, rank_uneigh, X_UP_TAG, comm);

      MPI_Recv((void *)xdown, grains[X_COORD], MPI_GRID_TYPE, rank_dneigh,
            X_UP_TAG, comm, &xdown_status);
      MPI_Recv((void *)xup, grains[X_COORD], MPI_GRID_TYPE, rank_uneigh,
            X_DOWN_TAG, comm, &xup_status);

      time_end_comm += MPI_Wtime() - time_start_comm;

      /* The freshly received xup and xdown have to be put in the grid. */
      for (i = 0; i < grains[X_COORD]; i++) {
         grid[i + 1][grains[Y_COORD] + 1] = xup[i];
         grid[i + 1][0] = xdown[i];
      }

      time_start_comm = MPI_Wtime();
      MPI_Send((void *)(grid[grains[X_COORD]] + 1), grains[Y_COORD], MPI_GRID_TYPE,
            rank_rneigh, Y_RIGHT_TAG, comm);
      MPI_Send((void *)(grid[1] + 1), grains[Y_COORD], MPI_GRID_TYPE,
            rank_lneigh, Y_LEFT_TAG, comm);
      MPI_Recv((void *)(grid[0] + 1), grains[Y_COORD], MPI_GRID_TYPE,
            rank_lneigh, Y_RIGHT_TAG, comm, &yright_status);
      MPI_Recv((void *)(grid[grains[X_COORD] + 1] + 1), grains[Y_COORD],
            MPI_GRID_TYPE, rank_rneigh, Y_LEFT_TAG, comm, &yleft_status);

      time_end_comm += MPI_Wtime() - time_start_comm;

      /* Do a non blocking send of the current grid for printing. */
      if ((time % params.freq) == 0) 
         send_grid(grid, grains, offset, rank, comm, &time_end_comm, PRINT_COMM);

      time_start_comp = MPI_Wtime();
      is_steady = true;
      for (x = 1; x < grains[X_COORD] + 1; x++) {
#ifdef NO_SSE
         for (y = ystart; y < grains[Y_COORD] + yend; y++) {
            /* Do the finite difference computation. */
            ngrid[x][y] = 0.25 * (grid[x][y + 1] + grid[x][y - 1]
                  + grid[x + 1][y] + grid[x - 1][y]);
            
            /* The test is always done although it is not used during each
             * iteration.  This is done to prevent overhead caused by the
             * additional test. */
#   ifdef DOUBLE
            is_steady &= (fabs(ngrid[x][y] - grid[x][y]) < STEADY_TOLERANCE);
#   else
            is_steady &= (fabsf(ngrid[x][y] - grid[x][y]) < STEADY_TOLERANCE);
#   endif /* DOUBLE */
         }
#else
         for (i = 0, y = ystart; i < y_qdl; ++i, y += SIMD_CAPACITY) {
#   ifdef DOUBLE
            /* Load all the necessary values to  SSE2 variables. */
            /* r1 = (x, y + 1)
             * r0 = (x, y)
             */
            curr_grid = _mm_loadu_pd(grid[x] + y);
            /* rr0 = (x + 1, y + 1)
             * rr0 = (x + 1, y) */
            currr_grid = _mm_loadu_pd(grid[x + 1] + y);
            /* rl0 = (x - 1, y + 1)
             * rl0 = (x - 1, y) */
            currl_grid = _mm_loadu_pd(grid[x - 1] + y);
            /* ru0 = (x, y + 2)
             * ru0 = (x, y + 1) */
            curru_grid = _mm_loadu_pd(grid[x] + y + 1);
            /* rd0 = (x, y)
             * rd0 = (x, y - 1) */
            currd_grid = _mm_loadu_pd(grid[x] + y - 1);

            /* Perform arithmetic in an order which should reduce the number of
             * bubbles in the processor pipeline. */
            /* rr = rr + rl */
            currr_grid = _mm_add_pd(currr_grid, currl_grid);
            /* ru = ru + rd */
            curru_grid = _mm_add_pd(curru_grid, currd_grid);
            currr_grid = _mm_add_pd(currr_grid, curru_grid);
            /* rr += ratio */
            ngrid_sse = _mm_mul_pd(currr_grid, sse_ratio);
            /* (x, y + 1) = nr1
             * (x, y) = nr0 */
            _mm_storeu_pd(ngrid[x] + y, ngrid_sse);
            curr_grid = _mm_sub_pd(ngrid_sse, curr_grid);

            if (time % params.freq == 0) 
               for (j = 0; j < SIMD_CAPACITY; j++)
                  is_steady &= (fabs(((double*)&curr_grid)[j]) <
                        STEADY_TOLERANCE);
            else
               is_steady &= false;
#   else
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
            currr_grid = _mm_add_ps(currr_grid, curru_grid);
            ngrid_sse = _mm_mul_ps(currr_grid, sse_ratio);
            _mm_storeu_ps(ngrid[x] + y, ngrid_sse);

            curr_grid = _mm_sub_ps(ngrid_sse, curr_grid);

            if (time % params.freq == 0) 
               for (j = 0; j < SIMD_CAPACITY; j++)
                  is_steady &= (fabsf(((float*)&curr_grid)[j]) <
                        STEADY_TOLERANCE);
            else
               is_steady &= false;
#   endif /* DOUBLE */
         }

         for (i = 0; i < y_qdl_r; ++i) {
            ngrid[x][y] = 0.25 * (grid[x][y + 1] + grid[x][y - 1]
                  + grid[x + 1][y] + grid[x - 1][y]);
#   ifdef DOUBLE
            is_steady &= (fabs(ngrid[x][y] - grid[x][y]) < STEADY_TOLERANCE);
#   else
            is_steady &= (fabsf(ngrid[x][y] - grid[x][y]) < STEADY_TOLERANCE);
#   endif /* DOUBLE */
            y++;
         }
#endif /* NO_SSE */
      }
      time_end_comp += MPI_Wtime() - time_start_comp;

      if (time % params.freq == 0)
         recv_grid(grid, grains, offset, time, rank, nnodes,
               comm, &time_end_comm, PRINT_COMM, &print_elem,
               (void *)profilefile);

      /* Copy the new grid to the current grid. Use the previously computed
       * y-offsets to determine where copying should start. */
      for (x = 1; x < grains[X_COORD] + 1; ++x) 
         memcpy((void *)(grid[x] + ystart), (void *)(ngrid[x] + ystart),
               (grains[Y_COORD] - (ystart - yend)) * sizeof(grid_type));

      if (time % params.freq == 0) {
         if (rank != 0) {
            /* Send details on the steady state of this node to the root node.
             */
            MPI_Isend((void *)&is_steady, 1, MPI_CHAR, 0, STEADY_TAG, comm,
                  &steady_comm);
         } else {
            /* Receive all the steady status of the nodes and OR this with the
             * status on this node. */
            for (proc = 1; proc < nnodes; proc++) {
               MPI_Recv((void *)&steady_recv, 1, MPI_CHAR, proc, STEADY_TAG,
                     comm, &steady_status);
               is_steady &= steady_recv;
            }
         }
         MPI_Bcast((void*)&is_steady, 1, MPI_CHAR, 0, comm);
      }

      /* Ensure that all the processes are at the same point. */
      MPI_Barrier(comm);
   }

   /* Free the memory used for the grid. */
   for (i = 0; i < grains[X_COORD] + 2; i++) {
      free(grid[i]);
      free(ngrid[i]);
   }

   free(grid);
   free(ngrid);
   free(xup);
   free(xdown);

   if (rank != 0) {
      MPI_Send(&time_end_comm, 1, MPI_DOUBLE, 0, TIME_COMM_TAG, MPI_COMM_WORLD);
      MPI_Send(&time_end_comp, 1, MPI_DOUBLE, 0, TIME_COMP_TAG, MPI_COMM_WORLD);
      MPI_Send(&time_end_init, 1, MPI_DOUBLE, 0, TIME_INIT_TAG, MPI_COMM_WORLD);
   } 

   /* Get all the information on the running time. */
   if (rank == 0) {
      for (i = 1; i < (size_t)nnodes; ++i) {
         MPI_Recv(&time_recv_buf, 1, MPI_DOUBLE, i, TIME_COMM_TAG,
               MPI_COMM_WORLD, &time_sts);
         time_end_comm += time_recv_buf;

         MPI_Recv(&time_recv_buf, 1, MPI_DOUBLE, i, TIME_COMP_TAG,
               MPI_COMM_WORLD, &time_sts);
         time_end_comp += time_recv_buf;

         MPI_Recv(&time_recv_buf, 1, MPI_DOUBLE, i, TIME_INIT_TAG,
               MPI_COMM_WORLD, &time_sts);
         time_end_init += time_recv_buf;
      }
      if (statusfile != NULL) {
         time_end_total = MPI_Wtime() - time_start_total;
         fprintf(statusfile, "%s %i %i %i %li %lf %lf %lf %lf %i %li\n",
               argv[0], nnodes, gsize[X_COORD], gsize[Y_COORD],
               sizeof(grid_type), time_end_total, time_end_comp, time_end_init,
               time_end_comm, (int)is_steady, time);
         fclose(statusfile);
      }

      fclose(profilefile);
   }


   MPI_Finalize();
   return EX_OK;
}

int
getparams(int argc, char *argv[], pparams *params, FILE **gridfile, 
      FILE **statusfile, MPI_Datatype *pparams_dt, int rank)
{
   MPI_Aint pparams_displ[NUM_PARAMS];
   int   arg;

   /* Compute the displacements necessary to create a new MPI datatype. */
   pparams_displ[0] = (size_t)&(params->dx) - (size_t)params;
   pparams_displ[1] = (size_t)&(params->dt) - (size_t)params;
   pparams_displ[2] = (size_t)&(params->ntotal) - (size_t)params;
   pparams_displ[3] = (size_t)&(params->ttotal) - (size_t)params;
   pparams_displ[4] = (size_t)&(params->l) - (size_t)params;
   pparams_displ[5] = (size_t)&(params->h) - (size_t)params;
   pparams_displ[6] = (size_t)&(params->freq) - (size_t)params;

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
   params->l = 0;
   params->h = 0;
   params->freq = -1;
   *gridfile = NULL;

   while ((arg = getopt(argc, argv, "x:t:f:s:h:l:g:")) != -1) {
      switch (arg) {
         case 'x':
            params->dx = (grid_type)strtof(optarg, NULL);
            break;
         case 't':
            params->dt = (grid_type)strtof(optarg, NULL);
            break;
         case 'g':
            if ((*gridfile = fopen(optarg, "w")) == NULL) 
               return EX_CANTCREAT;
            break;
         case 's':
            if ((*statusfile = fopen(optarg, "a+")) == NULL) 
               return EX_CANTCREAT;
            break;
         case 'l':
            params->l = (int)strtol(optarg, NULL, 10);
            break;
         case 'h':
            params->h = (int)strtol(optarg, NULL, 10);
            break;
         case 'f':
            params->freq = (int)strtol(optarg, NULL, 10);
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
   if (params->ntotal < 1) {
      warnx("ntotal > 1");
      usage();
   }
   if (*gridfile == NULL) {
      warnx("Could not open a file to store grid points.");
      usage();
   }
   if (params->l == 0 || params->h == 0) {
      warnx("please specify the processor dimensions of the Grid.");
      usage();
   }
   if (params->freq < 0) {
      warnx("frequency >= 0");
      usage();
   }


   return EX_OK;
}

void
usage(void)
{
   fprintf(stderr, "tidiffusion -t <delta t> -x <delta x>");
   fprintf(stderr, " -g <file> -f <freq> -s <file> -l <length>\n");
   fprintf(stderr, "          -h <height>\n");
   fprintf(stderr, "\n");
   fprintf(stderr, "Note that since a two dimensional grid is used the ");
   fprintf(stderr, "number of nodes should always be h * l\n");
   fprintf(stderr, "\n");
   fprintf(stderr, "-t <delta t>   the discretization step of the time\n");
   fprintf(stderr, "-x <delta x>   the discretization of the distance\n");
   fprintf(stderr, "-g <file>      file where the density profile is stored\n");
   fprintf(stderr, "-f <freq>      frequency of density store operations\n");
   fprintf(stderr, "-s <file>      file where the mpi status is store\n");
   fprintf(stderr, "-l <l decomp>  length of decomposition\n");
   fprintf(stderr, "-h <h decomp>  height of decomposition\n");

   MPI_Abort(MPI_COMM_WORLD, EX_USAGE);
   exit(EX_USAGE);
}

// vim:spell spelllang=en:cindent:tw=80:et:

