#include <stdlib.h>
#include <stdio.h>
#include <sysexits.h>
#include <err.h>

#include "mpi.h"
#include "diffusion_help.h"

   void
print_elem(long time, int rank, int x, size_t* grains, int* offset, 
      grid_type *column, void *fd)
{
   size_t y = 0;

   for (y = 0; y < grains[Y_COORD]; ++y) {
#ifndef DOUBLE
      fprintf((FILE *)fd, "%li %i %i %i %f\n", time, rank, x, ((int)y + offset[Y_COORD]), column[y]);
#else
      fprintf((FILE *)fd, "%li %i %i %i %lf\n", time, rank, x, ((int)y + offset[Y_COORD]), column[y]);
#endif /* DOUBLE */
   }
   fflush((FILE *)fd);
}

void
send_grid(grid_type **grid, size_t *grains, int *offset, int rank, 
      MPI_Comm comm, double *time_comm, int base_tag)
{
   if (rank == 0)
      return;

   double time_comm_start;
   size_t i;

   /* Send all the necessary data using a non blocking send. */
   time_comm_start = MPI_Wtime();
   MPI_Send((void *)grains, 2, MPI_INT, 0, base_tag, comm);
   MPI_Send((void *)offset, 2, MPI_INT, 0, base_tag + 1, comm);

   for (i = 1; i < grains[X_COORD] + 1; i++)  
      MPI_Send((void *)(grid[i] + 1), grains[Y_COORD], MPI_GRID_TYPE, 0,
            base_tag + i + 1, comm);

   *time_comm += MPI_Wtime() - time_comm_start;
}

void 
recv_grid(grid_type **grid, size_t *grains, int *offset, long time, int rank,
      int nnodes, MPI_Comm comm, double *time_comm, int base_tag, void
      (*handler)(long, int, int, size_t*, int*, grid_type*, void *), void
      *handlerargs) 
{
   /* Only rank 0 has to perform this function. */
   if (rank > 0)
      return;

   size_t x = 0;
   grid_type *recv_buff;
   size_t recv_grains[2];
   int recv_offset[2];
   MPI_Status status;
   double time_comm_start;
   int proc;

   /* When load balancing the node with rank 0 will always have the largest
    * amount of grains to compute. Therefore the number of grains which are
    * to be computed by the root can also be used as buffer size. */
   if ((recv_buff = calloc(grains[Y_COORD], sizeof(grid_type))) == NULL) {
      MPI_Abort(MPI_COMM_WORLD, EX_OSERR);
   }

   /* Print all the values computed by the node with rank 0 */
   for (x = 0; x < grains[X_COORD]; ++x) 
      handler(time, 0, (int)(offset[X_COORD] + x), grains, offset,
            (grid_type *)(grid[x + 1] + 1), handlerargs);

   /* Print all the values computed by the nodes with rank > 0. These 
    * values have to be received from the other nodes. */
   for (proc = 1; proc < nnodes; proc++) {
      time_comm_start = MPI_Wtime();
      /* Perform blocking receives from the non blocking sends. */
      MPI_Recv((void *)recv_grains, 2, MPI_INT, proc, base_tag, comm,
            &status);

      MPI_Recv((void *)recv_offset, 2, MPI_INT, proc, base_tag + 1, comm,
            &status);

      *time_comm += MPI_Wtime() - time_comm_start;
      /* Receive all the rows in the grid of the sender. */
      for (x = 0; x < recv_grains[X_COORD]; ++x) {
         time_comm_start = MPI_Wtime();
         MPI_Recv((void *)recv_buff, recv_grains[Y_COORD], MPI_GRID_TYPE, proc,
               base_tag + x + 2, comm, &status);

         *time_comm += MPI_Wtime() - time_comm_start;
         /* Print the buffer to the file. */
         handler(time, proc, (int)(offset[X_COORD] + x), recv_grains, recv_offset,
               recv_buff, handlerargs);
      }

   }

   free(recv_buff);
}

/* vim: set spell spelllang=en:cindent:tw=80:et*/

