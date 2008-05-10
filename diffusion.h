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

#ifndef _DIFFUSION_H_
#define _DIFFUSION_H_

#include "diffusion_help.h"
#include <mpi.h>

#define NUM_PARAMS 8

typedef struct {
	grid_type dx;
	grid_type dt;
	grid_type D;
	int	 ntotal;
	long	 ttotal;
	int	 l;
	int	 h;
	int	 freq;
} pparams;

int pparams_blength[NUM_PARAMS] = {1, 1, 1, 1, 1, 1, 1, 1};

MPI_Datatype pparams_type[NUM_PARAMS] = {MPI_GRID_TYPE, MPI_GRID_TYPE, MPI_GRID_TYPE, 
	MPI_INT, MPI_LONG, MPI_INT, MPI_INT, MPI_INT};

#endif /* _DIFFUSION_H_ */

/* vim: set spell spelllang=en:cindent:tw=80:et*/

