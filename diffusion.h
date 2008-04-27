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

#ifndef _DIFFUSION_H_
#define _DIFFUSION_H_

#include <mpi.h>

#define NUM_PARAMS 7

typedef struct {
	float dx;
	float dt;
	float D;
	int	 ntotal;
	long	 ttotal;
	int	 l;
	int	 h;
} pparams;

int pparams_blength[NUM_PARAMS] = {1, 1, 1, 1, 1, 1, 1};

MPI_Datatype pparams_type[NUM_PARAMS] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, 
	MPI_INT, MPI_LONG, MPI_INT, MPI_INT};

enum { X_COORD,
	Y_COORD,
	X_UP_TAG,
   X_DOWN_TAG,
	Y_LEFT_TAG,
	Y_RIGHT_TAG, 
	GRAIN_COMM,
	OFFSET_COMM,
	GRID_COMM = 1000
};
#endif //  _DIFFUSION_H_

/* vim: set spell spelllang=en:cindent:tw=80:et*/

