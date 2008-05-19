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

#ifndef _TIDIFFUSION_H_
#define _TIDIFFUSION_H_

#include <mpi.h>

#include "diffusion_help.h"

#define STEADY_TOLERANCE 1e-10

#define NUM_PARAMS 7

#ifndef __FAST_MATH__
#	warning enabling -ffast-math can really speed things up!
#endif

typedef struct {
	grid_type dx;
	grid_type dt;
	int	 ntotal;
	long	 ttotal;
	int	 l;
	int	 h;
	int	 freq;
} pparams;

int pparams_blength[NUM_PARAMS] = {1, 1, 1, 1, 1, 1, 1};

MPI_Datatype pparams_type[NUM_PARAMS] = {MPI_GRID_TYPE, MPI_GRID_TYPE, MPI_INT,
	MPI_LONG, MPI_INT, MPI_INT, MPI_INT};

enum {
	STEADY_TAG = FIRST_FREE_TAG,
	RED,
	BLACK,
	MPI_COLOR_UP,
	MPI_COLOR_UP_DUMMY,
	MPI_COLOR_DOWN,
	MPI_COLOR_DOWN_DUMMY,
	MPI_COLOR_LEFT,
	MPI_COLOR_LEFT_DUMMY,
	MPI_COLOR_RIGHT,
	MPI_COLOR_RIGHT_DUMMY
};

#endif /* _DIFFUSION_H_ */

/* vim: set spell spelllang=en:cindent:tw=80:et*/

