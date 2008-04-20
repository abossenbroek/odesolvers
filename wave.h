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
#ifndef _WAVE_H_
#define _WAVE_H_

#include <mpi.h>
   
#define NUM_PARAMS 9

typedef struct {
	 enum { NONE=0, SINE=1, PLUCKED=2 } method; /* NONE is an illegal value  */
    int periods;     /* number of sine waves on string in case of SINE      */
    double location; /* global location (0.0 .. 1.0) of plucked position    */
    double stime;    /* time in seconds to simulate                         */
    double delta;    /* time to advance per string computation/iteration    */
    int ntotal;      /* number of points on string in total (global string) */
    int freq;        /* visualize every <freq> iterations, <=0 if none      */
	 double height;   /* the height of the pluck.									 */
	 double tau;	
} solve_params;
   
int solve_params_blength[NUM_PARAMS] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
MPI_Datatype solve_params_type[NUM_PARAMS] = {MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE,
	MPI_DOUBLE, MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE};

#ifndef M_PI
#define M_PI        3.14159265358979323846264338327950288   /* pi */
#endif

enum {
	PDE_COMM,
	GRAIN_COMM,
	OFFSET_COMM,
	LVAL_COMM,
	RVAL_COMM
};

#endif /* _WAVE_H_ */


/* vim: set spell spelllang=en:cindent:et */
