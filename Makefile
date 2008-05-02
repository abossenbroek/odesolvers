all: wave diffus diffusdbl diffusdblnosse wavenosse diffusnosse

wavenosse: wavenosse.o
	mpicc -lm -o wavenosse wavenosse.o

wave: wave.o
	mpicc -lm -o wave wave.o

wave.o: wave.c wave.h
	mpicc -O3 -std=c99 -Wall -pedantic -msse2 -msse -mtune=prescott -c wave.c  -g

wavenosse.o : wave.c wave.h
	mpicc -o wavenosse.o -O3 -std=c99 -Wall -pedantic -msse2 -msse -mtune=prescott -DNO_SSE2 -c wave.c -g

diffus: diffusion.o 
	mpicc -lm -o diffus diffusion.o

diffusnosse: diffusionnosse.o 
	mpicc -lm -o diffusnosse diffusionnosse.o

diffusion.o: diffusion.h diffusion.c
	mpicc -o diffusion.o -std=c99 -Wall -pedantic -c diffusion.c -g -O3 -mtune=prescott -msse

diffusionnosse.o: diffusion.h diffusion.c
	mpicc -o diffusionnosse.o -std=c99 -Wall -pedantic -c diffusion.c -DNO_SSE -g -O3 -mtune=prescott

diffusdbl: diffusiondbl.o 
	mpicc -lm -o diffusdbl diffusiondbl.o

diffusdblnosse: diffusiondblnosse.o 
	mpicc -lm -o diffusdblnosse diffusiondblnosse.o

diffusiondbl.o: diffusion.h diffusion.c
	mpicc -o diffusiondbl.o -std=c99 -Wall -pedantic -c diffusion.c -g -DDOUBLE -O3 -mtune=prescott -msse -msse2

diffusiondblnosse.o: diffusion.h diffusion.c
	mpicc -o diffusiondblnosse.o -std=c99 -Wall -pedantic -c diffusion.c -DNO_SSE -g -DDOUBLE -O3 -mtune=prescott

clean:
	rm -f *o wave wavenosse diffus diffusnosse diffusdbl diffusdblnosse

	
