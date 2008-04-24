all: wave wavenosse diffusion

wavenosse: wavenosse.o
	mpicc -lm -o wavenosse wavenosse.o

wave: wave.o
	mpicc -lm -o wave wave.o

wave.o: wave.c wave.h
	mpicc -O3 -std=c99 -Wall -pedantic -msse2 -msse -mtune=nocona -c wave.c 

wavenosse.o : wave.c wave.h
	mpicc -o wavenosse.o -O3 -std=c99 -Wall -pedantic -msse2 -msse -mtune=prescott -DNO_SSE2 -c wave.c

diffusion: diffusion.o 
	mpicc -lm -o diffusion diffusion.o

diffusion.o: diffusion.h diffusion.c
	mpicc -o diffusion.o -std=c99 -Wall -pedantic -c diffusion.c

clean:
	rm -f *o wave wavenosse

	
