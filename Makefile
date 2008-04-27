all: wave wavenosse diffusion diffusionnosse

wavenosse: wavenosse.o
	mpicc -lm -o wavenosse wavenosse.o

wave: wave.o
	mpicc -lm -o wave wave.o

wave.o: wave.c wave.h
	mpicc -O3 -std=c99 -Wall -pedantic -msse2 -msse -mtune=nocona -c wave.c  -g

wavenosse.o : wave.c wave.h
	mpicc -o wavenosse.o -O3 -std=c99 -Wall -pedantic -msse2 -msse -mtune=prescott -DNO_SSE2 -c wave.c -g

diffusion: diffusion.o 
	mpicc -lm -o diffusion diffusion.o

diffusionnosse: diffusionnosse.o 
	mpicc -lm -o diffusionnosse diffusionnosse.o

diffusion.o: diffusion.h diffusion.c
	mpicc -o diffusion.o -std=c99 -Wall -pedantic -c diffusion.c -g

diffusionnosse.o: diffusion.h diffusion.c
	mpicc -o diffusionnosse.o -std=c99 -Wall -pedantic -c diffusion.c -DNO_SSE -g

clean:
	rm -f *o wave wavenosse diffusion diffusionnosse

	
