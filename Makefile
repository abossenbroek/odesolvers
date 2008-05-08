all: allwave alldiffusion alltidiffusion

alldiffusion: diffus diffusdbl diffusdblnosse diffusnosse

alltidiffusion: tidiffus tidiffusdbl tidiffusdblnosse tidiffusnosse

allwave: wave wavenosse

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

tidiffusnosse: tidiffusnosse.o 
	mpicc -lm -o tidiffusnosse tidiffusnosse.o

tidiffusnosse.o: tidiffusion.h tidiffusion.c
	mpicc -o tidiffusnosse.o -std=c99 -Wall -pedantic -c tidiffusion.c -g -DSTEADY -O3 -mtune=prescott -msse -msse2 -DNO_SSE

tidiffus: tidiffus.o 
	mpicc -lm -o tidiffus tidiffus.o

tidiffus.o: tidiffusion.h tidiffusion.c
	mpicc -o tidiffus.o -std=c99 -Wall -pedantic -c tidiffusion.c -g -DSTEADY -O3 -mtune=prescott -msse -msse2 

tidiffusdblnosse: tidiffusdblnosse.o 
	mpicc -lm -o tidiffusdblnosse tidiffusdblnosse.o

tidiffusdblnosse.o: tidiffusion.h tidiffusion.c
	mpicc -o tidiffusdblnosse.o -std=c99 -Wall -pedantic -c tidiffusion.c -g -DSTEADY -O3 -mtune=prescott -msse -msse2 -DNO_SSE -DDOUBLE

tidiffusdbl: tidiffusdbl.o 
	mpicc -lm -o tidiffusdbl tidiffusdbl.o

tidiffusdbl.o: tidiffusion.h tidiffusion.c
	mpicc -o tidiffusdbl.o -std=c99 -Wall -pedantic -c tidiffusion.c -g -DSTEADY -O3 -mtune=prescott -msse -msse2  -DDOUBLE

clean:
	rm -f *o wave wavenosse diffus diffusnosse diffusdbl diffusdblnosse tidiffus \
		tidiffusdbl tidiffusdblnosse tidiffusnosse

	
