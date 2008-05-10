diffusion_files=diffusion.c diffusion.h 
diffusion_help_files=diffusion_help.h diffusion_help.c
tidiffusion_files= tidiffusion.h tidiffusion.c
OPTFLAGS=-msse -msse2 -mtune=prescott
CFLAGS=-std=gnu9x -Wall -pedantic -g
LDFLAGS=-lm


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

diffus: diffusion.o diffusion_help.o
	mpicc $(LDFLAGS) -o diffus diffusion.o diffusion_help.o

diffusnosse: diffusionnosse.o diffusion_helpnosse.o
	mpicc $(LDFLAGS) -o diffusnosse diffusionnosse.o diffusion_helpnosse.o

diffusion.o: $(diffusion_files) $(diffusion_help_files)
	mpicc -o diffusion.o -c diffusion.c $(CFLAGS) $(OPTFLAGS)

diffusdbl: diffusiondbl.o diffusion_helpdbl.o
	mpicc $(LDFLAGS) -o diffusdbl diffusiondbl.o diffusion_helpdbl.o

diffusdblnosse: diffusiondblnosse.o diffusion_helpdblnosse.o
	mpicc $(LDFLAGS) -o diffusdblnosse diffusiondblnosse.o diffusion_helpdblnosse.o

diffusionnosse.o: $(diffusion_files) $(diffusion_help_files)
	mpicc -o diffusionnosse.o -c diffusion.c $(CFLAGS) $(OPTFLAGS) -DNO_SSE 

diffusiondbl.o: $(diffusion_files) $(diffusion_help_files)
	mpicc -o diffusiondbl.o -c diffusion.c $(CFLAGS) $(OPTFLAGS) -DDOUBLE

diffusiondblnosse.o: $(diffusion_files) $(diffusion_help_files)
	mpicc -o diffusiondblnosse.o -c diffusion.c $(CFLAGS) $(OPTFLAGS) -DDOUBLE -DNO_SSE

diffusion_help.o: $(diffusion_help_files)
	mpicc -o diffusion_help.o -c diffusion_help.c $(OPTFLAGS) $(CFLAGS)

diffusion_helpdbl.o: $(diffusion_help_files)
	mpicc -o diffusion_helpdbl.o -c diffusion_help.c $(OPTFLAGS) -DDOUBLE $(CFLAGS)

diffusion_helpnosse.o: $(diffusion_help_files)
	mpicc -o diffusion_helpnosse.o -c diffusion_help.c $(OPTFLAGS) -DNO_SSE $(CFLAGS)

diffusion_helpdblnosse.o: $(diffusion_help_files)
	mpicc -o diffusion_helpdblnosse.o -c diffusion_help.c $(OPTFLAGS) -DDOUBLE -DNO_SSE $(CFLAGS)

tidiffus: tidiffusion.o diffusion_help.o
	mpicc $(LDFLAGS) -o tidiffus tidiffusion.o diffusion_help.o

tidiffusnosse: tidiffusionnosse.o diffusion_helpnosse.o
	mpicc $(LDFLAGS) -o tidiffusnosse tidiffusionnosse.o diffusion_helpnosse.o

tidiffusion.o: $(tidiffusion_files) $(diffusion_help_files)
	mpicc -o tidiffusion.o -c tidiffusion.c $(CFLAGS) $(OPTFLAGS)

tidiffusdbl: tidiffusiondbl.o diffusion_helpdbl.o
	mpicc $(LDFLAGS) -o tidiffusdbl tidiffusiondbl.o diffusion_helpdbl.o

tidiffusdblnosse: tidiffusiondblnosse.o diffusion_helpdblnosse.o
	mpicc $(LDFLAGS) -o tidiffusdblnosse tidiffusiondblnosse.o diffusion_helpdblnosse.o

tidiffusionnosse.o: $(tidiffusion_files) $(diffusion_help_files)
	mpicc -o tidiffusionnosse.o -c tidiffusion.c $(CFLAGS) $(OPTFLAGS) -DNO_SSE 

tidiffusiondbl.o: $(tidiffusion_files) $(diffusion_help_files)
	mpicc -o tidiffusiondbl.o -c tidiffusion.c $(CFLAGS) $(OPTFLAGS) -DDOUBLE

tidiffusiondblnosse.o: $(tidiffusion_files) $(diffusion_help_files)
	mpicc -o tidiffusiondblnosse.o -c tidiffusion.c $(CFLAGS) $(OPTFLAGS) -DDOUBLE -DNO_SSE

clean:
	rm -f *o wave wavenosse diffus diffusnosse diffusdbl diffusdblnosse tidiffus \
		tidiffusdbl tidiffusdblnosse tidiffusnosse
	
