all: wave wavenosse

wavenosse: wavenosse.o
	mpicc -lm -o wavenosse wavenosse.o

wave: wave.o
	mpicc -lm -o wave wave.o

wave.o: wave.c wave.h
	mpicc -O3 -std=c99 -Wall -pedantic -msse2 -msse -mtune=nocona -c wave.c 

wavenosse.o : wave.c wave.h
	mpicc -o wavenosse.o -O3 -std=c99 -Wall -pedantic -msse2 -msse -mtune=nocona -DNO_SSE2 -c wave.c

clean:
	rm -f *o wave wavenosse

	
