
CC = icc
CFLAGS = -std=gnu99 -O3
LIBS = -lOpenCL

TARGETS = simulation

all:	$(TARGETS)

simulation: simulation.o common.o
	$(CC) -o $@ $(LIBS) simulation.o common.o -lOpenCL
simulation.o: simulation.c common.h
	$(CC) -c $(CFLAGS) simulation.c -lOpenCL
common.o: common.c common.h
	$(CC) -c $(CFLAGS) common.c

clean:
	rm -f *.o $(TARGETS) *.stdout *.txt
