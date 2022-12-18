CC = g++

default: sparsematmult

sparsematmult: sparsematmult.cpp
	${CC} -std=c++11 -O0 -g -Wall -Wextra -Wno-unused-parameter -fopenmp -o $@ sparsematmult.cpp

clean:
	-rm -f sparsematmult
