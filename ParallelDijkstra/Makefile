SRC := dijkstra.cpp
MPISRC := dijkstra_mpi.cpp
OMPSRC := dijkstra_omp.cpp

default: dijkstra_mpi

dijkstra: $(SRC)
	g++ -std=c++11 -O3 -Wall -Wextra -o $@ $<

dijkstra_mpi: $(MPISRC)
	mpic++ -w -O3 -Wall -Wextra -Wno-cast-function-type -o $@ $<

dijkstra_omp: $(OMPSRC)
	g++ -std=c++11 -g -fopenmp -O3 -Wall -Wextra -o $@ $<

clean: 
	rm -f dijkstra dijkstra_mpi dijkstra_omp
