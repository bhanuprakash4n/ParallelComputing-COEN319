#include <cerrno>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

#include "sparse_matrix.h"
#include "utility_functions.h"


using namespace std;




int main(int argc, char * argv[])
{
    SparseMatrix adjM;
    ifstream nodeFile;

    if(argc < 3){
        cerr << "Invalid options." << endl << "<program> <input_file> <nodes> [-t <num_threads>]" << endl;
        exit(1);
    }
    string fileName(argv[1]);
    uns32 N = atoi(argv[2]);
    uns32 nthreads = 1;
    if(argc == 5 && strcasecmp(argv[3], "-t") == 0){
        nthreads = atoi(argv[4]);

#ifdef OPEN_MP_PROJECT
        omp_set_num_threads(nthreads);
#endif
    }
    uns32 approach = 1;
    if(argc == 6 && strcasecmp(argv[5], "2") == 0)
        approach = 2;
    std::cout << "file: " << fileName << endl;
    std::cout << "number of nodes: " << N << endl;
    std::cout << "nthreads: " << nthreads << endl;   


    auto start = chrono::high_resolution_clock::now();

    cout << "============================================" << endl;
    cout << "Starting process..." << endl;
    cout << "Filling out NodeInfo for each node i...\n" << endl;


    // Get NodeInfo for each node
    NodeInfo *nodes = new NodeInfo[N];
    uns32 *oLinks = (uns32 *) malloc(sizeof(uns32) * N);
    memset(oLinks, 0x0, sizeof(uns32) * N);
    nodeFile.open(fileName);

    uns32 totalNnz = getNnzPerNode(nodes, oLinks, N, nodeFile);
    flt32 sparcity = (flt32)totalNnz / (N*N);
    cout << "Total number of non-zeros = " << totalNnz << endl;
    cout << "Sparcity = " << sparcity << endl;

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "NodeInfo obtained: duration = " << duration.count() << " ms" << endl;
    cout << "============================================" << endl;
    auto pause = stop;

    // Create the adjacency matrix from the NodeInfo array
    cout << "Creating adjacency matrix..." << endl;
    adjM.reserve(N, N, totalNnz);
    createNodeMatrix(&adjM, nodes, oLinks, N);
    free(oLinks);
    delete[] nodes;
    // adjM.printMatrix(25);
    // test_matrix(&adjM);

    stop = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(stop - pause);
    cout << "Adjacency matrix created: duration = " << duration.count() << " ms" << endl;
    cout << "============================================" << endl;
    pause = stop;


    // Creating and initializing page rank vector
    cout << "Creating and initializing page rank vector..." << endl;
    flt32 *pgRankV = (flt32 *) malloc(sizeof(flt32) * N);
    initializePageRankVector(pgRankV, N);

    stop = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(stop - pause);
    cout << "Page rank vector created and initialized: duration = " << duration.count() << " ms" << endl;
    cout << "============================================" << endl;
    pause = stop;


    // Calculate page rank
    cout << "Calculating page rank..." << endl;
    flt32 *finalPgRankV = (flt32 *) malloc(sizeof(flt32) * N);
    memset(finalPgRankV, 0x0, sizeof(flt32)*N);
    if(approach == 2) {
        cout << "Calculating page rank using Sparse Matrix X Sparse Matrix..." << endl;
        calculatePageRank2(&adjM, pgRankV, finalPgRankV, N);
    }
    else {
        cout << "Calculating page rank using Dense Vector X Sparse Matrix only..." << endl;
        calculatePageRank(&adjM, pgRankV, finalPgRankV, N);
    }

    stop = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(stop - pause);
    cout << "Page rank calculated: duration = " << duration.count() << " ms" << endl;
    cout << "============================================" << endl;
    pause = stop;




    stop = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "Program done: total duration = " << duration.count() << " ms" << endl;
    cout << "============================================" << endl;

    adjM.tearDown();

    return EXIT_SUCCESS;
}



