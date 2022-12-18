#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include <cassert>
#include <cstring>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#include <omp.h>

using namespace std;

#define graph(i,j) graph[i * numNodes + j]

static void loadData(const char *const filename, int &numNodes, float*& graph) {
    if(!filename) return;

    FILE *fp = NULL;

    /* open the file */
    fp = fopen(filename, "r");
    assert(fp);

    /* get the number of nodes in the graph */
    int ret = fscanf(fp, "%d", &numNodes);
    assert(1 == ret);

    /* allocate memory for local values */
    graph = (float *)malloc(numNodes * numNodes * sizeof(*graph));
    assert(graph);

    /* read in roots local values */
    for (int i = 0; i < numNodes; ++i) {
        for (int j = 0; j < numNodes; ++j) {
            ret = fscanf(fp, "%f", &graph(i, j));
            assert(1 == ret);
        }
    }

    /* close file */
    ret = fclose(fp);
    assert(!ret);
}

struct DistFromSource {
    float distance;
    int nodeId;
};

static void getMyNodes(int numNodes, int threadId,
                      int& startNode, int& endNode) {
    int numThreads = omp_get_max_threads();
    int nodesPerThread = ceil(1.0 * numNodes / numThreads);
    int remainingNodes = max(0, numNodes - nodesPerThread * threadId);

    if(remainingNodes < nodesPerThread) {
        if(remainingNodes == 0) {
            startNode = -1;
            endNode = -2;
        } else {
            startNode = threadId * nodesPerThread;
            endNode = startNode + remainingNodes - 1;
        }

    } else {
        startNode = threadId * nodesPerThread;
        endNode = startNode + nodesPerThread - 1;
    }
}

static void dijkstra(const int source, const int numNodes, const float* graph, float*& dist) {
    assert(source>=0 && source<numNodes && graph);

    bool *minDistCalculated = (bool*) calloc(numNodes, sizeof(*minDistCalculated));
    assert(minDistCalculated);

    int i, j, tid, startNode, endNode;

    #pragma omp parallel for default(shared) private(i)
    for(i=0; i<numNodes; ++i) {
        dist[i] = graph(i, source);
    }

    minDistCalculated[source] = true;
    DistFromSource minDistNode, minDistNodePerthread;

    #pragma omp parallel private(i, j, tid, startNode, endNode, minDistNodePerthread)\
                         shared(graph, dist, minDistCalculated, minDistNode)
    {
        tid = omp_get_thread_num();
        getMyNodes(numNodes, tid, startNode, endNode);
        for(i=0; i<numNodes; ++i) {
            #pragma omp single
            {
                minDistNode.nodeId = -1;
                minDistNode.distance = INFINITY;
            }

            minDistNodePerthread.nodeId = -1;
            minDistNodePerthread.distance = INFINITY;

            for(j = startNode; j <= endNode; ++j) {
                if(!minDistCalculated[j] && dist[j] < minDistNodePerthread.distance) {
                    minDistNodePerthread.distance = dist[j];
                    minDistNodePerthread.nodeId = j;
                }
            }

            #pragma omp critical
            {
                if(minDistNodePerthread.distance < minDistNode.distance) {
                    minDistNode.distance = minDistNodePerthread.distance;
                    minDistNode.nodeId = minDistNodePerthread.nodeId;
                }
            }
            #pragma omp barrier

            #pragma omp single 
            {
                if(minDistNode.nodeId != -1) {
                    minDistCalculated[minDistNode.nodeId] = true;
                }
            }
            #pragma omp barrier

            if(minDistNode.nodeId != -1) {
                for(j = startNode; j <= endNode; ++j) {
                    if(!minDistCalculated[j] &&
                        minDistNode.distance + graph(j, minDistNode.nodeId) < dist[j]) {
                        dist[j] = minDistNode.distance + graph(j, minDistNode.nodeId);
                    }
                }
            }
            #pragma omp barrier
        }
    }

    if(minDistCalculated) {
        free(minDistCalculated);
        minDistCalculated = nullptr;
    }
}

static void printNumbers(const char *const filename, const int numNodes, const float* dist) {
    if(!dist || !filename) return;

    FILE *fout;

    /* open file */
    if (NULL == (fout = fopen(filename, "w"))) {
        fprintf(stderr, "error opening '%s'\n", filename);
        abort();
    }

    /* write numbers to fout */
    for (int i = 0; i < numNodes; ++i) {
        fprintf(fout, "%10.4f\n", dist[i]);
    }

    fclose(fout);
}

static void printTime(const double seconds)
{
  printf("Search Time: %0.06fs\n", seconds);
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        printf("Invalid number of arguments.\nUsage: dijkstra <graph> <num_sources> [<output_file>].\n");
        return EXIT_FAILURE;
    }

    int numNodes;
    double time_start, time_end;
    float *graph = nullptr, *dist = nullptr;
    
    /* initialize random seed: */
    srand(time(NULL));
    unsigned int seed = time(NULL);

    /* figure out number of random sources to search from */
    int numRandomSources = atoi(argv[2]);
    assert(numRandomSources > 0);

    uint16_t numThreads = 1;
    if(argc >=6 && strcasecmp(argv[4], "-t") == 0) {
        numThreads = atoi(argv[5]);
    }
    omp_set_num_threads(numThreads);

    /* load data */
    printf("Loading graph from %s.\n", argv[1]);
    loadData(argv[1], numNodes, graph);

    dist = (float*) malloc(numNodes * sizeof(*dist));
    assert(dist);

    printf("Performing %d searches from random sources.\n", numRandomSources);
    time_start = omp_get_wtime();
    for (int i = 0, randomSource; i < numRandomSources; ++i)
    {
        randomSource = rand_r(&seed) % numNodes;
        dijkstra(randomSource, numNodes, graph, dist);
    }
    time_end = omp_get_wtime();
    printTime(time_end - time_start);

    if (argc >= 4)
    {
        printf("Computing result for source 0.\n");
        dijkstra(0, numNodes, graph, dist);
        printf("Writing result to %s.\n", argv[3]);
        printNumbers(argv[3], numNodes, dist);
    }

    if(graph) free(graph);
    if(dist) free(dist);

    return EXIT_SUCCESS;
}
