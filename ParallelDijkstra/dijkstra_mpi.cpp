#include <mpi.h>
#include <iostream>
#include <string>
#include <cstring>
#include <cassert>
#include <cmath>
#include <cstddef>

#include <unistd.h>

using namespace std;

#define graph(i,j) graph[i * graphInfo.totalNumNodes + j]
#define graphSendBuffer(i,j) graphSendBuffer[i * totalNumNodes + j]

#define PROCESS_COMPLETED  -1000

int totalNumProcesses = -1;
int myProcessId = -1;

struct DistFromSource {
    float distance;
    float nodeId;
};

MPI_Datatype DistFromSource_MPI;

struct GraphInfo {
    int totalNumNodes;
    int startNode;
    int endNode;
};

static int getNumNodesofProcess(int totalNumNodes, int processId = -1) {
    if(processId == -1)
        processId = myProcessId;
    int x = ceil(1.0 * totalNumNodes / totalNumProcesses);
    int y = max(0, totalNumNodes - x * processId);
    return min(x,y);
}

static void loadAndSendData(const char *const filename, float*& graph, GraphInfo& graphInfo) {
    if(!filename) return;

    FILE *fp = NULL;
    int totalNumNodes;

    /* open the file */
    fp = fopen(filename, "r");
    assert(fp);

    /* get the number of nodes in the graph */
    int ret = fscanf(fp, "%d", &totalNumNodes);
    assert(1 == ret);

    graphInfo.totalNumNodes = totalNumNodes;
    graphInfo.startNode = 0;
    graphInfo.endNode = getNumNodesofProcess(totalNumNodes) - 1;

    /* allocate memory for values */
    graph = (float *)malloc((graphInfo.endNode - graphInfo.startNode + 1) * totalNumNodes * sizeof(*graph));
    assert(graph);

    for (int i = 0; i < (graphInfo.endNode - graphInfo.startNode + 1); ++i) {
        for (int j = 0; j < totalNumNodes; ++j) {
            ret = fscanf(fp, "%f", &graph(i, j));
            assert(ret == 1);
        }
    }

    /* allocate memory for values to send */
    float *graphSendBuffer = (float *)malloc((graphInfo.endNode + 1) * totalNumNodes * sizeof(*graphSendBuffer));
    assert(graphSendBuffer);

    GraphInfo* graphInfoSendBuffer = (GraphInfo*)malloc(sizeof(*graphInfoSendBuffer));
    assert(graphInfoSendBuffer);

    for(int toProcessId = 1, numNodesSent = graphInfo.endNode + 1; toProcessId < totalNumProcesses; ++toProcessId) {
        int curProcessNumNodes = getNumNodesofProcess(totalNumNodes, toProcessId);
        graphInfoSendBuffer->totalNumNodes = totalNumNodes;
        graphInfoSendBuffer->startNode = numNodesSent;
        graphInfoSendBuffer->endNode = numNodesSent + curProcessNumNodes - 1;
        MPI_Send((void*)graphInfoSendBuffer,
                 3,
                 MPI_INT,
                 toProcessId,
                 1,
                 MPI_COMM_WORLD);

        for (int i = 0; i < curProcessNumNodes; ++i) {
            for (int j = 0; j < totalNumNodes; ++j) {
                ret = fscanf(fp, "%f", &graphSendBuffer(i, j));
                assert(ret == 1);
            }
        }
        if(curProcessNumNodes != 0) {
            MPI_Send(graphSendBuffer,
                    curProcessNumNodes * totalNumNodes,
                    MPI_FLOAT,
                    toProcessId,
                    1,
                    MPI_COMM_WORLD);
        }
        numNodesSent = graphInfoSendBuffer->endNode + 1;
    }

    /* close file */
    ret = fclose(fp);
    assert(!ret);

    if(graphSendBuffer)
        free(graphSendBuffer);
}

static void getData(float*& graph, GraphInfo& graphInfo) {
    MPI_Status recv_status;

    /* recv graph info for this current process */
    MPI_Recv( (void*)&graphInfo,
              3,
              MPI_INT,
              0,
              1,
              MPI_COMM_WORLD,
              &recv_status);
    
    assert(getNumNodesofProcess(graphInfo.totalNumNodes) == (graphInfo.endNode - graphInfo.startNode + 1));
    if((graphInfo.endNode - graphInfo.startNode + 1) == 0) {
        return;
    }
    /* allocate memory for local values */
    graph = (float *)malloc((graphInfo.endNode - graphInfo.startNode + 1) * graphInfo.totalNumNodes * sizeof(*graph));
    assert(graph);

    MPI_Recv((void*)graph,
             (graphInfo.endNode - graphInfo.startNode + 1) * graphInfo.totalNumNodes,
              MPI_FLOAT,
              0,
              1,
              MPI_COMM_WORLD,
              &recv_status);
}

static void initializeMpiDistFromSourceStruct() {
    int blocksCount = 2;
    int blocksLength[2] = {1, 1};
    MPI_Datatype types[2] = {MPI_FLOAT, MPI_INT};
    MPI_Aint offsets[2];
    MPI_Datatype DistFromSource_MPI;
    offsets[0] = offsetof(DistFromSource, distance);
    offsets[1] = offsetof(DistFromSource, nodeId);

    MPI_Type_create_struct(blocksCount, blocksLength, offsets, types, &DistFromSource_MPI);
    MPI_Type_commit(&DistFromSource_MPI);
}

void logAction(string str, bool logActions) {
    if(!logActions) return;
    cout << "[Process " << myProcessId << "] " << str << endl;
}

static void recvAndGetMinDistNode(DistFromSource* minDistNodeArr,
                                  DistFromSource& minDistNode,
                                  bool* processCompleted,
                                  MPI_Request* recv_requests,
                                  MPI_Status* recv_statuses,
                                  bool logActions) {
    int arrSize = 0;
    for(int fromProcessId = 1; fromProcessId < totalNumProcesses; ++fromProcessId) {
        if(processCompleted[fromProcessId]) continue;
        MPI_Irecv((minDistNodeArr + fromProcessId),
                  2,
                  MPI_FLOAT,
                  fromProcessId,
                  1,
                  MPI_COMM_WORLD,
                  recv_requests + arrSize++);
    }
    MPI_Waitall(arrSize, recv_requests, recv_statuses);

    for(int i = 1; i < totalNumProcesses; ++i) {
        if(processCompleted[i]) continue;
        if(minDistNodeArr[i].nodeId == PROCESS_COMPLETED) {
            logAction("Received PROCESS_COMPLETED code from process " + to_string(i) + ", marking it as completed", logActions);
            processCompleted[i] = true;
        } else {
            logAction("Received MinDistNode {ID, Distance} = {" 
                        + to_string(minDistNode.nodeId) + ", " 
                        + to_string(minDistNode.distance) + "}"
                        + " from process " + to_string(i), logActions);
            if(minDistNodeArr[i].distance < minDistNode.distance) {
                minDistNode.distance = minDistNodeArr[i].distance;
                minDistNode.nodeId = minDistNodeArr[i].nodeId;
            } 
        }
    }
    logAction("Calculated Global MinDistNode {ID, Distance} = {" 
                        + to_string(minDistNode.nodeId) + ", " 
                        + to_string(minDistNode.distance) + "}", logActions);
}

void broadcastMinDistNodeToAvailableProcesses(DistFromSource& minDistNode,
                                              bool* processCompleted,
                                              MPI_Request* send_requests,
                                              MPI_Status* send_statuses,
                                              bool logActions) {
    int numReq = 0;
    for(int toProcessId = 1; toProcessId < totalNumProcesses; ++toProcessId) {
        if(processCompleted[toProcessId]) continue;
        logAction("Sending global min dist node to process " + to_string(toProcessId), logActions);
        MPI_Isend((void*)&minDistNode,
                   2,
                   MPI_FLOAT,
                   toProcessId,
                   1,
                   MPI_COMM_WORLD,
                   send_requests + numReq++);
    }
    MPI_Waitall(numReq, send_requests, send_statuses);
}

static void dijkstra(int sourceNode, const float* graph, GraphInfo& graphInfo, float* dist, bool logActions = false) {
    MPI_Request send_requests[totalNumProcesses];
    MPI_Request recv_requests[totalNumProcesses];
    MPI_Status send_statuses[totalNumProcesses];
    MPI_Status recv_statuses[totalNumProcesses];

    MPI_Bcast((void*)&sourceNode,
                1,
                MPI_INT,
                0,
                MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    logAction("In dijkstra, sourceNode = " + to_string(sourceNode), logActions);

    for(int i = 0; i < (graphInfo.endNode - graphInfo.startNode + 1); ++i) {
        dist[i] = graph(i, sourceNode);
    }

    bool* minDistCalculated = (bool*)calloc((graphInfo.endNode - graphInfo.startNode + 1), sizeof(*minDistCalculated));
    assert(minDistCalculated);

    int numMinDistCalculated = 0;
    if(graphInfo.startNode <= sourceNode && sourceNode <= graphInfo.endNode) {
        minDistCalculated[sourceNode - graphInfo.startNode] = true;
        ++numMinDistCalculated;
    }

    DistFromSource* minDistNodeArr = nullptr;
    bool* processCompleted = nullptr;
    if(myProcessId == 0) {
        minDistNodeArr = (DistFromSource*)malloc(totalNumProcesses * sizeof(*minDistNodeArr));
        processCompleted = (bool*) malloc(totalNumProcesses * sizeof(*processCompleted));
        for(int i = 0; i < totalNumProcesses; ++i) {
            processCompleted[i] = (getNumNodesofProcess(totalNumProcesses, i) == 0);
        }
    }
    DistFromSource minDistNode;

    while(myProcessId == 0 || numMinDistCalculated != (graphInfo.endNode - graphInfo.startNode + 1)) {
        minDistNode.nodeId = -1;
        minDistNode.distance = INFINITY;
        for(int i = 0; i < (graphInfo.endNode - graphInfo.startNode + 1); ++i) {
            if(!minDistCalculated[i] && dist[i] < minDistNode.distance) {
                minDistNode.distance = dist[i];
                minDistNode.nodeId = graphInfo.startNode + i;
            }
        }

        if(myProcessId == 0) {
            logAction("Receiving Min Dist Node from all processess and calculating global min dist node", logActions);
            recvAndGetMinDistNode(minDistNodeArr, minDistNode, processCompleted, recv_requests, recv_statuses, logActions);
            logAction("Broadcasting global Min dist node to all the active processess in dijkstra", logActions);
            broadcastMinDistNodeToAvailableProcesses(minDistNode, processCompleted, send_requests, send_statuses, logActions);
        } else {
            logAction("Sending MinDistNode {ID, Distance} = {" 
                        + to_string(minDistNode.nodeId) + ", " 
                        + to_string(minDistNode.distance) + "}", logActions);
            MPI_Send((void*)&minDistNode,
                     2,
                     MPI_FLOAT,
                     0,
                     1,
                     MPI_COMM_WORLD);
            MPI_Recv((void*)&minDistNode,
                     2,
                     MPI_FLOAT,
                     0,
                     1,
                     MPI_COMM_WORLD,
                     recv_statuses + myProcessId); 
            logAction("Received Global MinDistNode {ID, Distance} = {" 
                        + to_string(minDistNode.nodeId) + ", " 
                        + to_string(minDistNode.distance) + "}", logActions);        
        }

        if(minDistNode.nodeId == -1) {
            logAction("Exiting dijkstra because, Global MinDistNodeId is -1. i.e., no node is the other set is reachable from current set", logActions);
            break;
        }

        logAction("Current Global MinDistNodeId = " + to_string(minDistNode.nodeId), logActions);

        if(graphInfo.startNode <= minDistNode.nodeId && minDistNode.nodeId <= graphInfo.endNode) {
            minDistCalculated[(int)minDistNode.nodeId - graphInfo.startNode] = true;
            ++numMinDistCalculated;
        }

        for(int i = 0; i < (graphInfo.endNode - graphInfo.startNode + 1); ++i) {
            if(!minDistCalculated[i] && 
                minDistNode.distance + graph(i, (int)minDistNode.nodeId) < dist[i]) {
                    dist[i] = minDistNode.distance + graph(i, (int)minDistNode.nodeId);
                }
        }
    }
    if(numMinDistCalculated == (graphInfo.endNode - graphInfo.startNode + 1)) {
        logAction("Exiting dijkstra because, done calculating min dist for all my nodes", logActions);
    }

    if(myProcessId != 0) {
        minDistNode.nodeId = PROCESS_COMPLETED;
        minDistNode.distance = INFINITY;
        logAction("Sending PROCESS_COMPLETED to process 0", logActions);
        MPI_Send((void*)&minDistNode,
                2,
                MPI_FLOAT,
                0,
                1,
                MPI_COMM_WORLD);
    }
    if(minDistCalculated) free(minDistCalculated);
    if(minDistNodeArr) free(minDistNodeArr);
    if(processCompleted) free(processCompleted);
}

static void printNumbers(const char *const filename, GraphInfo& graphInfo, float* dist) {
    if(!filename) return;

    float* globalDist = (float*)malloc(graphInfo.totalNumNodes * sizeof(*globalDist));
    assert(globalDist);

    MPI_Request recv_req[totalNumProcesses];
    MPI_Status statuses[totalNumProcesses];

    for(int fromProcessId = 0, i = 0; fromProcessId < totalNumProcesses; ++fromProcessId) {
        if(fromProcessId == 0) {
            memcpy(globalDist, dist, getNumNodesofProcess(graphInfo.totalNumNodes) * sizeof(*dist));
        } else {
            MPI_Irecv((void*)(globalDist + i),
                      getNumNodesofProcess(graphInfo.totalNumNodes, fromProcessId),
                      MPI_FLOAT,
                      fromProcessId,
                      1,
                      MPI_COMM_WORLD,
                      recv_req + fromProcessId);
        }
        i += ceil(1.0 * graphInfo.totalNumNodes / totalNumProcesses);
    }

    MPI_Waitall(totalNumProcesses - 1, recv_req + 1, statuses + 1);

    FILE *fout;

    /* open file */
    if (NULL == (fout = fopen(filename, "w"))) {
        fprintf(stderr, "error opening '%s'\n", filename);
        abort();
    }

    /* write numbers to fout */
    for (int i = 0; i < graphInfo.totalNumNodes; ++i) {
        fprintf(fout, "%10.4f\n", globalDist[i]);
    }

    fclose(fout);
}

static void printGraphAndDist(const float* graph, const GraphInfo& graphInfo, const float* dist) {
    cout << "========== Process " << myProcessId << " ==========" << endl;
    cout << "========== GRAPH ==========" << endl;
    cout << graphInfo.totalNumNodes << " " << graphInfo.startNode << " " << graphInfo.endNode << endl;
    cout << "========== GRAPH ==========" << endl;
    for(int i = 0; i < (graphInfo.endNode - graphInfo.startNode + 1); ++i) {
        for(int j = 0; j < graphInfo.totalNumNodes; ++j) {
            cout << graph(i,j) << " ";
        }
        cout << endl;
    }
    cout << "========== DIST ==========" << endl;
    for(int i = 0; i < (graphInfo.endNode - graphInfo.startNode + 1); ++i) {
        cout << dist[i] << " ";
    }
    cout << endl;
}

static void printTime(const double seconds)
{
  printf("Search Time: %0.06fs\n", seconds);
}

void initializeDistArray(float*& dist, GraphInfo& graphInfo) {
    dist = (float*) malloc((graphInfo.endNode - graphInfo.startNode + 1) * sizeof(*dist));
    assert(dist);
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        printf("Invalid number of arguments.\nUsage: dijkstra <graph> <num_sources> [<output_file>].\n");
        return EXIT_FAILURE;
    }

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &totalNumProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myProcessId);

    clock_t time_start, time_end;
    float* graph = nullptr;
    GraphInfo graphInfo;
    float* dist = nullptr;

    if (myProcessId == 0) {
        printf("Loading graph from %s.\n", argv[1]);
        loadAndSendData(argv[1], graph, graphInfo);
    } else {
        getData(graph, graphInfo);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* initialize random seed: */
    srand(time(NULL));
    unsigned int seed = time(NULL);

    /* figure out number of random sources to search from */
    int numRandomSources = atoi(argv[2]);
    assert(numRandomSources > 0);

    initializeDistArray(dist, graphInfo);
    initializeMpiDistFromSourceStruct();

    if (myProcessId == 0)
        printf("Performing %d searches from random sources.\n", numRandomSources);
    time_start = clock();
    for (int i = 0, randomSource; i < numRandomSources; ++i)
    {
        randomSource = rand_r(&seed) % graphInfo.totalNumNodes;
        dijkstra(randomSource, graph, graphInfo, dist);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    time_end = clock();
    if (myProcessId == 0)
        printTime((double)(time_end - time_start) / CLOCKS_PER_SEC);

    if (argc >= 4)
    {
        if (myProcessId == 0)
            printf("Computing result for source 0.\n");
        dijkstra(0, graph, graphInfo, dist);
        if (myProcessId == 0) {
            printf("Writing result to %s\n", argv[3]);
            printNumbers(argv[3], graphInfo, dist);
        } else {
            MPI_Send(dist,
                     graphInfo.endNode - graphInfo.startNode + 1,
                     MPI_FLOAT,
                     0,
                     1,
                     MPI_COMM_WORLD);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(graph) free(graph);
    if(dist) free(dist);
    MPI_Finalize();
    return EXIT_SUCCESS;
}