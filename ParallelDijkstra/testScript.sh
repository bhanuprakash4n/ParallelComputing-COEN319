outputDirectory="outputs"
inputFile="100.graph"
outputFile="$outputDirectory/output100.txt"

echo "make clean"
make clean
echo ""

echo "make dijkstra"
make dijkstra
echo ""

echo "./dijkstra $inputFile "100" $outputFile"
./dijkstra $inputFile "100" $outputFile

echo ""
echo ""

module load OpenMPI

echo "make dijkstra_mpi"
make dijkstra_mpi
echo ""

# test mpi with 1, 2, 4, 8, 16, 28 processess
declare -a numProcessesArr=(1 2 4 8 16 28)
declare -a outputFileMPI=("output100_mpi1.txt" "output100_mpi2.txt" "output100_mpi4.txt" "output100_mpi8.txt" "output100_mpi16.txt" "output100_mpi28.txt")
for i in "${!numProcessesArr[@]}";
do
    echo "mpirun -np ${numProcessesArr[$i]} --oversubscribe ./dijkstra_mpi $inputFile "100" $outputDirectory/${outputFileMPI[$i]}"
    mpirun -np ${numProcessesArr[$i]} --oversubscribe ./dijkstra_mpi $inputFile "100" $outputDirectory/${outputFileMPI[$i]}
    if ! diff -q $outputFile $outputDirectory/${outputFileMPI[$i]} &>/dev/null;
    then
        echo "Output mismatched for $outputFile and $outputDirectory/${outputFileMPI[$i]}"
        break
    else
        echo "Output matched for $outputFile and $outputDirectory/${outputFileMPI[$i]}"
    fi
    echo ""
done

echo ""
echo ""

echo "make dijkstra_omp"
make dijkstra_omp
echo ""

# test omp with 1, 2, 4, 8, 16, 28 threads
declare -a numThreadsArr=(1 2 4 8 16 28)
declare -a outputFileOMP=("output100_omp1.txt" "output100_omp2.txt" "output100_omp4.txt" "output100_omp8.txt" "output100_omp16.txt" "output100_omp28.txt")
for i in "${!numProcessesArr[@]}";
do
    echo "./dijkstra_omp $inputFile "100" $outputDirectory/${outputFileOMP[$i]} -t ${numThreadsArr[$i]}"
    ./dijkstra_omp $inputFile "100" $outputDirectory/${outputFileOMP[$i]} -t ${numThreadsArr[$i]}
    if ! diff -q $outputFile $outputDirectory/${outputFileOMP[$i]} &>/dev/null;
    then
        echo "Output mismatched for $outputFile and $outputDirectory/${outputFileOMP[$i]}"
        break
    else
        echo "Output matched for $outputFile and $outputDirectory/${outputFileOMP[$i]}"
    fi
    echo ""
done