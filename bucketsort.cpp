#include <mpi.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <cmath>

using namespace std;

#define DEBUG

void debug(const char *format, ...) {
  #ifdef DEBUG
  va_list args;
  va_start(args, format);
  vfprintf(stdout, "DEBUG :: ", NULL);
  vfprintf(stdout, format, args);
  vfprintf(stdout, "\n", NULL);
  va_end(args);
  fflush(stdout);
  #endif
}
     
// Power function with overflow consideration
long power(long base, long exp) {
  long retVal = 1;
  int i = 0;
  for (i; i < exp; i++) {
    retVal = (retVal * base);
  }
  return retVal;
}

// Power function with modulo and overflow consideration
long powerMod(long base, long exp, long modulo) {
  long retVal = 1;
  int i = 0;
  for (i; i < exp; i++) {
    retVal = (retVal * base) % modulo;
  }
  return retVal;
}

// Implementation of the random number generation algorithm
int generateRandomNumber(long aConst, int previousVal, long cConst, long modulo) {
  long val = (aConst * previousVal + cConst)% modulo;
  int retVal = (int)val;
  return retVal;
}

void initializeList(std::vector<int>* list, int starter, int size) {
  long modulo = power(2, 31), a = 1664525, c = 1013904223;
  list->push_back(starter);
  for (int i = 1; i < size; i++) {
    int num = generateRandomNumber(a, list->data()[i - 1], c, modulo);
    list->push_back(num);
  }
} 

void initializeTestList(std::vector<int>* list, int starter, int size) { 
  int count = 50;
  for (int i = 0; i < size; i++) {
    list->push_back((starter + count) % 99);
    count++;
  }
}

void initializeListRand(std::vector<int>* list, int max, int size) {
  srand(time(NULL));
  for (int i = 0; i < size; i++) {
    list->push_back(rand() % max);
  }
}

void printVector(vector<int>* vec) {
  for (vector<int>::iterator it = vec->begin(); it != vec->end(); it++) {
    printf("%d\n", *it);
  }
}

int* generateRecvSizeBuffer(int n_proc, int size) {
  int *arr = new int[n_proc];
  arr[0] = size;
  for (int i = 1; i < n_proc; i++) {
    arr[i] = 0;
  }
  return arr;
}

int compare(const void *a, const void *b) {
  return (*(int *)a - *(int *)b);
}

int main(int argc, char **argv) {
  // Initialise MPI
  MPI::Init(argc, argv);
  int n_proc = MPI::COMM_WORLD.Get_size(); // Number of processors
  int rank   = MPI::COMM_WORLD.Get_rank(); // Rank of current processor

  double initialTime = MPI::Wtime(), endTime;

  long range = power(2, 31);               // My test range

  // This is how many numbers to generate and sort
  int N = atoi(argv[1]);
  
  vector<int> numbers;                     // List of random numbers
  vector<int> dropsInBucket(n_proc);       // Count of how many numbers in each processors bucket
  vector<int> sendOffsets(n_proc);         // Count of stride between values

  int remainder = N % n_proc;
  int drops = N/n_proc;

  if (rank == 0) {
    initializeList(&numbers, 12345, N);
  }

  // Set how many values are sent to each process
  for (int i = 0; i < n_proc; i++) {
    dropsInBucket[i] = drops;
    if ((remainder != 0) && (i < remainder)) {
      dropsInBucket[i] += 1;
    }
  }

  // Set the stride for the scatter and initialise sendCount
  sendOffsets.push_back(0);     
  for (int i = 1; i < n_proc; i++) {
    sendOffsets[i] = sendOffsets[i - 1] + dropsInBucket[i - 1];
  }
  
  vector<int> initialBucket(dropsInBucket[rank]);
  MPI::COMM_WORLD.Scatterv(&numbers[0], &dropsInBucket[0], &sendOffsets[0], MPI::INT, 
                           &initialBucket[0], N, MPI::INT, 0);

  // ------------------------------------------------------------

  vector<int> segBucket(N, 0);
  vector<int> sendCount(n_proc, 0);
  vector<int> recvCount(n_proc, 0);
  vector<int> recvOffsets(n_proc, 0);
  int rangeUnit = ceil((double)range/(double)n_proc);

  // Put elements into small buckets and determine which process to send them to
  for (int i = 0; i < initialBucket.size(); i++) {
    int pool = (int)(initialBucket[i]/rangeUnit);
    segBucket[sendOffsets[pool] + sendCount[pool]] = initialBucket[i];
    sendCount[pool]++;
  }

  // Tell each process what it will receive from the other processes
  MPI::COMM_WORLD.Alltoall(&sendCount[0], 1, MPI::INT, &recvCount[0], 1, MPI::INT);

  // Get the size of the last bucket and receive offsets 
  int finalBucketSize = 0;
  for (int i = 0; i < recvCount.size(); i++) {
    recvOffsets[i] = finalBucketSize;
    finalBucketSize += recvCount[i];
  }

  // Bucket to keep numbers for sorting
  vector<int> sortBucket(finalBucketSize, 0);

  MPI::COMM_WORLD.Alltoallv(&segBucket[0],  &sendCount[0], &sendOffsets[0], MPI::INT, 
                            &sortBucket[0], &recvCount[0], &recvOffsets[0], MPI::INT);

  qsort(&sortBucket[0], sortBucket.size(), sizeof(int), compare);

  endTime = MPI::Wtime();

  if (rank == 0) {
    debug("p-%d - time: %f seconds", n_proc, endTime - initialTime);  
  }  

  MPI::Finalize();
}