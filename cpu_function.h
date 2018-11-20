#ifndef CPU_FUNCTION
#define CPU_FUNCTION
#include <cuda.h>
#include <cuda_runtime_api.h>


__host__ void computeA3D_cpu(int index, const int *neighbourList,double* x,double* y, double* z, int numRow, int numberNeighbourInOne, // input
					double *A, double *distance); // output
	



__host__ void computeB_cpu(size_t index, const int *neighbourList, size_t numRow, const double* inData, size_t maxNeiNum, 
								  double *b); // output 	
	












#endif
