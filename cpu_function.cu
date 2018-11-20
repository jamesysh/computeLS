#include<iostream>
#include"cpu_function.h"
using namespace std;

__host__ void computeA3D_cpu(int index, const int *neighbourList,double* x,double* y, double* z,
int numRow, int numberNeighbourInOne, // input
					double *A, double *dis)// output
	{
         int maxNeiNum = numberNeighbourInOne; 
         double distance = sqrt((x[neighbourList[index*maxNeiNum]] - x[index]) * (x[neighbourList[index*maxNeiNum]] -
         x[index]) + (y[neighbourList[index*maxNeiNum]] - y[index]) * (y[neighbourList[index*maxNeiNum]] - y[index]) +
         (z[neighbourList[index*maxNeiNum]] - z[index]) * (z[neighbourList[index*maxNeiNum]] - z[index]));
		for(size_t i=0; i<numRow; i++) { // Note that the neighbour list does not contain the particle itself
			int neiIndex = neighbourList[index*maxNeiNum+i];	
				
			double h = (x[neiIndex] - x[index])/distance;
			double k = (y[neiIndex] - y[index])/distance;
			double l = (z[neiIndex] - z[index])/distance;

			A[i]            = h;
			A[i + 1*numRow] = k;
			A[i + 2*numRow] = l;
		}

    (*dis)=distance;

   }

__host__ void computeB_cpu(size_t index, const int *neighbourList, size_t numRow, const double* inData, size_t maxNeiNum, 
								  double *b) { // output 	
	

	for(size_t i=0; i<numRow; i++) { 
		int neiIndex = neighbourList[index*maxNeiNum+i];	
		b[i] = inData[neiIndex] - inData[index];

	}	

}


