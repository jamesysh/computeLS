#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "matrix_build.h"
#include <cublas_v2.h>
#include <magma_lapack.h>
#include <magma_v2.h>
#include "matrix_build.h"
#include "cpu_function.h"
#include <omp.h>
using namespace std;

int main(){
    int numFluid = 34486;
    int numBoundary = 142172;
    int numGhost = 83771;

    int numParticle = numFluid + numBoundary + numGhost;
      
    int numNeighbourInOne = 276;

    magma_init();
    
    magma_int_t dev_t = 0;
    magma_queue_t queue_qr = NULL;
    magma_queue_create(dev_t,&queue_qr);
 


    double* inPressure = new double[numParticle];
    double* inVolume = new double[numParticle];
    double* inSoundSpeed = new double[numParticle];
    double* inVelocity = new double[numParticle];
    int* neighbourlist0 = new int[numFluid*numNeighbourInOne];
    int* neighbourlist1 = new int[numFluid*numNeighbourInOne];
    int* neighboursize0 =new int[numFluid];
    int* neighboursize1 = new int[numFluid];
    int* LPFOrder0 = new int[numFluid];
    int* LPFOrder1 = new int[numFluid];
    double* xPosition = new double[numParticle];
    double* yPosition = new double[numParticle];
    double* zPosition = new double[numParticle]; 

    ifstream myfile;
    
    myfile.open("data/xPosition.txt");
    for(int i=0;i<numParticle;i++){
        double tem;
        myfile>>tem;
        xPosition[i] = tem;
    }
    myfile.close();
    
    myfile.open("data/yPosition.txt");
    for(int i=0;i<numParticle;i++){
        double tem;
        myfile>>tem;
        yPosition[i] = tem;
    }
    myfile.close();
    
    myfile.open("data/zPosition.txt");
    for(int i=0;i<numParticle;i++){
        double tem;
        myfile>>tem;
        zPosition[i] = tem;
    }
    myfile.close();



   myfile.open("data/inPressure.txt");
    for(int i=0;i<numParticle;i++){
        double tem;
        myfile>>tem;
        inPressure[i]=tem;
    }
    myfile.close();
    
    myfile.open("data/inVelocity.txt");
    for(int i=0;i<numParticle;i++){
        double tem;
        myfile>>tem;
        inVelocity[i]=tem;
    }
    myfile.close();
    
    myfile.open("data/inSoundSpeed.txt");
    for(int i=0;i<numParticle;i++){
       double tem;
        myfile>>tem;
        inSoundSpeed[i]=tem;
    }
    myfile.close();
   
    myfile.open("data/inVolume.txt");
    for(int i=0;i<numParticle;i++){
       double tem;
        myfile>>tem;
        inVolume[i]=tem;
    }
    myfile.close();
   
  myfile.open("data/neighbourlist0.txt");
    for(int i=0;i<numFluid*numNeighbourInOne;i++){
        int tem;
        myfile>>tem;
        neighbourlist0[i]=tem;
    }
    myfile.close();
    myfile.open("data/neighbourlist1.txt");
    for(int i=0;i<numFluid*numNeighbourInOne;i++){
        int tem;
        myfile>>tem;
        neighbourlist1[i]=tem;
    }
    myfile.close();
      myfile.open("data/neighboursize0.txt");
    for(int i=0;i<numFluid;i++){
       double tem;
        myfile>>tem;
        neighboursize0[i]=tem;
    }
    myfile.close();
     myfile.open("data/neighboursize1.txt");
    for(int i=0;i<numFluid;i++){
       double tem;
        myfile>>tem;
        neighboursize1[i]=tem;
    }
    myfile.close();


    fill_n(LPFOrder0,numFluid,1);
    fill_n(LPFOrder1,numFluid,1);





    double* d_xPosition;
    double* d_yPosition;
    double* d_zPosition;
    double* d_inPressure;
    double* d_inVolume;
    double* d_inSoundSpeed;
    double* d_inVelocity;
    int* d_neighbourlist0;
    int* d_neighbourlist1;
    int* d_neighboursize0;
    int* d_neighboursize1;
    int* d_LPFOrder0;
    int* d_LPFOrder1;
 

       
    cudaMalloc((void**)&d_xPosition,sizeof(double)*numParticle);
    cudaMalloc((void**)&d_yPosition,sizeof(double)*numParticle);
    cudaMalloc((void**)&d_zPosition,sizeof(double)*numParticle);
    cudaMalloc((void**)&d_inPressure,sizeof(double)*numParticle);
    cudaMalloc((void**)&d_inVolume,sizeof(double)*numParticle );
    cudaMalloc((void**)&d_inVelocity, sizeof(double)*numParticle);
    cudaMalloc((void**)&d_inSoundSpeed,sizeof(double)*numParticle);
    cudaMalloc((void**)&d_neighbourlist0,sizeof(int)*numFluid*numNeighbourInOne);
    cudaMalloc((void**)&d_neighbourlist1,sizeof(int)*numFluid*numNeighbourInOne);
    cudaMalloc((void**)&d_neighboursize0,sizeof(int)*numFluid);
    cudaMalloc((void**)&d_neighboursize1,sizeof(int)*numFluid);
    cudaMalloc((void**)&d_LPFOrder0,sizeof(int)*numFluid);
    cudaMalloc((void**)&d_LPFOrder1,sizeof(int)*numFluid);



    cudaMemcpy(d_xPosition,xPosition,sizeof(double)*numParticle,cudaMemcpyHostToDevice);
    cudaMemcpy(d_yPosition,yPosition,sizeof(double)*numParticle,cudaMemcpyHostToDevice);
    cudaMemcpy(d_zPosition,zPosition,sizeof(double)*numParticle,cudaMemcpyHostToDevice);
    cudaMemcpy(d_inPressure,inPressure,sizeof(double)*numParticle,cudaMemcpyHostToDevice);
    cudaMemcpy(d_inVolume,inVolume,sizeof(double)*numParticle,cudaMemcpyHostToDevice);
    cudaMemcpy(d_inVelocity,inVelocity,sizeof(double)*numParticle,cudaMemcpyHostToDevice);
    cudaMemcpy(d_inSoundSpeed,inSoundSpeed,sizeof(double)*numParticle,cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbourlist0,neighbourlist0,sizeof(int)*numFluid*numNeighbourInOne,cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbourlist1,neighbourlist1,sizeof(int)*numFluid*numNeighbourInOne,cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighboursize0,neighboursize0,sizeof(int)*numFluid,cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighboursize1,neighboursize1,sizeof(int)*numFluid,cudaMemcpyHostToDevice);
    cudaMemcpy(d_LPFOrder0,LPFOrder0,sizeof(int)*numFluid,cudaMemcpyHostToDevice);
    cudaMemcpy(d_LPFOrder1,LPFOrder1,sizeof(int)*numFluid,cudaMemcpyHostToDevice);






    double** A;
    cudaMalloc((void**)&A,sizeof(double*)*numFluid);
    double** A_temp = new double*[numFluid];
   for(int i=0;i<numFluid;i++){
        cudaMalloc((void**)&A_temp[i],sizeof(double)*5*8);
    }
    cudaMemcpy(A, A_temp,sizeof(double*)*numFluid,cudaMemcpyHostToDevice);

    double* d_distance;
    cudaMalloc((void**)&d_distance,sizeof(double)*numFluid);
 


    cudaError err = cudaGetLastError();
    if(cudaSuccess != err){
            printf("Error occurs when setting up particle data on GPU!!! MSG: %s\n",cudaGetErrorString(err));
            assert(false);
        }
    int numRow = 8;
    int numCol = 3;
//--------------------gpu computeA------------------------------------

    computeA3D_gpu<<<96,64>>>(d_neighbourlist0 ,d_LPFOrder0 ,numRow ,d_xPosition ,d_yPosition ,d_zPosition ,0 ,numFluid ,numNeighbourInOne ,
            A ,d_distance);
for(int i=numFluid-1;i<numFluid;i++){
   cout<<"A of number: "<<i<<endl;
   magma_dprint_gpu(8,3,A_temp[3],8,queue_qr);

   }

//---------------------------computeB---------------------------------
    double** B;
    cudaMalloc((void**)&B,sizeof(double*)*numFluid);
    double** B_temp = new double*[numFluid];
    for(int i=0;i<numFluid;i++){
        cudaMalloc((void**)&B_temp[i],sizeof(double)*numRow);
    }
    cudaMemcpy(B,B_temp,sizeof(double*)*numFluid,cudaMemcpyHostToDevice);



    computeB_gpu<<<96,64>>>(d_neighbourlist0, numRow, d_inPressure, 0, numFluid, numNeighbourInOne, B);
     for(int i=numFluid-1;i<numFluid;i++){
           cout<<"B of number: "<<i<<endl;
           magma_dprint_gpu(8,1,B_temp[3],8,queue_qr);

       }  


    cublasHandle_t handle;
    cublasCreate(&handle);
    int info_gpu[numFluid];

   
    cudaEvent_t start,stop;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for(int i=0;i<1;i++){
    cudaEventRecord(start,0);
   
    cublasStatus_t succ = cublasDgelsBatched(handle,CUBLAS_OP_N,numRow,numCol,1,A,numRow,B,numRow,info_gpu,NULL,numFluid);
 cudaEventRecord(stop,0);
    cudaEventSynchronize(stop); 
    
    float gputime;
    cudaEventElapsedTime(&gputime,start,stop);

    printf("GPU time of computeLS is %f(ms)\n",gputime);
    if(succ == CUBLAS_STATUS_SUCCESS)
        cout<<"success"<<endl;
    
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
   for(int i=numFluid-1;i<numFluid;i++){
           cout<<"B of number: "<<i<<endl;
           magma_dprint_gpu(8,1,B_temp[3],8,queue_qr);

       }  




//--------------cpu computeA------------------------------------
    double distance_cpu = 0;
    double* A_cpu = new double[40];

    
   /*
    #ifdef _OPENMP
    #pragma omp parallel for 
    #endif
    */
    for(int index=0;index<numFluid;index++){
        computeA3D_cpu(index, neighbourlist0, xPosition,yPosition, zPosition, numRow,  numNeighbourInOne, // input
					A_cpu, &distance_cpu);
        }
  
/*   
for(int i=numFluid-1;i<numFluid;i++){
   cout<<"A of number: "<<i<<endl;
   magma_dprint_gpu(24,1,A_temp[i],24,queue_qr);

   }

cout<<"cpu results"<<endl;
for(int i=0;i<24;i++){
    cout<<A_cpu[i]<<endl;
    
    }
*/

//-----------------------cpu computeB-------------------------

    double* B_cpu = new double[8];
for(int i=0; i<1; i++){
    double start_cpu = omp_get_wtime();
    for(int index=0;index<numFluid-1977;index++){
       computeB_cpu(index, neighbourlist0, numRow, inPressure, numNeighbourInOne, B_cpu);
        
        }
    double stop_cpu = omp_get_wtime();
    double time_cpu = stop_cpu - start_cpu;
    cout<<"CPU time of computeB is "<<time_cpu*1000<<"(ms)"<<endl;  
}
/*
for(int i=numFluid-1978;i<numFluid-1977;i++){
   cout<<"A of number: "<<i<<endl;
   magma_dprint_gpu(numRow,1,B_temp[i],numRow,queue_qr);

   }

cout<<"cpu results"<<endl;
for(int i=0;i<8;i++){
    cout<<B_cpu[i]<<endl;
    
    }
*/

    cudaFree(d_neighboursize0);
    cudaFree(d_neighboursize1);
    cudaFree(d_neighbourlist0);
    cudaFree(d_neighbourlist1);
    cudaFree(d_LPFOrder0);
    cudaFree(d_LPFOrder1);
    cudaFree(d_inPressure);
    cudaFree(d_inVolume);
    cudaFree(d_inSoundSpeed);
    cudaFree(d_inVelocity);
    cudaFree(A);
    cudaFree(d_xPosition);
    cudaFree(d_yPosition);
    cudaFree(d_zPosition);
    cudaFree(B); 
    for(int i=0;i<numParticle;i++){
        cudaFree(A_temp[i]);
        cudaFree(B_temp[i]);
    }
    cudaFree(d_distance);
 




    delete[] inPressure;
    delete[] inVolume;
    delete[] inVelocity;
    delete[] inSoundSpeed;
    delete[] neighbourlist0;
    delete[] neighbourlist1;
    delete[] neighboursize0;
    delete[] neighboursize1;
    delete[] LPFOrder0;
    delete[] LPFOrder1;
    delete[] xPosition;
    delete[] yPosition;
    delete[] zPosition;
    delete[] A_cpu; 
    delete[] B_cpu;

    magma_queue_destroy(queue_qr);
    magma_finalize();


}
