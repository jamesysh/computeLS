#include <iostream>
#include "matrix_build.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdio.h>




__global__ void computeA2D_gpu(const int*neighbourList,const int*LPFOrder, int numRow,const double*x,const
double*y,int startIndex, int numComputing, int maxNeighbourOneDir,
        double**A,double*dis)//output
    {
        
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int offset = blockDim.x*gridDim.x;
    //printf("runing form %d\n",tid);
    while(tid<numComputing){
        int index = startIndex+tid;
        int numOfRow = numRow;
        double distance = sqrt((x[neighbourList[index*maxNeighbourOneDir+numOfRow/4]]-x[index])*(x[neighbourList[index*maxNeighbourOneDir+numOfRow/4]]-x[index])+(y[neighbourList[index*maxNeighbourOneDir+numOfRow/4]]-y[index])*(y[neighbourList[index*maxNeighbourOneDir+numOfRow/4]]-y[index]));
       if(LPFOrder[index] == 1){
            
            for(int i=0;i<numOfRow;i++){
            
                int neiIndex = neighbourList[index*maxNeighbourOneDir+i];
                    
                double h = (x[neiIndex]-x[index])/distance;
                double k = (y[neiIndex]-y[index])/distance;
                A[tid][i] = h;//Notice it should be A[tid] because A are assigned by order
                A[tid][i+numOfRow] = k;
            }   
    
        }
        else if(LPFOrder[index] == 2){
            for(int i=0;i<numOfRow;i++){
                int neiIndex = neighbourList[index*maxNeighbourOneDir+i];
                double h = (x[neiIndex]-x[index])/distance;
                double k = (y[neiIndex]-y[index])/distance;
                A[tid][i] = h;
                A[tid][i + numOfRow] = k;
                A[tid][i + 2*numOfRow] = 0.5*h*h;
                A[tid][i + 3*numOfRow] = 0.5*k*k;
                A[tid][i + 4*numOfRow] = h*k;
            }
        
        } 
    dis[tid] = distance;
    tid = tid + offset;
    }
}

__global__ void computeA3D_gpu(const int*neighbourList,const int* LPFOrder, int numRow, const double*x, const double*y,
const double*z, int startIndex, int numComputing, int maxNeighbourOneDir,
        double**A,double*dis)//output
   {
        
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int offset = blockDim.x*gridDim.x;
    //printf("runing form %d\n",tid);
    while(tid < numComputing){
        int index = tid + startIndex;
        int numOfRow = numRow;
        double x0 = x[index];
        double y0 = y[index];
        double z0 = z[index];
        int nei1 = neighbourList[index*maxNeighbourOneDir+0];
        double x1 = x[nei1];
        double y1 = y[nei1];
        double z1 = z[nei1];
        int nei2 = neighbourList[index*maxNeighbourOneDir+1];
        double x2 = x[nei2];
        double y2 = y[nei2];
        double z2 = z[nei2];
        int nei3 = neighbourList[index*maxNeighbourOneDir+2];
        double x3 = x[nei3];
        double y3 = y[nei3];
        double z3 = z[nei3];
        int nei4 = neighbourList[index*maxNeighbourOneDir+3];
        double x4 = x[nei4];
        double y4 = y[nei4];
        double z4 = z[nei4];
        int nei5 = neighbourList[index*maxNeighbourOneDir+4];
        double x5 = x[nei5];
        double y5 = y[nei5];
        double z5 = z[nei5];
        int nei6 = neighbourList[index*maxNeighbourOneDir+5];
        double x6 = x[nei6];
        double y6 = y[nei6];
        double z6 = z[nei6];
        int nei7 = neighbourList[index*maxNeighbourOneDir+6];
        double x7 = x[nei7];
        double y7 = y[nei7];
        double z7 = z[nei7];
        int nei8 = neighbourList[index*maxNeighbourOneDir+7];
        double x8 = x[nei8];
        double y8 = y[nei8];
        double z8 = z[nei8];
 

        double distance = sqrt((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0)+(z1-z0)*(z1-z0));
        if(LPFOrder[index] == 1){
           
            A[tid][0] = (x1-x0)/distance;
            A[tid][1] = (x2-x0)/distance;
            A[tid][2] = (x3-x0)/distance;
            A[tid][3] = (x4-x0)/distance;
            A[tid][4] = (x5-x0)/distance;
            A[tid][5] = (x6-x0)/distance;
            A[tid][6] = (x7-x0)/distance;
            A[tid][7] = (x8-x0)/distance;
            A[tid][8] = (y1-y0)/distance;
            A[tid][9] = (y2-y0)/distance;
            A[tid][10] = (y3-y0)/distance;
            A[tid][11] = (y4-y0)/distance;
            A[tid][12] = (y5-y0)/distance;
            A[tid][13] = (y6-y0)/distance;
            A[tid][14] = (y7-y0)/distance;
            A[tid][15] = (y8-y0)/distance;
            A[tid][16] = (z1-z0)/distance;
            A[tid][17] = (z2-z0)/distance;
            A[tid][18] = (z3-z0)/distance;
            A[tid][19] = (z4-z0)/distance;
            A[tid][20] = (z5-z0)/distance;
            A[tid][21] = (z6-z0)/distance;
            A[tid][22] = (z7-z0)/distance;
            A[tid][23] = (z8-z0)/distance;
    
        }
        else if(LPFOrder[index] == 2){
            for(int i=0;i<numOfRow;i++){
                int neiIndex = neighbourList[index*maxNeighbourOneDir+i];
                double h = (x[neiIndex]-x[index])/distance;
                double k = (y[neiIndex]-y[index])/distance;
                double l = (z[neiIndex] - z[index])/distance;
              
                A[tid][i] = h;
                A[tid][i + numOfRow] = k;
                A[tid][i + 2*numOfRow] = l;
                A[tid][i + 3*numOfRow] = 0.5*h*h;
                A[tid][i + 4*numOfRow] = 0.5*k*k;
                A[tid][i + 5*numOfRow] = 0.5*l*l;
                A[tid][i + 6*numOfRow] = h*k;
                A[tid][i + 7*numOfRow] = h*l;
                A[tid][i + 8*numOfRow] = k*l;

            }
        
        } 
    dis[tid] = distance;
    tid = tid + offset;
    }
}


__global__ void computeB_gpu(const int* neighbourList, int numRow, const double* inData, int startIndex, int
numComputing, const int maxNeighbourOneDir,
                        double** b)//output vector b

{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int offset = blockDim.x*gridDim.x;
    while(tid<numComputing){
        int index = tid + startIndex;
      
        double in0 = inData[index];
        int nei1 = neighbourList[index*maxNeighbourOneDir+0];
        double in1 = inData[nei1];
        int nei2 = neighbourList[index*maxNeighbourOneDir+1];
        double in2 = inData[nei2];
        int nei3 = neighbourList[index*maxNeighbourOneDir+2];
        double in3 = inData[nei3];
        int nei4 = neighbourList[index*maxNeighbourOneDir+3];
        double in4 = inData[nei4];
        int nei5 = neighbourList[index*maxNeighbourOneDir+4];
        double in5 = inData[nei5];
        int nei6 = neighbourList[index*maxNeighbourOneDir+5];
        double in6 = inData[nei6];
        int nei7 = neighbourList[index*maxNeighbourOneDir+6];
        double in7 = inData[nei7];
        int nei8 = neighbourList[index*maxNeighbourOneDir+7];
        double in8 = inData[nei8];
    

        b[tid][0] = in1 - in0;
        b[tid][1] = in2 - in0;
        b[tid][2] = in3 - in0;
        b[tid][3] = in4 - in0;
        b[tid][4] = in5 - in0;       
        b[tid][5] = in6 - in0;
        b[tid][6] = in7 - in0;
        b[tid][7] = in8 - in0;  

/*       double in0 = inData[index];
       for(int i=0;i<numRow;i++){
            int neiIndex = neighbourList[index*maxNeighbourOneDir + i];
            b[tid][i] = inData[neiIndex] - in0;
        }
  */  
        tid = tid + offset;
    }
}


__global__ void computeLS_gpu(double**A,double**B,double**Tau, int numRow, int numCol, int numComputing, 
                        double**Result)//output result
{

    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int offset = blockDim.x*gridDim.x;
    while(tid < numComputing){
        int nrow = numRow;
        int ncol = numCol;
        for(int i=0;i<ncol;i++){
            double v_times_b = 0.;
            for(int j=0;j<nrow;j++){
                if(j < i) continue;
                if(j == i) v_times_b += 1*B[tid][j];
               
                else v_times_b += A[tid][j+i*nrow]*B[tid][j];
            }
            v_times_b *= Tau[tid][i];

            for(int j=0;j<nrow;j++){
                if(j < i) continue;
                if(j == i) B[tid][j] -= v_times_b;
                else
                B[tid][j] -= v_times_b*A[tid][j+i*nrow];
           }

        }


//compute QTB complete

//Backsubstitution
        for(int i=ncol-1;i>=0;i--){
          Result[tid][i] = B[tid][i]/(A[tid][i*nrow+i]);
               for(int j=0;j<i;j++){
                
                   B[tid][j] -= A[tid][j+i*nrow]*Result[tid][i];
           }
        
        }
    tid += offset;
    
    }
}

__global__ void assignValue_gpu(const int* LPFOrder, int* valueAssigned,  double** Result, const double* distance, int numComputing, int startIndex,
int dir, int offset, double* d1st, double* d2nd)//output
{         
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int offset_in = blockDim.x*gridDim.x;
    while(tid<numComputing){
        int index = tid + startIndex;
        if(LPFOrder[index] == 1 && valueAssigned[index] == 0 ){
                 d1st[index] = Result[tid][dir]/distance[tid];
                 d2nd[index] = 0;

        }
         else if(LPFOrder[index] == 2 && valueAssigned[index] == 0 ){
                 d1st[index] = Result[tid][dir]/distance[tid];
                 d2nd[index] = Result[tid][dir+offset]/distance[tid]/distance[tid];
             }
              
        tid += offset_in;
        
    }

}

    





__global__ void checkSoundspeedAndVolume(double* inSoundSpeed, double* outSoundSpeed, double* inVolume, double* outVolume, double* inPressure,
double* outPressure, double* inVelocity, double* outVelocity, int numFluid){
    
    
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int offset = blockDim.x*gridDim.x;
      
    while(tid < numFluid){
        
        int index = tid;
        if(inSoundSpeed[tid] == 0 || inVolume[tid] == 0){
        printf("The %d particle has 0 soundspeed or involume", tid); 
       	outVolume[index]   = inVolume[index];
		outPressure[index] = inPressure[index];
		outVelocity[index] = inVelocity[index];
		outSoundSpeed[index] = inSoundSpeed[index];
            
        }

    tid += offset;
 }
    
}

__global__ void checkLPFOrder_gpu(const int* neighboursize, int* LPFOrder, double* vel_d, double* vel_dd, double* p_d,
double* p_dd, int* valueAssigned, int* warningCount,  int numFluid,int numRow){
    
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int offset = blockDim.x*gridDim.x;
   
    while(tid < numFluid){
        int numNeisize =  neighboursize[tid];
        if(valueAssigned[tid] == 0 ){
            if(LPFOrder[tid]==2){
                if(numNeisize < numRow){
               
                      LPFOrder[tid] = 1;
                     }    
            } 
           if(LPFOrder[tid]==1){
               if(numNeisize < numRow){
               
                 LPFOrder[tid] = 0;

              }
         }
            if(LPFOrder[tid] == 0){
            vel_d[tid] = 0;
            vel_dd[tid] = 0;
            p_d[tid] = 0;
            p_dd[tid] = 0;

            valueAssigned[tid] = 1;
            atomicAdd(warningCount,1);
            printf("The particle of Index %d has 0 order!!!!! neighbourSize is %d\n",tid,numNeisize);
            }

        }
        tid += offset;
        
        }
    }

__global__ void checkInvalid_gpu(int* valueAssigned, int* info, double* p_d, double* p_dd,  double* vel_d, double*
vel_dd,int startIndex ,int numComputing, int* warningCount ){
    
    
        int tid = threadIdx.x + blockIdx.x*blockDim.x;
        int offset = blockDim.x*gridDim.x;

        while(tid < numComputing){
            
           int index = tid + startIndex;
           if((!(isnan(p_d[index]) || isnan(p_dd[index]) || isnan(vel_d[index]) || isnan(vel_dd[index]) ||
			  isinf(p_d[index]) || isinf(p_dd[index]) || isinf(vel_d[index]) || isinf(vel_dd[index]) ||
              info[tid] != 0)) && valueAssigned[index] == 0)
              {  
                valueAssigned[index] = 1;
                 
                atomicAdd(warningCount,1);

           }
   /*             
           else{
               if(valueAssigned[index] == 0){
               valueAssigned[index] = 1;
               atomicAdd(warningCount,1);
            }
                         }*/
            tid += offset; 

            }
 
}

__global__ void timeIntegration_gpu( 
        double realDt, double multiplier1st, double multiplier2nd, int numFluid,
        double gravity,const double* inVolume,const double* inVelocity,const double* inPressure,const double* inSoundSpeed,
        double* vel_d_0, double* vel_dd_0, double* p_d_0, double* p_dd_0,
        double* vel_d_1, double* vel_dd_1, double* p_d_1, double* p_dd_1,
        double* outVolume, double* outVelocity, double* outPressure, double* outSoundSpeed, int* info )
{
    
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int offset = blockDim.x*gridDim.x;
    while(tid < numFluid){
        double soundspeed = inSoundSpeed[tid];
        double velocity = inVelocity[tid];
        double pressure = inPressure[tid];
        double volume = inVolume[tid];


  if(soundspeed == 0 || volume == 0  )
        {   printf("wrong!!!\n");
            outVolume[tid] = volume;
            outPressure[tid] = pressure;
            outVelocity[tid] = velocity;
            outSoundSpeed[tid] = soundspeed;
            }
        else{
 

        double v_d_0_t = vel_d_0[tid];
        double v_d_1_t = vel_d_1[tid];
        double p_d_0_t = p_d_0[tid];
        double p_d_1_t = p_d_1[tid];
        double v_dd_0_t = vel_dd_0[tid];
        double v_dd_1_t = vel_dd_1[tid];
        double p_dd_0_t = p_dd_0[tid];
        double p_dd_1_t = p_dd_1[tid];

        double K = soundspeed*soundspeed/volume/volume;
       //this K only works for poly and spoly eos
        //Pt
        double Pt1st = -0.5*volume*K*(v_d_0_t+v_d_1_t) +
            0.5*volume*sqrt(K)*(p_d_0_t-p_d_1_t);
             double Pt2nd = -volume*volume*pow(K,1.5)*(v_dd_0_t-v_dd_1_t) + 
            volume*volume*K*(p_dd_0_t+p_dd_1_t);
        double Pt = multiplier1st*Pt1st + multiplier2nd*Pt2nd;

        //vt
        double Vt = -Pt/K;
       

        //VELt
        double VELt1st = 0.5*volume*sqrt(K)*(v_d_0_t-v_d_1_t) - 
            0.5*volume*(p_d_0_t-p_d_1_t);
        double VELt2nd = volume*volume*(v_dd_0_t+v_dd_1_t) - 
            volume*volume*sqrt(K)*(p_dd_0_t-p_dd_1_t);
        double VELt = multiplier1st*VELt1st + multiplier2nd*VELt2nd;

        outVolume[tid] = volume + realDt*Vt;
        outPressure[tid] = pressure + realDt*Pt;
        outVelocity[tid] = velocity+ realDt*(VELt+gravity);
}
        if( isnan(outVolume[tid]) || isinf(outVolume[tid]) ||
            isnan(outPressure[tid]) || isinf(outPressure[tid]) ||
            isnan(outVelocity[tid]) || isinf(outVelocity[tid])
          )
          {  info[0] = 1;}
        
        tid += offset;
    }



}

__global__ void initLPFOrder_upwind_gpu(int* LPFOrder0, int* LPFOrder1,  int numFluid){
    
     int tid = threadIdx.x + blockIdx.x*blockDim.x;
     int offset = blockDim.x*gridDim.x;

    while(tid < numFluid){
        
        LPFOrder0[tid] = 1;

        LPFOrder1[tid] = 1;
        
        tid += offset;
        
        }
    
    
    }

__global__ void updateOutPressureForPellet_gpu(const double* Deltaq, double* outPressure, double realDt, int m_pGamma, int numFluid, int* info){
      int tid = threadIdx.x + blockIdx.x*blockDim.x;
      int offset = blockDim.x*gridDim.x;
      while(tid < numFluid){
          
          outPressure[tid] += realDt*Deltaq[tid]*(m_pGamma-1);
          
          if(isnan(outPressure[tid]) || isinf(outPressure[tid])){
              info[0] = 1;
              
              }

          tid += offset; 
          }
      
    }
__global__ void checkPressureAndDensity_gpu(double* outPressure, double* outVolume, double* outVelocity,
double* outSoundSpeed, const double* inPressure, const double* inVelocity, const double* inVolume, const double*
inSoundSpeed, int m_fInvalidPressure, int m_fInvalidDensity, int numFluid ){
    
      int tid = threadIdx.x + blockIdx.x*blockDim.x;
      int offset = blockDim.x*gridDim.x;
      while(tid < numFluid){
           
        if(outPressure[tid]<m_fInvalidPressure || (outVolume[tid]!=0&&1./outVolume[tid]<m_fInvalidDensity))
            {
                   outVolume[tid]   = inVolume[tid];
                   outPressure[tid] = inPressure[tid];
                   outVelocity[tid] = inVelocity[tid];
                   outSoundSpeed[tid] = inSoundSpeed[tid];
           printf("The particle of index %d has 0 order!\n",tid);             
                
                }              
         tid += offset;
          }
    }

__global__ void updateSoundSpeed_gpu(const double* outPressure,const double* outVolume, double* outSoundSpeed, double
m_fGamma, int numFluid, int* info){
    
      int tid = threadIdx.x + blockIdx.x*blockDim.x;
      int offset = blockDim.x*gridDim.x;
    while(tid < numFluid){
    
        double cs;
        double density = 1./outVolume[tid];
        if(density != 0){
            cs = m_fGamma * outPressure[tid] / density;
            if (cs > 0)
                outSoundSpeed[tid] = sqrt(cs);
            else if(cs == 0)
                outSoundSpeed[tid] = 0;

            else
               { printf("Taking sqrt of a negative cs!\n");
                info[0] = 1;}
            }
        else {
            printf("O density!!!\n");
            info[0] = 1;}
        
        tid += offset;
        }
    
    }


