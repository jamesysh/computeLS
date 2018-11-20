CC = nvcc
DEBUG = -g
CUDA_DIR = /gpfs/software/cuda/9.1/toolkit/lib64
MAGMA_DIR = /gpfs/home/shyyuan/local/magma-2.4.0/lib/
LIBS = -L $(CUDA_DIR) -L $(MAGMA_DIR) 
INCS = -I /gpfs/home/shyyuan/local/magma-2.4.0/include/

CFLAGS = $(DEBUG) $(LIBS) $(INCS) -DADD_ -c -Xlinker -lgomp -Xcompiler -fopenmp 
UPFLAGS = $(DEBUG) $(LIBS) $(INCS) -Xlinker -lgomp -Xcompiler -fopenmp 
OBJS =  main.o matrix_build.o cpu_function.o

all: computeLS
	
computeLS: $(OBJS)
		$(CC) $(UPFLAGS) $(OBJS) -o computeLS -lcudart -lcublas -lcusolver -lmagma

main.o:	main.cu matrix_build.h cpu_function.o
		$(CC) $(CFLAGS) main.cu
matrix_build.o: matrix_build.cu matrix_build.h
		$(CC) $(CFLAGS) matrix_build.cu
cpu_function.o: cpu_function.cu cpu_function.h
		$(CC) $(CFLAGS) cpu_function.cu

clean:
	rm *.o computeLS
