#include <stdio.h>

__global__ void cuda_hello(){
	int id = blockDim.x*blockIdx.x+ threadIdx.x;
	printf("Hello world from GPU %d! \n",id);
}


int main(){

	cuda_hello<<<2,3>>>();
	cudaDeviceSynchronize();
	return 0;
}
