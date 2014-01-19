#include <stdio.h>

__global__
void kernel()
{
    printf("Hier is Thread (%d, %d), (%d, %d)\n",
            blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

int main(int argc, char *argv[])
{
    dim3 blockDim(3, 3, 1);
    dim3 gridDim(2, 2, 1);
    kernel<<< gridDim, blockDim, 0 >>>();
    return 0;
}

