#include <iostream>

__global__
void square(float *d_in, float *d_out)
{
    d_out[threadIdx.x] = d_in[threadIdx.x] * d_in[threadIdx.x];
}

int main(int argc, char *argv[])
{
    const int arraysize = 64;
    const int bytesize = sizeof(float) * arraysize;
    float h_in[64];
    float h_out[64];
    for (int i = 0; i<bytesize; i++){
        h_in[i] = i;
    }
    float *d_in;
    float *d_out;

    cudaMalloc((void **) &d_in, bytesize);
    cudaMalloc((void **) &d_out, bytesize);
    cudaMemcpy(d_in, h_in, bytesize, cudaMemcpyHostToDevice);

    square<<< 1, arraysize >>>(d_in, d_out);

    cudaMemcpy(h_out, d_out, bytesize, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    for (int i = 0; i < arraysize; i += 4){
        std::cout << h_out[i] << "\t" << h_out[i+1] << "\t" << h_out[i+2] << "\t" << h_out[i+3] << std::endl;
    }

    return 0;
}

