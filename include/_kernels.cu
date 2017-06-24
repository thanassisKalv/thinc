#include <stdio.h>



void __global__
maxout(float* best__bo, int* which__bo,
        const float* cands__bop, int B, int O, int P)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x; 
    if (b >= B) return;

    for (int o=0; o < O; ++o)
    {
        which__bo[0] = 0;
        best__bo[0] = cands__bop[0];
        cands__bop += 1;
        for (int p=1; p < P; ++p)
	{
            if (cands__bop[0] > best__bo[0])
	    {
                which__bo[0] = p;
                best__bo[0] = cands__bop[0];
	    }
            cands__bop += 1;
	}
        best__bo += 1;
        which__bo += 1;
    }
}



void __global__
mean_pool(float* means__bo,
    const float* X__to, const int* lengths__b, int B, int T, int O)
{
	// Each CUDA block computes means of a batch of concatenated sequences, using the lengths. 
    	int bid = blockIdx.x;
	if(threadIdx.x>=O)
		return;
	__shared__ float local_means[512];	// can be bigger, depends on dimensions

	// At each step it keeps track of the total length of all previous batches (even those processed 
	// other CUDA blocks)
	int prevLengths = 0;
	for(int i = 0; i<bid; i++)
		prevLengths+=lengths__b[i];

	// Batch-items are processed by a fixed number of launched CUDA blocks
	// with a step equal to the total number gridDim.x
    	for(int step = bid; step < B; step += gridDim.x )
	{
		int lengthOfBatch = lengths__b[step];
		int batchStarts = prevLengths*O; 
		float scale = 1.0/(float)lengthOfBatch;
		local_means[threadIdx.x] = 0.0;

		for (int i = batchStarts + threadIdx.x; i < batchStarts+(lengthOfBatch*O) ; i += O)
			local_means[threadIdx.x] += X__to[i]*scale;
		
		__syncthreads();	// Block-wise synchronization

		means__bo[step*O + threadIdx.x] = local_means[threadIdx.x];

		// prepare prevLength for next steps
		for(int i = step; i<step+gridDim.x; i++)
			prevLengths+=lengths__b[i];
	}
}







void __global__
max_pool(float* maxes__bo, int* which__bo,
    const float* X__to, const int* lengths__b, int B, int T, int O)
{
	// Each CUDA block computes maxes of a batch of concatenated sequences, using the lengths. 
    	int bid = blockIdx.x;
	if(threadIdx.x>=O)
		return;
	__shared__ float local_maxes[512];		// take advantage of faster local memory
	__shared__ short local_which[512];

	// At each step block keeps track of the total length of all previous batches (even those processed 
	// other CUDA blocks)
	int prevLengths = 0;
	for(int i = 0; i<bid; i++)
		prevLengths+=lengths__b[i];

	// Batch-items are processed by a fixed number of launched CUDA blocks
	// with a step equal to the total number gridDim.x
    	for(int step = bid; step < B; step += gridDim.x )
	{
		int lengthOfBatch = lengths__b[step];
		int batchStarts = prevLengths*O; 

		local_maxes[threadIdx.x] = X__to[batchStarts+threadIdx.x];
		local_which[threadIdx.x] = 0;
		short j=1;	// the word index in a doc

		for (int i = batchStarts+O+threadIdx.x; i < batchStarts+(lengthOfBatch*O) ; i += O)
		{
			if(X__to[i]>local_maxes[threadIdx.x])
			{
				local_maxes[threadIdx.x] =  X__to[i];
				local_which[threadIdx.x] = j;
			}
			j++; 
		}
		__syncthreads();	// Block-wise synchronization

		maxes__bo[step*O + threadIdx.x] = local_maxes[threadIdx.x];
		which__bo[step*O + threadIdx.x] = local_which[threadIdx.x];

		// prepare prevLength for next steps
		for(int i = step; i<step+gridDim.x; i++)
			prevLengths+=lengths__b[i];
	}
}


void __global__
backprop_mean_pool(float* dX__to, const float* d_means__bo, const int* lengths__b,
    int B, int T, int O)
{
	// Each CUDA block computes maxes of a batch of concatenated sequences, using the lengths. 
    	int bid = blockIdx.x;
	if(threadIdx.x>=O)
		return;

	__shared__ float local_means[512];		// can be bigger, depends on dimensions

	// At each step it keeps track of the total length of all previous batches (even those processed 
	// other CUDA blocks)
	int prevLengths = 0;
	for(int i = 0; i<bid; i++)
		prevLengths+=lengths__b[i];

    	for(int step = bid; step < B; step += gridDim.x )
	{
		int lengthOfBatch = lengths__b[step];
		int batchStarts = prevLengths*O; 
		float scale = 1.0/(float)lengthOfBatch;
		local_means[threadIdx.x] = d_means__bo[step*O+threadIdx.x]*scale;

		for (int i = batchStarts + threadIdx.x; i < batchStarts+(lengthOfBatch*O) ; i += O)
			dX__to[i] = local_means[threadIdx.x];

		// prepare prevLength for next steps
		for(int i = step; i<step+gridDim.x; i++)
			prevLengths+=lengths__b[i];		
	}
}



void __global__
backprop_max_pool(float* dX__to,
    const float* d_maxes__bo, const int* which__bo, const int* lengths__b, int B, int T, int O)
{
	// Each CUDA block computes maxes of a batch of concatenated sequences, using the lengths. 
    	int bid = blockIdx.x;
	if(threadIdx.x>=O)
		return;
	__shared__ float local_maxes[512];		// can be bigger, depends on dimensions
	__shared__ short local_which[512];
	int prevLengths = 0;
	for(int i = 0; i<bid; i++)
		prevLengths+=lengths__b[i];

    	for(int step = bid; step < B; step += gridDim.x )
	{
		int lengthOfBatch = lengths__b[step];
		int batchStarts = prevLengths*O; 

		local_maxes[threadIdx.x] = d_maxes__bo[step*O+threadIdx.x];
		local_which[threadIdx.x] = which__bo[step*O+threadIdx.x];
		short j=0;	// the word index in a doc

		for (int i = batchStarts+threadIdx.x; i < batchStarts+(lengthOfBatch*O) ; i += O)
		{
			if(local_which[threadIdx.x]==j)
			{
				dX__to[i] =  local_maxes[threadIdx.x];
			}
			else
				dX__to[i]=0;
			j++; 
		}

		// prepare prevLength for next steps
		for(int i = step; i<step+gridDim.x; i++)
			prevLengths+=lengths__b[i];
	}
}


