/*
 * cuda_sorting.cu
 *
 * By Ankit Verma, 2020
 *
 */
#ifndef _GRAV_CUDA_ICOSPHERE_CU_
	#define _GRAV_CUDA_ICOSPHERE_CU_
#endif

#include "cuda_sorting.cuh"


/* Local variables */
int * pointers_arrs[2];	// the pointer to contain the address of the array and its copy
int ind2_arr;			// stores the index of the pointers_arrs which points to the most updated array
int * dev_arr;			// it contains the array to be sorted in the device memory (GPU memory)
int * dev_arr_cpy;		// a copy for the array to be sorted in the device memory (GPU memory)

int arr_length;			// the length of the array

/* Local functions */
__device__ void get_first_greatest(int * arr, int len, int a, int * res_fg);
__device__ void get_last_smallest(int * arr, int len, int a, int * res_ls);

/*******************************************************************************
 * Function:        cuda_cpy_input_data
 *
 * Description:     This function allocate the memory for various array in the GPU
 *					memory. It copies the input array to the array in the GPU.
 *
 * Arguments:       int * in_arr: The array to be sorted
 *					unsigned int length: The length of the array to be sorted
 *
 * Return Values:   None
*******************************************************************************/
void cuda_cpy_input_data(int * in_arr, unsigned int length){
	arr_length = length;

	CUDA_CALL(cudaMalloc((void**) &dev_arr, arr_length * sizeof(int)));
	CUDA_CALL(cudaMalloc((void**) &dev_arr_cpy, arr_length* sizeof(int)));

	// set the pointer
	pointers_arrs[0] = dev_arr;	
	pointers_arrs[1] = dev_arr_cpy;
	ind2_arr = 0;						// set the index denoting the latest array to 0

	// copy input to the GPU memory
	CUDA_CALL(cudaMemcpy(dev_arr, in_arr, arr_length*sizeof(int), cudaMemcpyHostToDevice));

}

/*******************************************************************************
 * Function:        cuda_cpy_output_data
 *
 * Description:     This function copies the latest array to the CPU memory.
 *
 * Arguments:       int * out_arr: The array to which the GPU result is to be 
 *										copied.
 *					unsigned int length: The length of the array.
 *
 * Return Values:   None
*******************************************************************************/
void cuda_cpy_output_data(int * out_arr, unsigned int length){
	CUDA_CALL(cudaMemcpy(out_arr, pointers_arrs[ind2_arr], length*sizeof(int), cudaMemcpyDeviceToHost));
	
}

/*******************************************************************************
 * Function:        free_gpu_memory
 *
 * Description:     This function frees the GPU memory.
 *
 * Arguments:       None
 *
 * Return Values:   None
*******************************************************************************/
void free_gpu_memory(){
	
	CUDA_CALL(cudaFree(dev_arr));
	CUDA_CALL(cudaFree(dev_arr_cpy));
}


/*******************************************************************************
 * Function:        dev_merge
 *
 * Description:     This function SEQUENTIALLY merges two already sorted arrays into
 * 					a single sorted array. The two arrays here are:
 *
 * 					s[idx] and s[start]. Let us say arr1 denotes s[idx] array and
 * 					arr2 denotes s[start] array.
 *
 * 					The size of arr1 = start - idx
 * 					The size of arr2 = end - start
 *
 * 					Following is the reference to the merging of sorted arrays into
 * 					one sorted array:
 * 					https://www.geeksforgeeks.org/merge-two-sorted-arrays/
 *
 * 					The arr1 and arr2 are sorted and the result is put into the 
 * 					array "r".
 *
 * Arguments:       int * s: The array to be sorted
 * 					int * r: The array to which the sorted sums array will be stored
 *					unsigned int idx: The start of first array
 *					unsigned int start: The start of second array
 *					unsinged int end: The end of second array
 *
 * Return Values:   None
*******************************************************************************/
__device__
void dev_merge(int * s, int * r, unsigned int idx, unsigned int start, unsigned int end){

	// refer to https://www.geeksforgeeks.org/merge-two-sorted-arrays/ for more detail
	unsigned int c=idx;
	unsigned int i=idx;unsigned int j=start;
	while(j<end && i<start){
		if(s[i] <= s[j]){
			r[c] = s[i];
			i++;
		}
		else{
			r[c] = s[j];
			j++;
		}
		c++;
	}
	while(i < start){
		r[c] = s[i];
		c++;i++;
	}

	while(j < end){
		r[c] = s[j];
		c++;j++;
	}
}

/*******************************************************************************
 * Function:        kernel_merge_sort
 *
 * Description:     This is a naive kernal which implements the a step of merge sorting.
 * 					The input "r" represents the total length of the two arrays. For 
 * 					example consider the following example:
 *
 * 					arr[] = {4, 3, 2, 1, 0};
 *
 * 					We have ceil(log2(length(arr))) = 3, hence the kernel has to 
 * 					be called 3 times, for ith time (starting i=0) the value of
 * 					"r" should be 2^(i+1).
 *
 * 					So when kernel is called:
 * 					For r = 2 (iteration 0);
 * 						Thread 0 works on arrays partitions [4] [3]
 * 						Thread 1 works on arrays partitions [2] [1]
 *
 * 						Result stored in "res" array:
 * 						[3, 4, 1, 2, 0];
 *
 * 					For r = 4 (iteration 1)
 * 						Thread 0 works on arrays partitions [3, 4] [1, 2]
 *
 *						Result stored in "res" array:
 * 						[1, 2, 3, 4, 0];
 *
 * 					for r = 8 (iteration 2)
 * 						Thread 0 works on arrays partitions [1, 2, 3, 4] [0]
 *
 *						Result stored in "res" array:
 * 						[0, 1, 2, 3, 4];	
 *
 * 					This kernel perform one step of the merge sort on chuncks of 1024 elements
 * 					by copying the elements to the shared memory and then copying the results
 * 					back to the global memory
 *
 * Arguments:       int * arr: The array to be sorted
 * 					int * res: The array to which the sorted sums array will be stored
 *					const unsigned int length: The total length of the array
 *					const unsigned int r: Equals to 2 times of the length of sub-arrays which have to be sorted.
 *
 * Return Values:   None
*******************************************************************************/
__global__
void kernel_merge_sort(int * arr, int * res, const unsigned int length, const unsigned int r){

	__shared__ int sh_sums[1024];
	__shared__ int sh_res[1024];
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numthrds = blockDim.x * gridDim.x;

	const int stride = r/2;

	int id = threadIdx.x;
	int t_len = min(1024, length - blockIdx.x * blockDim.x);

	while(idx < length){
		// copy to shared mem
		sh_sums[threadIdx.x] = arr[idx];

		__syncthreads();

		// perform a step of merge sort
		if(id%r == 0)
			dev_merge(sh_sums, sh_res, id, min(t_len, id + stride), min(t_len, id+r));

		__syncthreads();

		// copy result to global mem
		res[idx] = sh_res[threadIdx.x];
		
		__syncthreads();
		
		idx += numthrds;
	}
}

/*******************************************************************************
 * Function:        kernel_merge_chuncks
 *
 * Description:     This is a kernal which implements PARALLELED merging of sorted
 * 					arrays. 
 * 					The Algorithm 1 of following reference describes the 
 * 					parallel merging of sorted arrays:
 *
 * 					: http://www2.hawaii.edu/~nodari/teaching/f16/notes/notes10.pdf
 *
 * 					The above reference assumes that the arrays don't contain the
 * 					duplicate elements.
 *
 * 					A new algorithm which is a modified version of the above 
 * 					algorithm is designed to overcome this constrain. Following is the
 *					description of the algorithm.
 *
 * 					Say we have two sorted arrays, arr1 and arr2 to be merged and
 * 					have duplicate elements. The result of merging arr1 and arr2 has 
 * 					to be put in arr_res array.
 *
 * 		Task	1.	For an element "i" in arr1 i.e. arr1[i], find the index of largest
 * 					number in arr2 which is smaller than arr1[i] i.e.
 *
 * 						LS_index = argmax_j (arr2[j] < arr1[i]).
 *
 * 					LS_index is the index of the largest number in arr2 such that 
 * 					arr2[LS_index] < arr1[i]. Now place the arr1[i] at position
 * 					i + LS_index + 1 in the array arr_res i.e.
 *
 * 						arr_res[i+LS_index+1] = arr1[i].
 *
 * 					Do the task 1 for every element of arr1.
 *
 * 		Task	2.	Now for an elemetn "i" in arr2 i.e. arr2[i], find the index of smallest
 * 					number in arr1 which is larger than arr2[o] i.e.
 *
 * 						SL_index = argmin_j (arr2[i] < arr1[j])
 *
 * 					SL_index is the index of smallest number in arr1 such that
 * 					arr2[i] < arr1[SL_index]. Now place the arr2[i] at position i + SL_index
 * 					in the array arr_res i.e.
 *
 * 						arr_res[i+SL_index] = arr2[2].
 *
 * 					Do the task 2 for every element of arr2.
 *
 * 					The above algorithm has been parallelized with each threads operating
 * 					at each element of the array. Let us say we have two arrays, each of size
 *					1024 elements, then total of 2048 threads are used to merge the two arrays.
 *
 * Arguments:       int * arr: The array to be sorted
 * 					int * res: The array to which the sorted arr array will be stored
 *					const unsigned int length: The total length of the array
 *					const unsigned int r: Equals to 2 times of length of sub-arrays which have to be sorted.
 *
 * Return Values:   None
*******************************************************************************/
__global__
void kernel_merge_chuncks(int * arr, int * res, const unsigned int length, const unsigned int r){
	
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numthrds = blockDim.x * gridDim.x;
	const int stride = r/2;
	
	int tmp_res[1];
	
	int k;
	int local_k;
	int arr_len;
	int arr_start;
	int arr_ind_L, arr_ind_HE, final_index;
	
	while(idx < length){
		k = idx % r;
		local_k = k%stride;
		arr_ind_L = idx - local_k + stride;    // finding the starting of arr2 
		arr_ind_HE = idx - local_k - stride;   // finding the starting of arr1 

		if(k < stride && arr_ind_L < length){
			// This thread works on element of arr 1
			arr_len = min(stride, length - arr_ind_L);
			arr_start = idx - local_k + 1;

			// get the index of largest number which is smaller than arr[idx]
			get_last_smallest(&arr[arr_ind_L], arr_len, arr[idx], tmp_res);

			// calculate the index in the resulting array where arr[idx] will
			// be placed
			final_index = local_k + tmp_res[0] + arr_start;

		}else if( k>=stride && 0 <= arr_ind_HE){
			// This thread works on element of arr 2
			arr_len = min(stride, length - arr_ind_HE);
			arr_start = idx - local_k - stride;

			// get the index of smallest number which is greater than arr[idx]
			get_first_greatest(&arr[arr_ind_HE], arr_len, arr[idx], tmp_res);
			
			// calculate the index in the resulting array where arr[idx] will
			// be placed
			final_index = local_k + tmp_res[0] + arr_start;
		}
		
		// now place the element
		res[final_index] = arr[idx];
		
		idx += numthrds;
	}

}

/*******************************************************************************
 * Function:        cudacall_fill_vertices
 *
 * Description:     This calls the optimized sorting algorithms.
 *
 * 					The order the sequential implementation of this algorithm is: O(Log(n)^2)
 * 					The order the parallel implementation of this algorithm is: O(Log(n)^2/m)
 * 					where m: no. of parallel running processors
 *
 * Arguments:       None
 *
 * Return Values:   None
*******************************************************************************/
void cudacall_merge_sort() {
	int thread_num = GPU_THREAD_NUM;
	unsigned int len = arr_length;
	int n_blocks = min(65535, (len + thread_num  - 1) / thread_num);


	// first sort using the sequential merging and shared memory
	unsigned int l = ceil(log2(thread_num)), ind1;
	for(int i=0; i<l; i++){
		ind1 = i%2;
		ind2_arr = (i+1)%2;
		unsigned int r = pow(2, i+1);
		kernel_merge_sort<<<n_blocks, thread_num>>>(pointers_arrs[ind1], pointers_arrs[ind2_arr], len, r);

	}

	// now sort the chunks of size 1024 
	l = ceil(log2(n_blocks));
	for(int i=0; i<l; i++){
		ind1 = (ind1+1)%2;
		ind2_arr = (ind2_arr+1)%2;
		unsigned int r = pow(2, i+1)*1024;
		kernel_merge_chuncks<<<n_blocks, thread_num>>>(pointers_arrs[ind1], pointers_arrs[ind2_arr], len, r);
	}
}


/*******************************************************************************
 * Function:        get_first_greatest
 *
 * Description:     This kernel finds the index of the smallest value in the array
 * 					which is greater than the value "a" passed to it i.e. it finds
 * 					the index first greatest number.
 *
 * 					Mathematically:
 *
 *  					ref_fg = argmin_j (a < arr2[j]).
 *
 * Arguments:       int * arr: The array in which the first greatest is to be found
 * 					int len: The length of the array
 * 					int a: The value with respect to which the first greatest has 
 * 								to be found
 * 					int * res_fg: It is used to return the result
 *
 * Return Values:   None
*******************************************************************************/
__device__
void get_first_greatest(int * arr, int len, int a, int * res_fg){
	int first = 0, last = len - 1;
	while (first <= last)
	{
		int mid = (first + last) / 2;
		if (arr[mid] > a)
			last = mid - 1;
		else
			first = mid + 1;
	}
	res_fg[0] =  last + 1 == len ? len : last + 1;

}

/*******************************************************************************
 * Function:        get_last_smallest
 *
 * Description:     This kernel finds the index of the largest value in the array
 * 					which is smaller than the value "a" passed to it i.e. it finds
 * 					the index first greatest number.
 *
 * 					Mathematically:
 *
 *  					ref_ls = argmax_j (arr2[j] < a).
 *
 * Arguments:       int * arr: The array in which the last smallest is to be found
 * 					int len: The length of the array
 * 					int a: The value with respect to which the last smallest has 
 * 								to be found
 * 					int * res_fg: It is used to return the result
 *
 * Return Values:   None
*******************************************************************************/
__device__
void get_last_smallest(int * arr, int len, int a, int * res_ls){
	int first = 0, last = len - 1;
	while (first <= last)
	{
		int mid = (first + last) / 2;
		if (arr[mid] >= a)
			last = mid - 1;
		else
			first = mid + 1;
	}
	res_ls[0] = first - 1 < 0 ? -1 : first - 1;
}