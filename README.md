# Parallel duplicate removal sort (todo)

This repository implements the parallel removal of duplicate elements from an array on CUDA. As an example, if the input to the program is: 
```C++
in_arr = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 2, 3, 3];
```

The output of the program would be:
```C++
[1, 2, 3];
```

The removing of duplicates takes place into four steps:

1. Sort the array
2. Mark the duplicate and unique elements
3. Find the amount by which the unique elements have to be shifted.
4. Shift the elements.

## Implementation

### Sort the array
The input array is sorted using [Parallel_Merge_Sort ](https://github.com/ankit98verma/Parallel_Merge_Sort). For more details look at the GitHub repository.

### Marking duplicate elements
The sorted array is used to create a new array which will contain marking for every unique and duplicate elements. The duplicate elements are marked with -1 while the unique elements are represented by the value of their index. Using the above example, the input to this step is following array
```C++
[1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
```
and the output of this step is
```C++
[-1, -1, -1, 3, -1, -1, -1, -1, 8, -1, -1, -1, -1, 13]
```
The operation of this step becomes clear by looking at the input and the output array together:
```C++
input_array	= 	[	1, 	1,	1,	1,	2,	2,	2,	2,	2,	3,	3,	3,	3,	3]
marking_array = [	-1,	-1,	-1,	3,	-1,	-1,	-1,	-1,	8,	-1,	-1,	-1,	-1,	13]
```

In this step, each thread compares the value of elements present in array at index corresponding to the thread ID with next element. If the value are equal the thread put value "-1" in the output array otherwise puts it's thread index in the output array.

### Find the required shift
We know the position of unique elements from previous step. We need to shift the elements to get the required output array. First, we find the amount of shift required for each unique element such that it becomes adjacent to its immediate previous unique element assuming that the previous unique element has not been shifted.

So the input and output of this step are as follows:
```C++
input_array	= 	[	-1,	-1,	-1,	3,	-1,	-1,	-1,	-1,	8,	-1,	-1,	-1,	-1,	13]
output_array = 	[	0,	0,	0,	3,	0,	0,	0,	0,	4,	0,	0,	0,	0,	4]
```

This is done parallel as follows:

1. (In parallel) For every element in input_array perform the following:
	
	a. COUNTER = 0;

	b. If the value of input_array at thread index is -1 then go to step 'c' else goto step 'd'.

	c. Set COUNTER to 0 and goto step 'f'.

	d. While previous element is -1 do step 'e'.

	e. increase COUNTER by 1.

	f. Put COUNTER in the output array at thread index and EXIT.

Now we need to find the total amount of shift required for a unique element so that we get the desired output. For this [prefix algorithm: shorter span, more parallel](https://en.wikipedia.org/wiki/Prefix_sum#Algorithm_1:_Shorter_span,_more_parallel) is used. The resulting array is as follows:
```C++
input_array	= 	[	0,	0,	0,	3,	0,	0,	0,	0,	4,	0,	0,	0,	0,	4]
shift_array = 	[	0,	0,	0,	3,	3,	3,	3,	3,	7,	7,	7,	7,	7,	11]
```

### Shifting the elements
The "marking_array" and "shift_array" is used to shift the unique elements to shift the elements from input array to the output array. The length of output array is obtained by subtracting the value of last element of the "shift_array" from the length of input array. In this case it is 14-11 = 3.

Each thread looks at the marking array element, if the element is -1 then it doesn't do anything otherwise it subtracts the corresponding shift_array value from its thread index and put the input array value at the resulting position.

For the given example, thread 3 will observe that the marking is NOT "-1", so it will subtract the corresponding shift_array value from its index i.e. 3-3 = 0, now thread 3 will put the input_array[3] on output_array[0]. Similarly thread 8 will observer that the marking is NOT "-1", so it puts input_array[8] at position output_array[8 - shift_array[8]] => output_array[1] = input_array[8].

## Time complexity
Say, we have an array of size *N* to sort. There are total of *O(log<sub>2</sub>(N))* steps to sort the array using the merge sort. In each step we use binary search to find the largest smallest element and the smallest largest element for which the complexity is *O(log<sub>2</sub>(N))*. Hence the time complexity of the algorithm is *O(log<sup>2</sup>(N))* (the base of the log is 2). The complexity of marking the duplicates, finding the required and shifting the elements are *O(n/m)*, *O(n\*d/m)*, *O(n/m)* respectively, where *d* is maximum number of duplicates present in the array. So the overall complexity of the algorithm is: *O(n\*d/m)*

## Building and running the example code
The *main.cpp* implements a example usage of the parallel removing of duplicate elements. The code generates an array filled with random numbers between 0 and 15 uniformly, then remove duplicate elements from it and export the result as ".csv" file in the "results" folder. The Matlab code provided in the "utility" folder can be used to plot the exported ".csv" file to verify the results.

Following steps are to be followed to build and run the program
1. Make sure we are in the same correct folder
    ```sh
    $ ls
    bin  Makefile  README.md  results  src  utilities
    ```
2. Make the file. Note that it is assumed that the nvidia cuda toolkit is installed (ignore warnings, if any).
    ```sh
    $ make clean all
    rm -f rm_dup *.o bin/*.o *~
    rm -f src/*~
    g++ -g -Wall -D_REENTRANT -std=c++0x -pthread -c -o bin/gpu-main.cpp.o -I/include src/main.cpp  
    /bin/nvcc -m64 -g -dc -Wno-deprecated-gpu-targets --std=c++11 --expt-relaxed-constexpr -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61 -c -o bin/cuda_remove_duplicates.cu.o  src/cuda_remove_duplicates.cu
	/bin/nvcc -dlink -Wno-deprecated-gpu-targets -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61 -o bin/cuda.o  bin/cuda_remove_duplicates.cu.o
	g++ -g -Wall -D_REENTRANT -std=c++0x -pthread -o rm_dup -I/include bin/gpu-main.cpp.o bin/cuda.o bin/cuda_remove_duplicates.cu.o -L/lib64 -lcudart -lcurand 
    ```
3. A new file "rm_dup" is created, check using "ls"
    ```sh
    $ ls
    bin  Makefile  README.md  results  rm_dup  src  utilities
    ```
4. Run the program using ./short < length of array > < verbose: 0/1 >
    ```sh
    $ ./rm_dup 1000 1
   Verbose ON

	----------Running GPU Code----------

	No kernel error detected
	Resulting length: 15
	GPU Input data copy time: 0.131072 ms
	GPU remove duplicates time: 0.269312 ms
	GPU Output data copy time: 0.012096 ms
	Total GPU time: 0.412480 ms
	Exporting: gpu_arr.csv
	-----------------------
    ```

## Using the parallel remove duplicate
The *cuda_remove_duplicates.cu* and *cuda_remove_duplicates.cuh* files are required to use the parallel remove duplicate program. Following code gives an example on how to use it:

```C++
#include <cstdio>
#include <cstdlib>
#include <iostream>

/* Cuda includes */
#include <cuda_runtime.h>
#include "cuda_remove_duplicates.cuh"

int main(int argc, char **argv) {

    /* allocate the memory for the array to sorted and the resulting array */
    int arr_len = 100;
    cpu_arr = (int *)malloc(arr_len*sizeof(float));
    
    /* Initialize the array (here it is randomly initialized) */
    srand(0);
    for(unsigned int i = 0; i<arr_len; i++){
        cpu_arr[i] = rand()%10;
    }
    
    /* Initialize the GPU memory and copy the array to it */
    cuda_cpy_input_data(cpu_arr, arr_len);
    
    /* sort the array in GPU */
    out_arr_len = cudacall_remove_duplicates();
    /* check for errors */
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err){
        cerr << "Error " << cudaGetErrorString(err) << endl;
        return -1;
    }else{
    	cerr << "No kernel error detected" << endl;
    }

    gpu_out_arr = (int *)malloc(out_arr_len*sizeof(int));
    /* Copy the result back to the CPU memory*/
    cuda_cpy_output_data(gpu_out_arr);
    
    /* Free the CPU memory */
    free(cpu_arr);
	free(gpu_out_arr);
	
	/* Free the GPU memory */
	free_gpu_memory();
	
    return 1;
}
```