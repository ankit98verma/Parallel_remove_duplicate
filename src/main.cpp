/*
 * main: 
 *
 * Ankit Verma, 2020
 *
 * This file runs the merge sorting on GPU
 *
 */

/* C++ includes */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

/* Cuda includes */
#include <cuda_runtime.h>
#include "cuda_sorting.cuh"


using namespace std;


/* Local functions */
void export_gpu_outputs(bool verbose);

/* Variable definitions */
int * cpu_arr;
unsigned int arr_len;
int * gpu_out_arr;

/* Definitions for time profiling the GPU program */
cudaEvent_t start;
cudaEvent_t stop;
#define START_TIMER() {                         \
      CUDA_CALL(cudaEventCreate(&start));       \
      CUDA_CALL(cudaEventCreate(&stop));        \
      CUDA_CALL(cudaEventRecord(start));        \
    }

#define STOP_RECORD_TIMER(name) {                           \
      CUDA_CALL(cudaEventRecord(stop));                     \
      CUDA_CALL(cudaEventSynchronize(stop));                \
      CUDA_CALL(cudaEventElapsedTime(&name, start, stop));  \
      CUDA_CALL(cudaEventDestroy(start));                   \
      CUDA_CALL(cudaEventDestroy(stop));                    \
    }


/*******************************************************************************
 * Function:        chech_args
 *
 * Description:     Checks for the user inputs arguments to run the file
 *
 * Arguments:       int argc, char argv
 *
 * Return Values:   0
*******************************************************************************/
int check_args(int argc, char **argv){
	if (argc != 3){
        // printf("Usage: ./grav [depth] [thread_per_block] \n");
        printf("Usage: ./grav [length of array] [verbose: 0/1]\n");
        return 1;
    }
    return 0;
}

/*******************************************************************************
 * Function:        time_profile_gpu
 *
 * Description:     Runs the GPU code
 *
 * Arguments:       bool verbose: If true then it will prints messages on the
 *                  console
 *
 * Return Values:   GPU computational time
*******************************************************************************/
void time_profile_gpu(bool verbose){

	float gpu_time_sorting = 0;
	float gpu_time_indata_cpy = 0;
	float gpu_time_outdata_cpy = 0;

	cudaError err;

    /* Fill the input */
	START_TIMER();
		cuda_cpy_input_data(cpu_arr, arr_len);
	STOP_RECORD_TIMER(gpu_time_indata_cpy);


    /* Sort the array */
	START_TIMER();
		cudacall_merge_sort();
	STOP_RECORD_TIMER(gpu_time_sorting);
    err = cudaGetLastError();
    if (cudaSuccess != err){
        cerr << "Error " << cudaGetErrorString(err) << endl;
    }else{
    	if(verbose)
        	cerr << "No kernel error detected" << endl;
    }

    /* Copy the result to the CPU memory */
    START_TIMER();
		cuda_cpy_output_data(gpu_out_arr, arr_len);
	STOP_RECORD_TIMER(gpu_time_outdata_cpy);

	if(verbose){
		printf("GPU Input data copy time: %f ms\n", gpu_time_indata_cpy);
	    printf("GPU Sorting time: %f ms\n", gpu_time_sorting);
		printf("GPU Output data copy time: %f ms\n", gpu_time_outdata_cpy);
		printf("Total GPU time: %f ms\n",gpu_time_indata_cpy+ gpu_time_sorting + gpu_time_outdata_cpy );
	}
}


/*******************************************************************************
 * Function:        init_vars
 *
 * Description:     This function initializes global variables. This should be
 *					the first function to be called from this file.
 *
 * Arguments:       unsigned int depth: The maximum depth of the icosphere
 *					float r: The radius of sphere
 *
 * Return Values:   None.
 *
*******************************************************************************/
void init_vars(unsigned int len){
	arr_len = len;
    cpu_arr = (int *)malloc(arr_len*sizeof(float));
    gpu_out_arr = (int *)malloc(arr_len*sizeof(int));
    
    /* Randomly generate integers b/w 0 to 100 */
    srand(0);
    for(unsigned int i = 0; i<arr_len; i++){
        cpu_arr[i] = rand()%100;
    }
}

/*******************************************************************************
 * Function:        main
 *
 * Description:     Run the main function
 *
 * Arguments:       int argc, char argv
 *
 * Return Values:   int 1 if code executes successfully else 0.
*******************************************************************************/
int main(int argc, char **argv) {

	if(check_args(argc, argv))
		return 0;

	int len = atoi(argv[1]);

	bool verbose = (bool)atoi(argv[2]);
	
	if(verbose)
		cout << "Verbose ON" << endl;
	else
		cout << "Verbose OFF" << endl;

	init_vars(len);

	if(verbose)
		cout << "\n----------Running GPU Code----------\n" << endl;
	
	time_profile_gpu(verbose);

	export_gpu_outputs(verbose);
	
	free(cpu_arr);
	free(gpu_out_arr);
	free_gpu_memory();

    return 1;
}

/*******************************************************************************
 * Function:        export_gpu_outputs
 *
 * Description:     Exports the gpu output array
 *
 * Arguments:       bool verbose: If true then it will prints messages on the c
 *                  console
 *
 * Return Values:   none
*******************************************************************************/
void export_gpu_outputs(bool verbose){

    cout << "Exporting: gpu_arr.csv"<<endl;

    string filename2 = "results/gpu_arr.csv";
    ofstream obj_stream2;
    obj_stream2.open(filename2);
    obj_stream2 << "array" << endl;
    cout <<"-----------------------" << endl;
    for(unsigned int i=0; i< arr_len; i++){
        obj_stream2 << gpu_out_arr[i] << endl;
    }
    obj_stream2.close();
}