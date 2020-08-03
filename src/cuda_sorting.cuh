/*
 * cuda_sorting.cuh
 *
 * By Ankit Verma, 2020
 *
 */

#ifndef _GRAV_CUDA_CUH_
#define _GRAV_CUDA_CUH_


#include "device_launch_parameters.h"
#include "cuda_calls_helper.h"

#define GPU_THREAD_NUM		1024

void cuda_cpy_input_data(int * in_arr, unsigned int length);

void cudacall_merge_sort();

void cuda_cpy_output_data(int * out_arr, unsigned int length);

void free_gpu_memory();

#endif
