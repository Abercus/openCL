#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"

// Headers for openCL
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


#define MAX_SOURCE_SIZE (0x100000)


#define NUM_THREADS 256

extern double size;


/* 
IMPLEMENT BINNING (REBINNING ON GPU). MIGHT NEED TO USE ATOMIC ADD

*/

int main(int argc, char **argv) {

	if (find_option(argc, argv, "-h") >= 0)
	{
		printf("Options:\n");
		printf("-h to see this help\n");
		printf("-n <int> to set the number of particles\n");
		printf("-o <filename> to specify the output file name\n");
		printf("-s <filename> to specify the summary output file name\n");
		return 0;
	}


	int n = read_int(argc, argv, "-n", 1000);

	char *savename = read_string(argc, argv, "-o", NULL);
	char *sumname = read_string(argc, argv, "-s", NULL);

	// For return values.
	cl_int ret;

	// OpenCL stuff.
	// Loading kernel files.
	FILE *kernelFile;
	char *kernelSource;
	size_t kernelSize;

	kernelFile = fopen("simulationKernel.cl", "r");

	if (!kernelFile) {
		fprintf(stderr, "No file named simulationKernel.cl was found\n");
		exit(-1);
	}
	kernelSource = (char*)malloc(MAX_SOURCE_SIZE);
	kernelSize = fread(kernelSource, 1, MAX_SOURCE_SIZE, kernelFile);
	fclose(kernelFile);


	// Getting platform and device information
	cl_platform_id platformId = NULL;
	cl_device_id deviceID = NULL;
	cl_uint retNumDevices;
	cl_uint retNumPlatforms;
	ret = clGetPlatformIDs(1, &platformId, &retNumPlatforms);
	ret = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, &retNumDevices);


	// Max workgroup size
	size_t max_available_local_wg_size;
	ret = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_available_local_wg_size, NULL);

	// Creating context.
	cl_context context = clCreateContext(NULL, 1, &deviceID, NULL, NULL, &ret);


	// Creating command queue
	cl_command_queue commandQueue = clCreateCommandQueue(context, deviceID, 0, &ret);

	// Build program
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, (const size_t *)&kernelSize, &ret);

	ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);
	
	// Create kernels
	cl_kernel forceKernel = clCreateKernel(program, "compute_forces_gpu", &ret);
	cl_kernel moveKernel = clCreateKernel(program, "move_gpu", &ret);

	FILE *fsave = savename ? fopen(savename, "w") : NULL;
	FILE *fsum = sumname ? fopen(sumname, "a") : NULL;
	particle_t *particles = (particle_t*)malloc(n * sizeof(particle_t));

	// GPU particle data structure

	cl_mem d_particles = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(particle_t), NULL, &ret);

	// Set size
	set_size(n);

	init_particles(n, particles);

	double copy_time = read_timer();

	// Copy particles to device.
	ret = clEnqueueWriteBuffer(commandQueue, d_particles, CL_TRUE, 0, n * sizeof(particle_t), particles, 0, NULL, NULL);
	copy_time = read_timer() - copy_time;
	cl_event kernelDone;


	// sizes
	size_t globalItemSize;
	size_t localItemSize;
	// Global item size
	if (n <= NUM_THREADS) {
		globalItemSize = NUM_THREADS;
		localItemSize = 16;
	}
	else if (n % NUM_THREADS != 0) {
		globalItemSize = (n / NUM_THREADS + 1) * NUM_THREADS;
	}
	else {
		globalItemSize = n;
	}

	//printf("globalItemSize %i\n", globalItemSize);

	// Local item size
	localItemSize = globalItemSize / NUM_THREADS;

	

		// Set arguments for force kernel.
		ret = clSetKernelArg(forceKernel, 0, sizeof(cl_mem), (void *)&d_particles);
		ret = clSetKernelArg(forceKernel, 1, sizeof(int), &n);	

		// Set arguments for move kernel
		ret = clSetKernelArg(moveKernel, 0, sizeof(cl_mem), (void *)&d_particles);
		ret = clSetKernelArg(moveKernel, 1, sizeof(int), &n);
		ret = clSetKernelArg(moveKernel, 2, sizeof(double), &size);
	
	
	
	double simulation_time = read_timer();
	for (int step = 0; step < NSTEPS; step++) {

		// Execute force kernel
		ret = clEnqueueNDRangeKernel(commandQueue, forceKernel, 1, &localItemSize, &globalItemSize, NULL, 0, NULL, &kernelDone);
		ret = clWaitForEvents(1, &kernelDone);

		// Execute move kernel
		ret = clEnqueueNDRangeKernel(commandQueue, moveKernel, 1, &localItemSize, &globalItemSize, NULL, 0, NULL, &kernelDone);
		ret = clWaitForEvents(1, &kernelDone);


		if (fsave && (step%SAVEFREQ) == 0) {
			// Copy the particles back to the CPU
			ret = clEnqueueReadBuffer(commandQueue, d_particles, CL_TRUE, 0, n * sizeof(particle_t), particles, 0, NULL, &kernelDone);
			ret = clWaitForEvents(1, &kernelDone);

			save(fsave, n, particles);
		}


	}
	simulation_time = read_timer() - simulation_time;
	printf("CPU-GPU copy time = %g seconds\n", copy_time);
	printf("n = %d, simulation time = %g seconds\n", n, simulation_time);

	if (fsum)
		fprintf(fsum, "%d %lf \n", n, simulation_time);

	if (fsum)
		fclose(fsum);
	free(particles);
	if (fsave)
		fclose(fsave);


	ret = clFlush(commandQueue);
	ret = clFinish(commandQueue);
	ret = clReleaseCommandQueue(commandQueue);
	ret = clReleaseKernel(forceKernel);
	ret = clReleaseKernel(moveKernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(d_particles);
	ret = clReleaseContext(context);


	return 0;
}
