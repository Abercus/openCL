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
	// Different types of devices to pick from. At the moment picks the default opencl device.
	//CL_DEVICE_TYPE_GPU
	//CL_DEVICE_TYPE_ACCELERATOR
	//CL_DEVICE_TYPE_DEFAULT
	//CL_DEVICE_TYPE_CPU
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
	//printf("%i\n", ret);
	ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);
	
	// Create kernels
	cl_kernel forceKernel = clCreateKernel(program, "compute_forces_gpu", &ret);

	cl_kernel moveKernel = clCreateKernel(program, "move_gpu", &ret);

	cl_kernel binInitKernel = clCreateKernel(program, "bin_init_gpu", &ret);

	cl_kernel binKernel = clCreateKernel(program, "bin_gpu", &ret);

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
	

	// Calculating thread and thread block counts.
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

	// Local item size
	localItemSize = globalItemSize / NUM_THREADS;	

	// Bins and bin sizes.
	// Because of uniform distribution we will know that bins size is amortized. Therefore I picked the value of 10.
	// There will never be 10 particles in one bin.
	int maxParticles = 10;
	
	// Calculating the number of bins.
	int numberOfBins = (int)ceil(size/(2*cutoff)) + 2;
	
	// Bins will only exist on the device.
	particle_t* bins;
	
	// How many particles are there in each bin - also only exists on the device.
	volatile long* binSizes;
	
	// Number of bins to be initialized.
	size_t clearAmt = numberOfBins*numberOfBins;
	
	// Allocate memory for bins on the device.
	cl_mem d_binSizes = clCreateBuffer(context, CL_MEM_READ_WRITE, numberOfBins * numberOfBins * sizeof(volatile long), NULL, &ret);
	cl_mem d_bins = clCreateBuffer(context, CL_MEM_READ_WRITE, numberOfBins * numberOfBins * maxParticles * sizeof(particle_t), NULL, &ret);
	
	// SETTING ARGUMENTS FOR THE KERNELS
	
	// Set arguments for the init / clear kernel
	ret = clSetKernelArg(binInitKernel, 0, sizeof(cl_mem), (void *)&d_binSizes);
	ret = clSetKernelArg(binInitKernel, 1, sizeof(int), &numberOfBins);

	// Set arguments for the binning kernel
	ret = clSetKernelArg(binKernel, 0, sizeof(cl_mem), (void *)&d_particles);
	ret = clSetKernelArg(binKernel, 1, sizeof(int), &n);
	ret = clSetKernelArg(binKernel, 2, sizeof(cl_mem), (void *)&d_bins);
	ret = clSetKernelArg(binKernel, 3, sizeof(cl_mem), (void *)&d_binSizes);
	ret = clSetKernelArg(binKernel, 4, sizeof(int), &numberOfBins);

	// Set arguments for force kernel.
	ret = clSetKernelArg(forceKernel, 0, sizeof(cl_mem), (void *)&d_particles);
	ret = clSetKernelArg(forceKernel, 1, sizeof(int), &n);
	ret = clSetKernelArg(forceKernel, 2, sizeof(cl_mem), (void *)&d_bins);
	ret = clSetKernelArg(forceKernel, 3, sizeof(cl_mem), (void *)&d_binSizes);
	ret = clSetKernelArg(forceKernel, 4, sizeof(int), &numberOfBins);


	// Set arguments for move kernel
	ret = clSetKernelArg(moveKernel, 0, sizeof(cl_mem), (void *)&d_particles);
	ret = clSetKernelArg(moveKernel, 1, sizeof(int), &n);
	ret = clSetKernelArg(moveKernel, 2, sizeof(double), &size);
	
	
	// Variable to check if kernel execution is done.
	cl_event kernelDone;
	
	
	double simulation_time = read_timer();

	for (int step = 0; step < NSTEPS; step++) {


		// Execute bin initialization (clearing after first iteration)
		ret = clEnqueueNDRangeKernel(commandQueue, binInitKernel, 1, NULL, &clearAmt, NULL, 0, NULL, &kernelDone);
		ret = clWaitForEvents(1, &kernelDone);
		// Execute binning kernel
		ret = clEnqueueNDRangeKernel(commandQueue, binKernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, &kernelDone);
		ret = clWaitForEvents(1, &kernelDone);	
		// Execute force kernel
		ret = clEnqueueNDRangeKernel(commandQueue, forceKernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, &kernelDone);
		ret = clWaitForEvents(1, &kernelDone);
		// Execute move kernel
		ret = clEnqueueNDRangeKernel(commandQueue, moveKernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, &kernelDone);
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
