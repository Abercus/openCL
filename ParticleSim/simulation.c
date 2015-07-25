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
	
	if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
		printf( "-s <filename> to specify the summary output file name\n" );
        return 0;
    }
	
	
	int n = read_int( argc, argv, "-n", 1000 );

	//n = 100;
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );




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

	
	// Creating context.
	cl_context context = clCreateContext(NULL, 1, &deviceID, NULL, NULL,  &ret);


	// Creating command queue
	cl_command_queue commandQueue = clCreateCommandQueue(context, deviceID, 0, &ret);

	
	
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, (const size_t *)&kernelSize, &ret);	


	ret = clBuildProgram(program, 1, &deviceID, "-I /home/ubuntu/Desktop/sim", NULL, NULL);


	
//	ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);
			printf("%i \n", ret);
    cl_kernel forceKernel = clCreateKernel(program, "compute_forces_gpu", &ret);
	cl_kernel moveKernel = clCreateKernel(program, "move_gpu", &ret);
	
	
	
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen(sumname,"a") : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    // GPU particle data structure
	
	cl_mem d_particles = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(particle_t), NULL, &ret);
	
	
	
	// Create force kernel, move kernel, etc..
	
	
	// Set size
    set_size( n );

    init_particles( n, particles );
	
	double copy_time = read_timer();
	
	// Copy particles to device.
	ret = clEnqueueWriteBuffer(commandQueue, d_particles, CL_TRUE, 0, n * sizeof(particle_t), particles, 0, NULL, NULL);
    
	copy_time = read_timer( ) - copy_time;
	
	
	double simulation_time = read_timer( );
    for( int step = 0; step < NSTEPS; step++ ) {
		
		 //  compute forces
		size_t localItemSize = (n + NUM_THREADS - 1) / NUM_THREADS;
		size_t globalItemSize = n;
		
		// Set arguments for force kernel.
		ret = clSetKernelArg(forceKernel, 0, sizeof(cl_mem), (void *)&d_particles);	
		ret = clSetKernelArg(forceKernel, 1, sizeof(int), &n);	

		// Execute force kernel
		ret = clEnqueueNDRangeKernel(commandQueue, forceKernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL);

		//compute_forces_gpu <<< blks, NUM_THREADS >>> (d_particles, n);
		
		// Set arguments for move kernel
		ret = clSetKernelArg(moveKernel, 0, sizeof(cl_mem), (void *)&d_particles);	
		ret = clSetKernelArg(moveKernel, 1, sizeof(int), &n);
		ret = clSetKernelArg(moveKernel, 2, sizeof(double), &size);
		// Execute move kernel
		ret = clEnqueueNDRangeKernel(commandQueue, moveKernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL);

		
		//move_gpu <<< blks, NUM_THREADS >>> (d_particles, n, size);
		/*
        if( fsave && (step%SAVEFREQ) == 0 ) {
	    // Copy the particles back to the CPU
            cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
            save( fsave, n, particles);
	}
	*/
		
	}
    simulation_time = read_timer( ) - simulation_time;
    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );

    if (fsum)
	fprintf(fsum,"%d %lf \n",n,simulation_time);

    if (fsum)
	fclose( fsum );    
    free( particles );
    if( fsave )
        fclose( fsave );
    
	
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

