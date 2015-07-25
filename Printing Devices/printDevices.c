

// Headers for openCL
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif



#include <stdio.h>
#include <stdlib.h>



int main() {
	// Getting platform count
	cl_uint platCount;
	clGetPlatformIDs(0, NULL, &platCount);
	
	// Allocate memory, get list of platforms
	cl_platform_id *platforms = (cl_platform_id*) malloc(platCount*sizeof(cl_platform_id));
	clGetPlatformIDs(platCount, platforms, NULL);
	
	// Iterate over platforms
	cl_uint i;
	for (i=0; i<platCount; ++i) {
		char buf[256];
		clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(buf), buf, NULL);
	    printf("platform %d: vendor '%s'\n", i, buf);

		cl_uint devCount;
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &devCount);
		
		cl_device_id *devices = (cl_device_id*) malloc(devCount*sizeof(cl_device_id));
		
		// List of devices in platform
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, devCount, devices, NULL);
		cl_uint j;
		for (j=0; j<devCount; ++j) {
			char buf[256];
			clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(buf), buf, NULL);
			printf("device %d: '%s'\n", j, buf);
		}
		free(devices);
	}
	free(platforms);
	return 0;
}