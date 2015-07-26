CUDA code from the course "Applications of Parallel Computing" converted OpenCL code.

Use command make in the root folder to compile and then run ./simulation

use "-n 1000" to get 1000 particles and "-o output" to get output for visualization tool.

Visualization tools are from the course. For windows use the .zip for Linux distriutions compile the program code in tar.gz (requires SDL).

Sample image from visualization:

![alt tag](http://puu.sh/jcWWG/3bf4b4c26f.png)


Time taken for 1000 steps (ran on virtual machine, script in speed.sh):

CPU-GPU copy time = 2.5e-05 seconds
n = 128, simulation time = 0.099288 seconds
CPU-GPU copy time = 2.8e-05 seconds
n = 256, simulation time = 0.350236 seconds
CPU-GPU copy time = 3e-05 seconds
n = 512, simulation time = 1.44735 seconds
CPU-GPU copy time = 2.8e-05 seconds
n = 1024, simulation time = 4.63172 seconds
CPU-GPU copy time = 3.5e-05 seconds
n = 2048, simulation time = 12.7454 seconds
CPU-GPU copy time = 4.9e-05 seconds
n = 4096, simulation time = 46.984 seconds