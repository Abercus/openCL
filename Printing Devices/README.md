This code is supposed to print the available devices on the system.

To compile:
gcc printDevices.c -o printDevices -lOpenCL

Example output:

ubuntu@ubuntu-VirtualBox:~/Desktop$ ./printDevices platform 0: vendor 'Advanced Micro Devices, Inc.'
device 0: 'Intel(R) Core(TM) i5-3437U CPU @ 1.90GHz'
platform 1: vendor 'Intel(R) Corporation'
device 0: '       Intel(R) Core(TM) i5-3437U CPU @ 1.90GHz'
