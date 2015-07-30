#!/bin/bash

#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -p phi
#SBATCH --exclusive

./simulation -n 128
./simulation -n 256
./simulation -n 512
./simulation -n 1024
./simulation -n 2048
./simulation -n 4096
./simulation -n 8192
./simulation -n 16384
./simulation -n 32768
./simulation -n 65536
./simulation -n 131072
./simulation -n 262144

