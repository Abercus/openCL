#!/bin/sh

./simulation -n 128 >> out.txt 2>&1
./simulation -n 256 >> out.txt 2>&1
./simulation -n 512 >> out.txt 2>&1
./simulation -n 1024 >> out.txt 2>&1
./simulation -n 2048 >> out.txt 2>&1
./simulation -n 4096 >> out.txt 2>&1
./simulation -n 8192 >> out.txt 2>&1
./simulation -n 16384 >> out.txt 2>&1
./simulation -n 32768 >> out.txt 2>&1
./simulation -n 65536 >> out.txt 2>&1
./simulation -n 131072 >> out.txt 2>&1
