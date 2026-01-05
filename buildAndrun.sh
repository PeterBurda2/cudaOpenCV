#!/bin/bash

echo "Build the code"
cmake -S . -B build
cmake --build build -j$(nproc)

echo "Running the code"

./build/perforamnceTest