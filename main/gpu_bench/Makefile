# GPU Hash Table Benchmark — simple Makefile, no cmake needed
#
# Usage:
#   make SM=90   # H100
#   make SM=80   # A100
#   make SM=89   # RTX 6000 Ada / L40S
#   make SM=75   # GTX series (desktop nodes)
#
# Default targets sm_90 (H100).

SM ?= 90

NVCC     = nvcc
NVCCFLAGS = -O2 -arch=sm_$(SM) --ptxas-options=-v

TARGET = gpu_benchmark

$(TARGET): benchmark.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lm

clean:
	rm -f $(TARGET)

.PHONY: clean
