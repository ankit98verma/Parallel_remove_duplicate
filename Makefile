# CS 179 Project Unix Makefile
# Written by Loko Kung, 2018
# Modified by Ankit Verma, 2020
#


# Directory for object files
OBJDIR = bin

# Product Names
CUDA_OBJ = $(OBJDIR)/cuda.o

# Input Names
CUDA_FILES = src/cuda_sorting.cu
CPP_FILES =  src/main.cpp

# ------------------------------------------------------------------------------

# CUDA Compiler and Flags
CUDA_PATH = 
CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_BIN_PATH = $(CUDA_PATH)/bin
CUDA_LIB_PATH = $(CUDA_PATH)/lib64

NVCC = $(CUDA_BIN_PATH)/nvcc

# OS-architecture specific flags
# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
NVCC_FLAGS := -m32
else
NVCC_FLAGS := -m64
endif
NVCC_FLAGS += -g -dc -Wno-deprecated-gpu-targets --std=c++11 \
             --expt-relaxed-constexpr
NVCC_INCLUDE =
NVCC_LIBS = 
NVCC_GENCODES = -gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_61,code=sm_61 \
		-gencode arch=compute_61,code=compute_61

# CUDA Object Files
CUDA_OBJ_FILES = $(addprefix $(OBJDIR)/, $(notdir $(addsuffix .o, $(CUDA_FILES))))

# ------------------------------------------------------------------------------

# CUDA Linker and Flags
CUDA_LINK_FLAGS = -dlink -Wno-deprecated-gpu-targets

# ------------------------------------------------------------------------------

# C++ Compiler and Flags
GPP = g++
FLAGS = -g -Wall -D_REENTRANT -std=c++0x -pthread
INCLUDE = -I$(CUDA_INC_PATH)
LIBS = -L$(CUDA_LIB_PATH) -lcudart -lcurand

# ------------------------------------------------------------------------------
# Make Rules (Project specific)
# ------------------------------------------------------------------------------

# C++ Object Files
OBJ_GPU = $(addprefix $(OBJDIR)/, $(addprefix gpu-, $(notdir $(addsuffix .o, $(CPP_FILES)))))

# Top level rules
all: sort

sort: $(OBJ_GPU) $(CUDA_OBJ) $(CUDA_OBJ_FILES)
	$(GPP) $(FLAGS) -o sort $(INCLUDE) $^ $(LIBS) 

# Compile C++ Source Files
$(OBJDIR)/gpu-%.cpp.o: src/%.cpp
	$(GPP) $(FLAGS) -c -o $@ $(INCLUDE) $< 

# Compile CUDA Source Files
$(OBJDIR)/%.cu.o: src/%.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_GENCODES) -c -o $@ $(NVCC_INCLUDE) $<

cuda: $(CUDA_OBJ_FILES) $(CUDA_OBJ)

# Make linked device code
$(CUDA_OBJ): $(CUDA_OBJ_FILES)
	$(NVCC) $(CUDA_LINK_FLAGS) $(NVCC_GENCODES) -o $@ $(NVCC_INCLUDE) $^


# Clean everything including temporary Emacs files
clean:
	rm -f sort *.o $(OBJDIR)/*.o *~
	rm -f src/*~

.PHONY: clean
