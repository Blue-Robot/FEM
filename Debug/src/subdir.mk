################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/main.cpp \
../src/mesh_function.cpp 

CU_SRCS += \
../src/FEM_common.cu 

CU_DEPS += \
./src/FEM_common.d 

OBJS += \
./src/FEM_common.o \
./src/main.o \
./src/mesh_function.o 

CPP_DEPS += \
./src/main.d \
./src/mesh_function.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-5.5/bin/nvcc -I/usr/local/lib/OpenMesh -I/usr/local/include/OpenMesh/Core -I/usr/local/cuda-5.5/samples/common/inc -G -g -O0 -gencode arch=compute_35,code=sm_35 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-5.5/bin/nvcc --device-c -G -I/usr/local/lib/OpenMesh -I/usr/local/include/OpenMesh/Core -I/usr/local/cuda-5.5/samples/common/inc -O0 -g -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-5.5/bin/nvcc -I/usr/local/lib/OpenMesh -I/usr/local/include/OpenMesh/Core -I/usr/local/cuda-5.5/samples/common/inc -G -g -O0 -gencode arch=compute_35,code=sm_35 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-5.5/bin/nvcc -I/usr/local/lib/OpenMesh -I/usr/local/include/OpenMesh/Core -I/usr/local/cuda-5.5/samples/common/inc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


