################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/angle.cu \
../src/deviceProps.cu \
../src/vectorAdd.cu 

OBJS += \
./src/angle.o \
./src/deviceProps.o \
./src/vectorAdd.o 

CU_DEPS += \
./src/angle.d \
./src/deviceProps.d \
./src/vectorAdd.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.1/bin/nvcc -I"/usr/local/cuda-9.1/samples/0_Simple" -I"/usr/local/cuda-9.1/samples/common/inc" -I"/Users/gulsum/Documents/UNV4/CS 426/Parallel-P4" -G -g -O0 -gencode arch=compute_30,code=sm_30  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.1/bin/nvcc -I"/usr/local/cuda-9.1/samples/0_Simple" -I"/usr/local/cuda-9.1/samples/common/inc" -I"/Users/gulsum/Documents/UNV4/CS 426/Parallel-P4" -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


