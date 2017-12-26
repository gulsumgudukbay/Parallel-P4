#!/bin/bash
# run script

/usr/local/cuda-9.1/bin/nvcc angle.cu -o angle

rm -f gudukbay_gulsum.output
touch gudukbay_gulsum.output

n_values=(1000000 5000000 10000000)
block_sizes=(32 64 128 256 512)


for(( j = 0; j < 3; j++))
do
  echo -e "Number of Elements:${n_values[$j]}\n" >> gudukbay_gulsum.output
  for (( i=0; i<5; i++ ))
  do
    echo -e "block size:${block_sizes[$i]}\n" >> gudukbay_gulsum.output
    ./angle ${n_values[$j]} ${block_sizes[$i]} >> gudukbay_gulsum.output
    echo -e "\n" >> gudukbay_gulsum.output
  done
done
