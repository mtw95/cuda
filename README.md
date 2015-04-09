# cuda

Input Parameters:

-t 20 (sets elapsed time to 20 seconds)

-s 100000 (sets number of elements in each array to 100,000)

-c 3 (sets the number of threads to create to 3)*


*-c is only included on tests that stress the cpu (CPUbench and GPU_CPUbench). GPU_CPUbench creates an extra thread to send kernel calls to the GPU)
