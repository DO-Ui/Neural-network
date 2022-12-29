Note:
When calling cuda functions, [x,y] is required
x stands for the number of tasks
y stands for the number of threads per task

This means that the total number of threads is x\*y and the function will be run x\*y times

[1, 16] for example is simply 16 threads running once
This will cause the function to be run 16 times in order
Example output of [1, 16] (just printing the index):
```
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
```
[16, 1] acomplishes a similar function but in a different way
Rather than execute the function in order, the functions are called in parallel
Example output of [16, 1] (just printing the index):
```
1
5
9
13
3
7
11
0
15
4
8
12
2
6
10
14
```
The absolute index of thread can be found using cuda.grid(1)
This simply returns the index of the thread in the grid

cuda.threadIdx.x returns the index of the thread assigned to the task
Example of cuda.threadIdx.x with [16, 1]:
```
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
```
The reason this is all 0's is because the function is being run 16 times in parallel and so the index of the thread is always 0

cuda.blockIdx.x returns the index of the task
Example of cuda.blockIdx.x with [16, 1]:
```
1
5
9
13
3
0
7
4
11
8
15
12
2
6
10
14
```
This shows that 16 tasks are being run in parallel

If we use [1, 16] the index of the task would always be 0 but the index of the thread would be 0-15

Note: cuda kernals are infact 3D but we are using them as a 2D array with the third dimension being 1 for simplicity
