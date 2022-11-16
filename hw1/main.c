#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
const int N = 100000000;
 
/// Function to compute the partial integral
double compute(double boundary, int k, double delta){
        double ans = 0;
        for (int i = 0; i < k; ++i) {
                double x = boundary*delta + i*delta;
                double first = (4*delta)/(1+x*x); //f(a)
                x += delta;
                double second = (4*delta)/(1+x*x); //f(b)
                ans += 0.5*(first + second); //1/2*(f(a) + f(b))
        }
        return ans;
}
 
int main(int argc, char *argv[]){
        MPI_Init(&argc, &argv);
        int size; ///number of processors
        int myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
 
        double delta = 1.0/N; //length of an interval
        double begin, end, total;
        //linear time
        MPI_Barrier(MPI_COMM_WORLD);
        begin = MPI_Wtime();
        if (myrank == 0) {
                printf("Answer using linear approach: %f\n", compute(0, N, delta));
        }
        MPI_Barrier(MPI_COMM_WORLD);
        end = MPI_Wtime();
        double linear_time = end - begin;
 
        //process time
        MPI_Barrier(MPI_COMM_WORLD);
        begin = MPI_Wtime();
        int k; // number of small intervals one process must compute
        k = N/size;
        double partial_answer;
        double first_boundary;
        double boundary;
        double result;
        result = 0;
        double remain = N%size;
        first_boundary = k + remain;
 
        if (myrank == 0) {
                result += compute(0, k + remain, delta); //processor 0 will handle the case when N is not divisible by number of processors 
                for (int i = 1; i < size; ++i) {
                        MPI_Recv(&partial_answer, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        result += partial_answer;
                }
                printf("Answer using process approach: %f\n", result);
        } else {
                boundary = first_boundary + k*(myrank - 1);
                partial_answer = compute(boundary, k, delta);
                MPI_Send(&partial_answer, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        }
 
        MPI_Barrier(MPI_COMM_WORLD);
        end = MPI_Wtime();
        double process_time = end - begin;
 
 
        if (myrank == 0) {
                printf("linear time: %f\n", linear_time);
                printf("process time: %f\n", process_time);
                printf("speed up: %f\n", linear_time/process_time);
        }
        MPI_Finalize();
        return 0;
}
