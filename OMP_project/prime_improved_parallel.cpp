#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <iostream>

#define NUM_THREADS 8
#define SHOW_RESULTS false
#define MANUAL_INPUT false
#define SHOW_NUMBER_OF_PRIMES true
#define START_NUM 2
#define END_NUM 50000000

bool isPrime(int num) {
   for (int i = 3; i <= int(sqrt(num)); i += 2) {
       if (num % i == 0)
           return false;
   }
   return true;
}

int main(int argc, char* argv[])
{
   double startClock, endClock;
   int iterPrime = 0;
   int *primes;
   int start = START_NUM, end = END_NUM;

   omp_set_num_threads(NUM_THREADS);

   if (MANUAL_INPUT) {
       std::cout << "Type start and end of numbers range: ";
       std::cin >> start >> end;
       while (end < start || std::cin.fail() || start < 2) {
           std::cout << "Incorrect data" << std::endl;
           std::cout << "Type start and end of numbers range: ";
           std::cin >> start >> end;
           std::cin.clear();
           std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
       }
   }

   primes = new int[static_cast<unsigned>((end - start) / 4)];
   int** primesHandler = new int*[NUM_THREADS];
   int* counters = new int[NUM_THREADS];

   startClock = omp_get_wtime();

    if (start == 2) {
        primes[0] = 2;
        ++iterPrime;
    }

    if (start % 2 == 0) 
        ++start;

#pragma omp parallel
   {
       int id = omp_get_thread_num();
       counters[id] = 0;
       primesHandler[id] = new int[static_cast<unsigned>((end - start) / NUM_THREADS)];

#pragma omp for 
       for (int i = start; i <= end; i += 2)
       {
           if (isPrime(i))
           {
               primesHandler[id][counters[id]] = i;
               ++counters[id];
           }
       }
    }
    
    for (int i = 0; i < NUM_THREADS; ++i) {
        for (int j = 0; j < counters[i]; ++j) {
            primes[iterPrime] = primesHandler[i][j];
            ++iterPrime;
        }
    }

    endClock = omp_get_wtime();
    
    for (int i = 0; i < NUM_THREADS; ++i)
        delete[] primesHandler[i];

    delete[] primesHandler;
    delete[] counters;


   if (SHOW_NUMBER_OF_PRIMES) {
       std::cout << std::endl << "Result: ";
       for (int i = 0; i < iterPrime; ++i) {
            if (SHOW_RESULTS) {
                if (i % 10 == 0)
                    std::cout << std::endl;
                std::cout << primes[i] << ", ";
            }
       }
       std::cout << std::endl;
       std::cout << "Ilosc liczb: " << iterPrime << std::endl;
   }
   printf("Czas przetwarzania wynosi %f sekund\n", (endClock - startClock));
   delete[] primes;
   return 0;
}
