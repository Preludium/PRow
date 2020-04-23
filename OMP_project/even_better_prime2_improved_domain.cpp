// #include <stdio.h>
// #include <omp.h>
// #include <time.h>
// #include <math.h>
// #include <iostream>

// #define NUM_THREADS 8
// #define SHOW_RESULTS true
// #define MANUAL_INPUT false
// #define SHOW_NUMBER_OF_PRIMES true
// #define START_NUM 2
// #define END_NUM 65

// bool isPrime(int num)
// {
//    for (int i = 2; i < num; ++i)
//    {
//        if (num % i == 0)
//            return false;
//    }
//    return true;
// }

// int main(int argc, char* argv[])
// {
//    double startClock, endClock;
//    int iterPrime, iterNum;
//    int* numbers, * primes;
//    int start = START_NUM, end = END_NUM;

//    omp_set_num_threads(NUM_THREADS);

//    if (MANUAL_INPUT) {
//        std::cout << "Type start and end of numbers range: ";
//        std::cin >> start >> end;
//        while (end < start || std::cin.fail() || start < 2) {
//            std::cout << "Incorrect data" << std::endl;
//            std::cout << "Type start and end of numbers range: ";
//            std::cin >> start >> end;
//            std::cin.clear();
//            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
//        }
//    }

//    numbers = new int[end - start];
//    primes = new int[static_cast<unsigned>(round(sqrt(end)))];
//    int** primesHandler = new int* [NUM_THREADS];
//    int* counters = new int[NUM_THREADS];

//    iterNum = 0;
//    for (int i = start; i <= end; ++i)
//    {
//        numbers[iterNum] = i;
//        ++iterNum;
//    }

//    startClock = omp_get_wtime();

//    primes[0] = 2;
//    iterPrime = 1;

// #pragma omp parallel
//    {
//        int id = omp_get_thread_num();
//        counters[id] = 0;
//        primesHandler[id] = new int[static_cast<unsigned>(round(sqrt(end)))];

// #pragma omp for 
//        for (int i = 3; i <= static_cast<unsigned>(round(sqrt(end))); ++i)
//        {
//            if (i % 2 == 0)
//                continue;

//            if (isPrime(i))
//            {
//                primesHandler[id][counters[id]] = i;
//                ++counters[id];
//            }
//        }

// #pragma omp single 
//        {
//            for (int i = 0; i < NUM_THREADS; ++i) {
//                for (int j = 0; j < counters[i]; ++j) {
//                    primes[iterPrime] = primesHandler[i][j];
//                    ++iterPrime;
//                }
//            }

//            for (int i = 0; i < NUM_THREADS; ++i)
//                delete[] primesHandler[i];

//            delete[] primesHandler;
//            delete[] counters;
//        }

// #pragma omp barrier

//        for (int i = 0; i < iterPrime; ++i)
//        {
//            int currentlyChecked = start;

//            while (currentlyChecked % primes[i] != 0)
//                ++currentlyChecked;

//            if (currentlyChecked == primes[i])
//                currentlyChecked *= 2;

// #pragma omp for nowait schedule(static)
//            for (int j = currentlyChecked; j <= end; j += primes[i]) {
//                numbers[j - start] = 0;
//            }
//        }
//    }

//    endClock = omp_get_wtime();

//    int numberOfPrimes = 0;

//    if (SHOW_NUMBER_OF_PRIMES) {
//        std::cout << std::endl << "Result: ";
//        for (int i = 0; i < iterNum; ++i) {
//            if (numbers[i] != 0) {
//                if (SHOW_RESULTS) {
//                    if (i % 10 == 0)
//                        std::cout << std::endl;
//                    std::cout << numbers[i] << ", ";
//                }
//                numberOfPrimes++;
//            }
//        }
//        std::cout << std::endl;
//        std::cout << "Ilosc liczb: " << numberOfPrimes << std::endl;
//    }
//    printf("Czas przetwarzania wynosi %f sekund\n", (endClock - startClock));
//    delete[] numbers;
//    delete[] primes;
//    return 0;
// }
