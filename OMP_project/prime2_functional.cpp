//#include <stdio.h>
//#include <omp.h>
//#include <time.h>
//#include <math.h>
//#include <iostream>
//
//#define NUM_THREADS 8
//#define SHOW_RESULTS false
//
//bool isPrime(int num) {
//    for (int i = 2; i < num; ++i) {
//        if (num % i == 0)
//            return false;
//    }
//    return true;
//}
//
//
//int main(int argc, char* argv[])
//{
//    double startClock, endClock;
//    int iterPrime, iterNum;
//    int* numbers, *primes;
//    int start, end;
//
//    omp_set_num_threads(NUM_THREADS);
//
//    std::cout << "Type start and end of numbers range: ";
//    std::cin >> start >> end;
//    while (end < start || std::cin.fail() || start < 2) {
//        std::cout << "Incorrect data" << std::endl;
//        std::cout << "Type start and end of numbers range: ";
//        std::cin >> start >> end;
//        std::cin.clear();
//        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
//    }
//
//    primes = new int[round(sqrt(end)) - 2];
//    numbers = new int[end - start];
//    iterNum = 0;
//    for (int i = start; i <= end; ++i) {
//        numbers[iterNum] = i;
//        ++iterNum;
//    }
//
//
//    iterPrime = 0;
//    primes[iterPrime] = 2;
//    iterPrime++;
//
//
//    startClock = omp_get_wtime();
//
////  Uzyskanie wszystkich liczb pierwszych do sqrt(MAX)
//
//    for (int i = 3; i <= (int)round(sqrt(end)); ++i) {
//        if (i % 2 == 0)
//            continue;
//
//        if (isPrime(i)) {
//            primes[iterPrime] = i;
//            ++iterPrime;
//        }
//    }
//
//#pragma omp parallel
//    {
//    int *threadPrimes = new int[round((round(sqrt(end)) - 2) / NUM_THREADS) + 1];
//    int threadIterPrime = 0;
//
////  Dystrybucja wczeœniej uzyskanych liczb pierwszych miêdzy w¹tkami
//#pragma omp for
//    for (int i = 0; i < iterPrime; ++i) {
//        threadPrimes[threadIterPrime] = primes[i];
//        threadIterPrime++;
//    }
//
//    for (int i = 0; i < iterNum; ++i) {
//        for (int j = 0; j < threadIterPrime; ++j) {
//            if (numbers[i] == threadPrimes[j]) {
//                break;
//            }
//
//            if (numbers[i] % threadPrimes[j] == 0) {
//                numbers[i] = 0;
//                break;
//            }
//        }
//    }
//    }
//    endClock = omp_get_wtime();
//
//    int numberOfPrimes = 0;
//
//    std::cout << std::endl << "Result: ";
//
//    for (int i = 0; i < iterNum; ++i)
//    {
//        if (numbers[i] != 0)
//        {
//            if (SHOW_RESULTS) {
//                if (i % 10 == 0)
//                    std::cout << std::endl;
//                std::cout << numbers[i] << ", ";
//            }
//            numberOfPrimes++;
//        }
//    }
//    std::cout << std::endl;
//    std::cout << "Ilosc liczb: " << numberOfPrimes << std::endl;
//    printf("Czas przetwarzania wynosi %f sekund\n", (endClock - startClock));
//    delete[] numbers;
//    delete[] primes;
//    return 0;
//}
