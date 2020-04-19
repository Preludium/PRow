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
//	double startClock, endClock;
//	int iterPrime, iterNum;
//    int *numbers, *primes;
//    int start, end;
//
//	omp_set_num_threads(NUM_THREADS);
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
//
//    numbers = new int[end - start];
//    primes = new int[round(sqrt(end)) - 2];
//    iterNum = 0;
//    for (int i = start; i <= end; ++i) {
//        numbers[iterNum] = i;
//        ++iterNum;
//    }
//
//	startClock = omp_get_wtime();
//
//    primes[0] = 2;
//    iterPrime = 1;
//    for (int i = 3; i <= round(sqrt(end)); ++i) {
//        if (i % 2 == 0)
//            continue;
//
//        if (isPrime(i)) {
//            primes[iterPrime] = i;
//            ++iterPrime;
//        }
//    }
//
//    for (int i = 0; i < iterPrime; ++i) {
//        int currentlyChecked = start;
//
//        while (currentlyChecked % primes[i] != 0)
//            ++currentlyChecked;
//        
//        if (currentlyChecked == primes[i])
//            currentlyChecked *= 2;
//
//        while (currentlyChecked <= end) {
//            numbers[currentlyChecked - start] = 0;
//            currentlyChecked += primes[i];
//        }
//    }
//
//	endClock = omp_get_wtime();
//
//    int numberOfPrimes = 0;
//
//
//    std::cout << std::endl << "Result: ";
//    for (int i = 0; i < iterNum; ++i) {
//        if (numbers[i] != 0) {
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
//	printf("Czas przetwarzania wynosi %f sekund\n", (endClock - startClock));
//    delete[] numbers;
//    delete[] primes;
//	return 0;
//}