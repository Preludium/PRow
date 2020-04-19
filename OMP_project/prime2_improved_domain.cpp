//#include <stdio.h>
//#include <omp.h>
//#include <time.h>
//#include <math.h>
//#include <iostream>
//
//#define NUM_THREADS 8
//#define SHOW_RESULTS false
//
//bool isPrime(int num)
//{
//    for (int i = 2; i < num; ++i)
//    {
//        if (num % i == 0)
//            return false;
//    }
//    return true;
//}
//
//int main(int argc, char *argv[])
//{
//    double startClock, endClock;
//    int iterPrime, iterNum;
//    int *numbers, *primes;
//    int start, end;
//
//    omp_set_num_threads(NUM_THREADS);
//
//    std::cout << "Type start and end of numbers range: ";
//    std::cin >> start >> end;
//    while (end < start || std::cin.fail() || start < 2)
//    {
//        std::cout << "Incorrect data" << std::endl;
//        std::cout << "Type start and end of numbers range: ";
//        std::cin >> start >> end;
//        std::cin.clear();
//        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
//    }
//
//    numbers = new int[end - start];
//    primes = new int[round(sqrt(end)) - 2];
//    iterNum = 0;
//    for (int i = start; i <= end; ++i)
//    {
//        numbers[iterNum] = i;
//        ++iterNum;
//    }
//
//    startClock = omp_get_wtime();
//
//#pragma omp parallel
//    {
//        int localEnd = end;
//#pragma omp single
//        {
//            primes[0] = 2;
//            iterPrime = 1;
//            for (int i = 3; i <= round(sqrt(end)); ++i)
//            {
//                if (i % 2 == 0)
//                    continue;
//
//                if (isPrime(i))
//                {
//                    primes[iterPrime] = i;
//                    ++iterPrime;
//                }
//            }
//        }
//
//        //    dzi�ki temu wiemy gdzie sko�czy�
//        //    w�tkom przetwarzaj�cym �rodek tablicy dajemy pewien zapas (�eby �adna liczba nie uciek�a)
//        //    natomiast ostatni w�tek (NUM_THREADS - 1) ko�czy tam gdzie powinno si� sko�czy�, �eby nie by�o nadmiarowych liczb
//        int part = ceil(double(end - start) / NUM_THREADS);
//        if (omp_get_thread_num() != NUM_THREADS - 1)
//            localEnd = start + part * (omp_get_thread_num() + 1);
//        std::cout << omp_get_thread_num() << ": " << localEnd << std::endl;
//
//        for (int i = 0; i < iterPrime; ++i)
//        {
//            int currentlyChecked = start;
//
//            while (currentlyChecked % primes[i] != 0)
//                ++currentlyChecked;
//
//            currentlyChecked += part * omp_get_thread_num();
//            currentlyChecked -= currentlyChecked % primes[i];
//
//            if (currentlyChecked == primes[i])
//                currentlyChecked *= 2;
//
//            while (currentlyChecked <= localEnd)
//            {
//                numbers[currentlyChecked - start] = 0;
//                currentlyChecked += primes[i];
//            }
//        }
//    }
//
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
