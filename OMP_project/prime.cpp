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
//    if (num % 2 == 0)
//        return false;
//
//    for (int i = 3; i <= round(sqrt(num)); i += 2) {
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
//	int iter = 0;
//    int *result;
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
//    result = new int[round((end - start) / 2)];
//
//    if (start == 2) {
//        result[iter] = 2;
//        iter++;
//    }
//
//    if (start % 2 == 0)
//        start++;
//
//	startClock = omp_get_wtime();
//
//    for (int i = start; i <= end; i += 2) {
//        if (isPrime(i)) {
//            result[iter] = i;
//            ++iter;
//        }
//    }
//	endClock = omp_get_wtime();
//
//    if(SHOW_RESULTS) {
//        std::cout << std::endl << "Result: ";
//        for (int i = 0; i < iter; ++i) {
//            if (i % 10 == 0) {
//                if (i < iter - 1)
//                    std::cout << result[i] << ", ";
//                else
//                    std::cout << result[i] << std::endl;
//            }
//        }
//    }
//    std::cout << std::endl;
//    std::cout << "Ilosc liczb: " << iter << std::endl;
//	printf("Czas przetwarzania wynosi %f sekund\n", (endClock - startClock));
//    delete[] result;
//	return 0;
//}
