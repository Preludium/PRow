#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <iostream>

#define num_threads 8
#define show_results false


bool isprime(int num) {
    if (num % 2 == 0)
        return false;

    for (int i = 3; i <= round(sqrt(num)); i += 2) {
        if (num % i == 0)
            return false;
    }
    return true;
}


int main(int argc, char* argv[])
{
    double startclock, endclock;
    int iter = 0;
    int* result;
    int start, end;

    omp_set_num_threads(num_threads);

    std::cout << "type start and end of numbers range: ";
    std::cin >> start >> end;
    while (end < start || std::cin.fail() || start < 2) {
        std::cout << "incorrect data" << std::endl;
        std::cout << "type start and end of numbers range: ";
        std::cin >> start >> end;
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    result = new int[round((end - start) / 2)];

    if (start == 2) {
        result[iter] = 2;
        iter++;
    }

    if (start % 2 == 0)
        start++;

    startclock = omp_get_wtime();

#pragma omp parallel
    {
#pragma omp for
        for (int i = start; i <= end; i += 2) {
            if (isprime(i)) {
                //mo¿na daæ atomic na iter i te¿ bêdzie dzia³aæ (spoko alternatywa do testów)
#pragma omp critical
                {
                    result[iter] = i;
                    ++iter;
                }
            }
        }

    }
    endclock = omp_get_wtime();

    std::cout << std::endl << "result: ";
    if(show_results) {
        for (int i = 0; i < iter; ++i) {
            if (i % 10 == 0) {
                if (i < iter - 1)
                    std::cout << result[i] << ", ";
                else
                    std::cout << result[i] << std::endl;
            }
        }
    }
    std::cout << std::endl;
    std::cout << "ilosc liczb: " << iter << std::endl;

    printf("czas przetwarzania wynosi %f sekund\n", (endclock - startclock));
    delete[] result;
    return 0;
}
