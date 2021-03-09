#include "solver.h"

#include <iostream>

#include <vector>

//#include <malloc.h>
#include <string>

#include <sys/time.h>
#include <sys/resource.h>

// using namespace std;
using namespace SharpSAT;

int main(int argc, char *argv[]) {

    std::string input_file;
    Solver theSolver;


    if (argc <= 1) {
        std::cout << "Usage: sharpSAT [options] [CNF_File]" << std::endl;
        std::cout << "Options: " << std::endl;
        std::cout << "\t -noPP  \t turn off preprocessing" << std::endl;
        std::cout << "\t -q     \t quiet mode" << std::endl;
        std::cout << "\t -t [s] \t set time bound to s seconds" << std::endl;
        std::cout << "\t -noCC  \t turn off component caching" << std::endl;
        std::cout << "\t -cs [n]\t set max cache size to n MB" << std::endl;
        std::cout << "\t -noIBCP\t turn off implicit BCP" << std::endl;
        std::cout << "\t" << std::endl;

        return -1;
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-noCC") == 0)
            theSolver.config().perform_component_caching = false;
        if (strcmp(argv[i], "-noIBCP") == 0)
            theSolver.config().perform_failed_lit_test = false;
        if (strcmp(argv[i], "-noPP") == 0)
            theSolver.config().perform_pre_processing = false;
        else if (strcmp(argv[i], "-q") == 0)
            theSolver.config().quiet = true;
        else if (strcmp(argv[i], "-v") == 0)
            theSolver.config().verbose = true;
        else if (strcmp(argv[i], "-t") == 0) {
            if (argc <= i + 1) {
                std::cout << " wrong parameters" << std::endl;
                return -1;
            }
            theSolver.config().time_bound_seconds = atol(argv[i + 1]);
            if (theSolver.config().verbose)
                std::cout << "time bound set to" << theSolver.config().time_bound_seconds << "s\n";
        } else if (strcmp(argv[i], "-cs") == 0) {
            if (argc <= i + 1) {
                std::cout << " wrong parameters" << std::endl;
                return -1;
            }
            theSolver.statistics().maximum_cache_size_bytes_ = atol(argv[i + 1]) * (uint64_t) 1000000;
        } else
            input_file = argv[i];
    }

    theSolver.solve(input_file);

//  std::cout << sizeof(LiteralID)<<"MALLOC_STATS:" << std::endl;
//  malloc_stats();

//  rusage ru;
//  getrusage(RUSAGE_SELF,&ru);
//
//   std::cout << "\nRus: " <<  ru.ru_maxrss*1024 << std::endl;
//  std::cout << "\nMALLINFO:" << std::endl;
//
//  std::cout << "total " << mallinfo().arena + mallinfo().hblkhd << std::endl;
//  std::cout <<  mallinfo().arena << "non-mmapped space allocated from system " << std::endl;
//  std::cout <<  mallinfo().ordblks << "number of free chunks " << std::endl;
//  std::cout <<  mallinfo().smblks<< "number of fastbin blocks " << std::endl;
//  std::cout <<  mallinfo().hblks<< " number of mmapped regions " << std::endl;
//  std::cout <<  mallinfo().hblkhd<< "space in mmapped regions " << std::endl;
//  std::cout <<  mallinfo().usmblks<< " maximum total allocated space " << std::endl;
//  std::cout <<  mallinfo().fsmblks<< "space available in freed fastbin blocks " << std::endl;
//  std::cout <<  mallinfo().uordblks<< " total allocated space " << std::endl;
//  std::cout <<  mallinfo().fordblks<< "total free space " << std::endl;
//  std::cout <<  mallinfo().keepcost<< " top-most, releasable (via malloc_trim) space " << std::endl;
    return 0;
}
