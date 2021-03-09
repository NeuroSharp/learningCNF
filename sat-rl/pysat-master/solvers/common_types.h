#ifndef COMMON_TYPES
#define COMMON_TYPES


#include <vector>
#include <exception>
#include <unordered_map>

using namespace std;

// A key-value mapping that contains solver specific stats information
typedef std::unordered_map<std::string, double> StatsMap;

// Unsigned Dequeue
typedef std::vector<unsigned> u_vec;
// Clause Variable Insident Graph (CVIG)
typedef std::tuple<u_vec, u_vec, std::vector<u_vec>> CVIG;

class TerminatedException: public std::exception
{
  virtual const char* what() const throw()
  {
    return "Solver terminated";
  }
};

#endif /* COMMON_TYPES_H */
