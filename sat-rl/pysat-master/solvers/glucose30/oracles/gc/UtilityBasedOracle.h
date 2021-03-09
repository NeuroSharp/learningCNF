#ifndef Glucose30_UtilityBasedOracle_h
#define Glucose30_UtilityBasedOracle_h

#include <assert.h>

#include "glucose30/oracles/gc/GCOracle.h"


namespace Glucose30 {

class UtilityBasedOracle: public GCOracle
{
public:
    UtilityBasedOracle(Solver *solver, std::function<double*()> cb, bool collectStats) :
        GCOracle(solver, cb, collectStats)
    {}

    ~UtilityBasedOracle() {}

    void reduceDBDelegate(double* action)
    {
        //Not implemented

        // sort clauses if you need
    }

    void updateGCFreq(){
        if (solver->gc_freq == GCFreq::UTILITY) {
            vec<CRef> &learnts = solver->learnts;
            sort(learnts, utility_lt(solver->ca));

            // NOTE: the values are taken straight from the glucose, use different numbers/logic here...
            if(solver->ca[learnts[learnts.size() / RATIOREMOVECLAUSES]].utility()<=3) solver->specialIncReduceDB += solver->specialIncReduceDB;
            // Useless :-)
            if(solver->ca[learnts.last()].utility()<=5)  solver->specialIncReduceDB += solver->specialIncReduceDB;
        } else
            Oracle::updateGCFreq();
    }
private:
    struct utility_lt {
        ClauseAllocator& ca;
        utility_lt(ClauseAllocator& ca_) : ca(ca_) {}
        bool operator () (CRef x, CRef y) {
            if (ca[x].size() == 2) return 0;
            if (ca[y].size() == 2) return 1;

            if (ca[x].utility() > ca[y].utility()) return 1;
            if (ca[x].utility() < ca[y].utility()) return 0;

            return ca[x].activity() < ca[y].activity();
        }
    };

};

}
#endif
