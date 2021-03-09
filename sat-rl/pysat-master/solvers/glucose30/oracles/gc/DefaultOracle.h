#ifndef Glucose30_DefaultOracle_h
#define Glucose30_DefaultOracle_h

#include <assert.h>

#include "glucose30/oracles/gc/GCOracle.h"


namespace Glucose30 {

class DefaultOracle: public GCOracle
{
public:
    DefaultOracle(Solver *solver, std::function<double*()> cb, bool collectStats) :
        GCOracle(solver, cb, collectStats)
    {}

    ~DefaultOracle() {}

    void reduceDBDelegate(double* action) {
        vec<CRef> &learnts  = getLearnts();
        int nLearnts    = learnts.size();
        int limit = nLearnts / 2;

        for (int i = 0; i < nLearnts; i++) {
            Clause& c = ca(learnts[i]);
            if (c.lbd()>2 && c.size() > 2 && c.canBeDel() && !locked(c) && (i < limit))
                c.del(true);
            else {
                if(!c.canBeDel()) limit++; //we keep c, so we can delete an other clause
                c.setCanBeDel(true);       // At the next step, c can be delete
            }
        }
    }
};

}
#endif
