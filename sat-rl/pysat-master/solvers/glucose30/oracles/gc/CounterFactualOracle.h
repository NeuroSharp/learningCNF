#ifndef Glucose30_CounterFactualOracle_h
#define Glucose30_CounterFactualOracle_h

#include <assert.h>

#include "glucose30/oracles/gc/GCOracle.h"


namespace Glucose30 {

class CounterFactualOracle: public GCOracle
{
public:
    CounterFactualOracle(Solver *solver, std::function<double*()> cb, bool collectStats) :
        GCOracle(solver, cb, collectStats)
    {}

    ~CounterFactualOracle() {}

    void reduceDBDelegate(double* tags) {
        vec<CRef> &learnts  = getLearnts();
        int nLearnts    = learnts.size() - getNumDels();
        int limit = nLearnts / 2;

        for (int i = 0; i < nLearnts; i++) {
            Clause& c = ca(learnts[i]);

            if (((double*) tags)[i])
                c.tag();

            if (c.lbd() > 2 && c.size() > 2 && c.canBeDel() && !locked(c) && (i < limit))
                c.del(true); // Mark for delete
                // if (c.del()) printf("Clause already marked for DEL. It shouldn't be delled again\n");
            else {
                if (!c.canBeDel()) limit++; //we keep c, so we can delete an other clause
                c.setCanBeDel(true);       // At the next step, c can be delete
            }
        }
    }
};

}
#endif
