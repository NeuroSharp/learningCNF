#ifndef Glucose30_ThreeValuedOracle_h
#define Glucose30_ThreeValuedOracle_h

#include <assert.h>

#include "glucose30/oracles/gc/GCOracle.h"


namespace Glucose30 {


/* This oracle first performs the Default clause reduction and then follows the actions
    received form the python side. Actions are:
        -1: Delete the clause
        0 : Don't care (Follow Glucose's decision)
        1 : Keep the clause
*/
class ThreeValuedOracle: public GCOracle
{
public:
    ThreeValuedOracle(Solver *solver, std::function<double*()> cb, bool collectStats) :
        GCOracle(solver, cb, collectStats)
    {}

    ~ThreeValuedOracle() {}

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

        for (int i = 0; i < nLearnts; i++) {
            Clause& c = ca(learnts[i]);
            if (action[i] == -1 && !locked(c)) // Delete
                c.del(true);
            else if (action[i] == 1) // Keep
                c.del(false);
            // else Don't care
        }
    }
};

}
#endif
