#ifndef Glucose30_LBDThresholdOracle_h
#define Glucose30_LBDThresholdOracle_h

#include <assert.h>

#include "glucose30/oracles/gc/GCOracle.h"


namespace Glucose30 {

class LBDThresholdOracle: public GCOracle
{
public:
    LBDThresholdOracle(Solver *solver, std::function<double*()> cb, bool collectStats) :
        GCOracle(solver, cb, collectStats)
    {}

    ~LBDThresholdOracle() {}

    void reduceDBDelegate(double* action)
    {
        uint64_t  nbReduceDB  = solver->nbReduceDB;
        vec<CRef> &learnts    = getLearnts();
        int       nLearnts    = learnts.size();
        double    thresh      = ((double*) action)[0];

        if (solver->verbosity >= 2)
            printf("LBD threshold at %lluth reduce db: %f\n", nbReduceDB, thresh);

        for (int i = 0; i < nLearnts; i++) {
            Clause& c = ca(learnts[i]);

            if (c.lbd()>=thresh && c.canBeDel() && !locked(c))
                c.del(true);
            else
                c.setCanBeDel(true);       // At the next step, c can be delete
        }

        if (collectStats) {
            // Incremental means and S(n) for threshold
            // S(n) = S(n-1) + [x(n) - avg(n-1)][x(n)-avg(n)]
            // where S(n) = std(n)^2 * n
            double old_thresh_avg = thresh_avg;
            thresh_avg += (thresh - thresh_avg)/nbReduceDB;
            thresh_S += (thresh - old_thresh_avg)*(thresh - thresh_avg);
            thresh_std = sqrt(thresh_S/nbReduceDB);
        }
    }

    void getStats(StatsMap& stats)
    {
        if (collectStats) {
            stats["thresh_avg"] = thresh_avg;
            stats["thresh_std"] = thresh_std;
        }

        GCOracle::getStats(stats);
    }

protected:
    double    thresh_avg = 0;
    double    thresh_S   = 0;  // S(n) =std(n)^2 * n
    double    thresh_std = 0;
};

}
#endif
