#ifndef Glucose30_PercentageOracle_h
#define Glucose30_PercentageOracle_h

#include <assert.h>

#include "glucose30/oracles/gc/GCOracle.h"


namespace Glucose30 {

class PercentageOracle: public GCOracle
{
public:
    PercentageOracle(Solver *solver, std::function<double*()> cb, bool collectStats) :
        GCOracle(solver, cb, collectStats)
    {}

    ~PercentageOracle() {}

    void reduceDBDelegate(double* action) {
        uint64_t  nbReduceDB = solver->nbReduceDB;
        vec<CRef> &learnts   = getLearnts();
        int       nLearnts   = learnts.size();
        double    percent    = ((double*) action)[0];
        double    thresh     = ((double*) action)[1];

        if (solver->verbosity >= 2)
            printf("(LBD threshold, percentage) at %lluth reduce db: (%f, %f)\n", nbReduceDB, thresh, percent);

        int limit = learnts.size() * percent;
        for (int i = 0; i < nLearnts; i++) {
            Clause& c = ca(learnts[i]);
            if (c.lbd() > thresh && c.size() > 2 && c.canBeDel() && !locked(c) && (i < limit))
                c.del(true);
            else {
                if(!c.canBeDel()) limit++; //we keep c, so we can delete an other clause
                c.setCanBeDel(true);       // At the next step, c can be delete
            }
        }

        if (collectStats) {
            // Incremental means and S(n) for threshold
            // S(n) = S(n-1) + [x(n) - avg(n-1)][x(n)-avg(n)]
            // where S(n) = std(n)^2 * n
            double old_thresh_avg = thresh_avg;
            thresh_avg += (thresh - thresh_avg)/nbReduceDB;
            thresh_S += (thresh - old_thresh_avg)*(thresh - thresh_avg);
            thresh_std = sqrt(thresh_S/nbReduceDB);

            double old_percent_avg = percent_avg;
            percent_avg += (percent - percent_avg)/nbReduceDB;
            percent_S += (percent - old_percent_avg)*(percent - percent_avg);
            percent_std = sqrt(percent_S/nbReduceDB);
        }
    }

    void getStats(StatsMap& stats)
    {
        if (collectStats) {
            stats["thresh_avg"] = thresh_avg;
            stats["thresh_std"] = thresh_std;

            stats["percent_avg"] = percent_avg;
            stats["percent_std"] = percent_std;
        }

        GCOracle::getStats(stats);
    }

protected:
    double    thresh_avg = 0;
    double    thresh_S   = 0;    // S(n) =std(n)^2 * n
    double    thresh_std = 0;

    double    percent_avg = 0;
    double    percent_S   = 0;   // S(n) =std(n)^2 * n
    double    percent_std = 0;
};

}
#endif
