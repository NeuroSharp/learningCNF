#ifndef Glucose30_GCOracle_h
#define Glucose30_GCOracle_h

#include <assert.h>
#include <string>
#include <functional>

#include "glucose30/core/Solver.h"
#include "glucose30/core/SolverTypes.h"
#include "glucose30/mtl/Sort.h"
#include "common_types.h"


namespace Glucose30 {

class GCOracle
{

protected:
    Solver  *solver       = NULL;
    double  del_div_avg   = 0;
    double  keep_div_avg  = 0;
    double  kept_rate_avg = 0;
    bool    collectStats  = false;

    std::function<double*()>    callback;

    // Anti-pattern: exposing some protected members of the friend's class Solver
    // to the subclasses of GCOracle:
    Clause&       ca            (uint32_t ref)  { return _ca[ref]; }
    bool          locked        (Clause& c)     { return solver->locked(c); }
    vec<CRef>&    getLearnts    ()              { return solver->learnts;   }
    int           getNumDels    ()              { return solver->nDels();   }
    virtual void  reduceDBDelegate  (double* res) = 0;

private:
    ClauseAllocator &_ca;

public:
    GCOracle(Solver *solver, std::function<double*()> cb, bool collectStats) :
        solver(solver),
        callback(cb),
        collectStats(collectStats),
        _ca(solver->ca)
    {}

    virtual ~GCOracle() {}

    void      setSolver   (Solver   *psolver)          { solver = psolver; }
    Solver*   getSolver   ()                           { return solver; }
    void      setCallback (std::function<double*()> cb)  { callback = cb; }

    void      reduceDB ()
    {
        int       i, j        = 0;
        int       del_div     = 0;
        int       keep_div    = 0;
        int       nbCurrentRemovedClauses = 0;
        vec<CRef> &learnts    = solver->learnts;
        int       nLearnts    = learnts.size() - getNumDels();
        uint64_t  nbReduceDB  = solver->nbReduceDB;
        std::vector<bool> vanilla;

        updateGCFreq();

        double* action = (double*) GCOracle::callback();
        // TODO: Add logic to separate training vs testing logics:
        //       std::vector<double> action = Model::forward(feat);

        // Checking to see if terminate() was called in the callback...
        if (solver->asynch_interrupt) throw TerminatedException();

        if (collectStats)
            vanilla = getDefaultActions(); //Only for stats computation

        reduceDBDelegate(action);

        for (i = j = 0; i < nLearnts; i++){
            Clause& c = _ca[learnts[i]];

            // Deep Solver behavior
            if (c.del()) {
                // if (c.del() > 1) printf("********* Clause is delled more than once: %i\n", c.del());
                if (c.tagged())
                    learnts[j++] = learnts[i];

                solver->removeClause(learnts[i]);
                solver->nbRemovedClauses++;
                nbCurrentRemovedClauses++;
            }
            else {
                learnts[j++] = learnts[i];
            }

            if (collectStats) {
                if (c.del() && !vanilla[i])
                    del_div++;
                if (!c.del() && vanilla[i])
                    keep_div++;
            }
        }
        for (i = nLearnts; i < learnts.size(); i++) {
            Clause& c = _ca[learnts[i]];
            // if(!c.del()) printf("FAAAAAAAAAAAAAIL\n");
            learnts[j++] = learnts[i];
        }

        learnts.shrink(i - j);
        solver->checkGarbage();

        if (collectStats) {
            // Incremental means for divergance
            del_div_avg  += ((del_div*100.0)/nLearnts - del_div_avg)/ nbReduceDB;
            keep_div_avg += ((keep_div*100.0)/nLearnts - keep_div_avg)/ nbReduceDB;

            // Incremental percent kept
            kept_rate_avg += ((nbCurrentRemovedClauses*100.0)/nLearnts - kept_rate_avg)/ nbReduceDB;
        }
    }

    virtual void getStats(StatsMap& stats)
    {
        if (collectStats) {
            stats["del_divergence"] = del_div_avg;
            stats["keep_div"]       = keep_div_avg;
        }
    }

private:
    inline std::vector<bool> getDefaultActions()
    {
        vec<CRef> &learnts  = solver->learnts;
        int       nLearnts  = learnts.size();
        int       limit     = nLearnts / 2; // Limit for vanilla case
        std::vector<bool> actions(nLearnts);

        for (int i = 0; i < nLearnts; i++){
            Clause& c = _ca[learnts[i]];

            // Default LBD behavior
            actions[i] = false;
            if (c.lbd() > 2 && c.size() > 2 && c.canBeDel() && !locked(c) && (i < limit))
                actions[i] = true;
            else
                if (!c.canBeDel()) limit++;
        }

        return actions;
    }

protected:
    void updateGCFreq()
    {
        solver->updateGCFreq();
    }

    // struct reduceDB_lt {
    //     ClauseAllocator& ca;
    //     reduceDB_lt(ClauseAllocator& ca_) : ca(ca_) {}
    //     bool operator () (CRef x, CRef y) {

    //     // Main criteria... Like in MiniSat we keep all binary clauses
    //     if(ca[x].size()> 2 && ca[y].size()==2) return 1;

    //     if(ca[y].size()> 2 && ca[x].size()==2) return 0;
    //     if(ca[x].size()==2 && ca[y].size()==2) return 0;

    //     // Second one  based on literal block distance
    //     if(ca[x].lbd()> ca[y].lbd()) return 1;
    //     if(ca[x].lbd()< ca[y].lbd()) return 0;

    //     // Finally we can use old activity or size, we choose the last one
    //     return ca[x].activity() < ca[y].activity();
    //     //return x->size() < y->size();

    //     //return ca[x].size() > 2 && (ca[y].size() == 2 || ca[x].activity() < ca[y].activity()); }
    //     }
    // };
};

}
#endif
