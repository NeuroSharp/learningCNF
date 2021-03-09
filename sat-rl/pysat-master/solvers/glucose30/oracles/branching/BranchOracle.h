#ifndef Glucose30_BranchOracle_h
#define Glucose30_BranchOracle_h

#include <assert.h>
#include <string>
#include <functional>

#include "glucose30/core/Solver.h"
#include "glucose30/core/SolverTypes.h"
#include "glucose30/mtl/Sort.h"
#include "common_types.h"


namespace Glucose30 {

enum class BRTrigger {STEP_CNT, OP_CNT, CONFLICT};

class BranchOracle
{

protected:
    Solver  *solver       = NULL;
    std::function<double*()>    callback;
    BRTrigger brTrigger;
    int       brFreq;

public:
    BranchOracle(Solver *solver, std::function<double*()> cb, BRTrigger brTrigger, int brFreq) :
        solver(solver),
        callback(cb),
        brTrigger(brTrigger),
        brFreq(brFreq)

    {}

    virtual ~BranchOracle() {}

    void      setSolver   (Solver   *psolver)         { solver = psolver; }
    Solver*   getSolver   ()                          { return solver; }
    void      setCallback (std::function<double*()> cb)  { callback = cb; }

    void       priodic_rest() {
        if (!triggered()) return;

        double* scores = (double*) BranchOracle::callback();
        if (scores[0] < 0) return;

        // 1. Setting the Variable Scores
        for (int i = 0; i < solver->nVars(); i++)
            solver->activity[i] = scores[i];
        // 2. Resetting the variable increment to 1.0
        solver->var_inc = 1.0;
        // 3. Rebuilding the order-heap
        vec<Var> vs;
        for (Var v = 0; v < solver->nVars(); v++)
            if (solver->decision[v])// && solver->value(v) == g3l_Undef)
                vs.push(v);
        solver->order_heap.build(vs);

        // Checking to see if terminate() was called in the callback...
        if (solver->asynch_interrupt) throw TerminatedException();
    }

private:
    inline bool triggered() {
        int currVal;
        switch(brTrigger) {
            case BRTrigger::OP_CNT:
                currVal = solver->op_cnt;
                break;
            case BRTrigger::CONFLICT:
                currVal = solver->conflicts;
                break;
            default: // STEP_CNT
                currVal = solver->decisions;
                break;
        }

        return currVal%brFreq == 0;
    }

};

}
#endif
