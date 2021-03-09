#ifndef Glucose30_FeatureExtractor_h
#define Glucose30_FeatureExtractor_h

#include "glucose30/core/Solver.h"
#include "glucose30/core/SolverTypes.h"
#include "glucose30/mtl/Sort.h"
#include "common_types.h"

#include <assert.h>
#include <math.h>
#include <vector>
#include <unordered_map>
#include <algorithm>

#include <execinfo.h>
#include <stdlib.h>
#include <unistd.h>

namespace Glucose30 {

#define GSS_SIZE 201

class FeatureExtractor
{
private:
    static const int lbd_range  = 30;

    Solver           *solver   = NULL;
    ClauseAllocator  &ca;

    double                      trail_sz_sum;         // Trail Size
    Solver::MyQueue<double>     trail_sz_agg;         // Ditto but aggregated

    double                      decl_cnt_sum;         // Decision Level Count
    Solver::MyQueue<double>     decl_cnt_agg;         // Ditto but aggregated

    std::vector<double>          lbd_hist_recent;      // Un-normalized LBD Histogram (since last GC)
    Solver::MyQueue<std::vector<double>>
                                lbd_hist_q_recent;    // Queue of the LBD Histograms (since last GC)

    std::vector<double>          lbd_hist_total;       // Un-normalized LBD Histogram
    Solver::MyQueue<std::vector<double>>
                                lbd_hist_q_total;     // Queue of the LBD Histograms

public:
    FeatureExtractor(Solver *solver) :
        solver           (solver)
        , ca             (solver->ca)

        // GSS stats
        , trail_sz_sum       (0)
        , trail_sz_agg       (2, true)

        , decl_cnt_sum       (0)
        , decl_cnt_agg       (2, true)

        , lbd_hist_recent    (lbd_range)
        , lbd_hist_q_recent  (2, true)

        , lbd_hist_total     (lbd_range)
        , lbd_hist_q_total   (1, true)
    {}

    ~FeatureExtractor() {}

    // Creates a csr sparse matrix representing the Clause Variable Insident Graph (CVIG)
    // TODO: Convert the tuple to a struct or array of vector<unsigned>'s
    // CVIG clause2SPArray(bool learnts_info)
    // {
    //     vec<CRef> &cl_list = learnts_info ? solver->learnts : solver->clauses;
    //     int n_cl           = learnts_info ? solver->nLearnts() : solver->nClauses();
    //     // This is the sum of literals in all learnt (or problem) clauses
    //     int n_lits         = learnts_info ? solver->learnts_literals : solver->nLits();

    //     u_vec rows_arr(n_lits, 0);
    //     u_vec cols_arr(n_lits, 0);
    //     u_vec data_arr(n_lits, 0);

    //     int i, lits_count;
    //     for (i = lits_count = 0; i < n_cl; i++){
    //         Clause &c = ca[cl_list[i]];
    //         for (int j = 0; j < c.size(); j++) {
    //             assert(lits_count<n_lits);

    //             rows_arr[lits_count] = c.id();
    //             // We keep the variable convention of minisat in that
    //             // the variable indices start from 0 and not 1
    //             cols_arr[lits_count] = var (c[j]);
    //             data_arr[lits_count] = sign(c[j])?-1:1;

    //             lits_count++;
    //         }
    //     }

    //     return std::make_tuple(rows_arr, cols_arr, data_arr);
    // }

    CVIG cl_adj()
    {
        int edge_feat_size = 2;
        vec<CRef> &cl_list_lrnt = solver->learnts;
        vec<CRef> &cl_list_orig = solver->clauses;

        int n_cl_lrnt = solver->nLearnts();
        int n_cl_orig = solver->nClauses();

        // This is the sum of literals in all learnt (or problem) clauses
        int n_lits_lrnt = solver->learnts_literals;
        int n_lits_orig = solver->nLits();
        int n_lits = n_lits_lrnt + n_lits_orig;

        u_vec rows_arr(n_lits, 0);
        u_vec cols_arr(n_lits, 0);

        std::vector<u_vec> data_arr(n_lits, u_vec(edge_feat_size, 0));

        int i, lits_count;
        for (i = lits_count = 0; i < n_cl_orig; i++){
            Clause &c = ca[cl_list_orig[i]];
            for (int j = 0; j < c.size(); j++) {
                assert(lits_count<n_lits);

                rows_arr[lits_count] = i;
                cols_arr[lits_count] = toInt(c[j]);
                data_arr[lits_count][0] = c.id();
                data_arr[lits_count][1] = c(j).level;

                lits_count++;
            }
        }

        for (i = 0; i < n_cl_lrnt; i++){
            Clause &c = ca[cl_list_lrnt[i]];
            for (int j = 0; j < c.size(); j++) {
                assert(lits_count<n_lits);

                rows_arr[lits_count] = n_cl_orig + i;
                cols_arr[lits_count] = toInt(c[j]);
                data_arr[lits_count][0] = c.id();
                data_arr[lits_count][1] = c(j).level;

                lits_count++;
            }
        }

        return std::make_tuple(rows_arr, cols_arr, data_arr);
    }

    std::vector<std::vector<double>> getClaLabels(bool learnts_info)
    {
        vec<CRef> &cl_list = learnts_info ? solver->learnts : solver->clauses;
        int nClauses = learnts_info ? solver->nLearnts() : solver->nClauses();

        std::vector<double> row(cl_lable_size(), 0.0);
        std::vector<std::vector<double> > cl_label_arr(nClauses, row);

        int i, j;
        for (i = j = 0; i < nClauses; i++){
            Clause& c = ca[cl_list[i]];

            cl_label_arr[i][j++] = c.id(); // id
            cl_label_arr[i][j++] = c.num_used(); //num_used (make it temporal)
            cl_label_arr[i][j++] = c.size(); //size
            cl_label_arr[i][j++] = c.lbd(); //lbd
            cl_label_arr[i][j++] = c.activity(); //activity
            cl_label_arr[i][j++] = solver->locked(c); //is_locked
            cl_label_arr[i][j++] = learnts_info; // learnt?
            cl_label_arr[i][j++] = c.tagged(); // tagged?
            cl_label_arr[i][j++] = c.del(); // marked for delete?

            assert(("A clause here should either ba tagged or not marked for deletion!",
                !(!c.tagged() && !c.del())));

            j = 0;
        }

        return cl_label_arr;
    }

    std::vector<std::vector<double>> getVarLabels()
    {
        int nVars  = solver->nVars();
        int var_lable_size = 5;

        std::vector<double> row(var_lable_size, 0.0);
        std::vector<std::vector<double> > var_label_arr(nVars, row);

        // Populating the array
        int i, j;
        for (i = 0, j = 0; i < nVars; i++){
            var_label_arr[i][j++] = isWatch(i); // is there a watch literal form this var?
            var_label_arr[i][j++] = solver->level_avg(i); // average of decision levels at which the variable was assigned recently
            var_label_arr[i][j++] = solver->activity[i]; // activity (avergae for a clasue)
            var_label_arr[i][j++] = solver->polarity[i]; // polarity
            var_label_arr[i][j++] = solver->value(i) != g3l_Undef && solver->level(i) == 0; // is var forced?

            j = 0;
        }

        return var_label_arr;
    }

    std::vector<std::vector<double>> getLitLabels()
    {
        int nLits  = solver->nVars() * 2;
        int lit_lable_size = 5;

        std::vector<double> row(lit_lable_size, 0.0);
        std::vector<std::vector<double> > lit_label_arr(nLits, row);

        // Populating the array
        int i, j;
        for (i = 0, j = 0; i < nLits; i++){
            Lit lit_i = toLit(i);
            Var var_i = var(toLit(i));

            lit_label_arr[i][j++] = isWatch(lit_i); // Is this a watch literal?
            lit_label_arr[i][j++] = solver->level_avg(var_i); // average of decision levels at which the variable was assigned recently
            lit_label_arr[i][j++] = solver->activity[var_i]; // activity
            lit_label_arr[i][j++] = solver->polarity[var_i]; // polarity
            lit_label_arr[i][j++] = solver->value(var_i) != g3l_Undef && solver->level(var_i) == 0; // is var forced?

            j = 0;
        }

        return lit_label_arr;
    }

    std::unordered_map<std::string, std::vector<double>> getGSS()
    {
        double nClause_d = solver->nClauses();
        double nLearnts_d = solver->nLearnts();

        std::unordered_map<std::string, std::vector<double>> featMap;
        featMap["learnts_ratio"] = std::vector<double>(1, nLearnts_d  / (nClause_d + nLearnts_d));
        featMap["lbd_avg"]  = std::vector<double>(1, avgLBD());
        featMap["trail_size_avg"] = trail_sz_agg.vec();
        featMap["decision_level_avg"] = decl_cnt_agg.vec();

        unsigned int index = 0;
        for (auto &item : lbd_hist_q_recent.q)
            featMap["lbd_hist_recent_" + std::to_string(index++)] = item;

        index = 0;
        for (auto &item : lbd_hist_q_total.q)
            featMap["lbd_hist_total_" + std::to_string(index++)] = item;

        return featMap;
    }

    void saveGSS()
    {
        int partition = solver->nbclausesbeforereduce/2;
        trail_sz_sum += solver->trail.size();
        decl_cnt_sum += solver->decisionLevel();
        if (solver->conflicts % partition == 0) { // stats buffers are full, aggregate the buffers
            trail_sz_agg.push(trail_sz_sum / (solver->nFreeVars() * (double) partition)); // Aggregate
            decl_cnt_agg.push(decl_cnt_sum / (solver->nFreeVars() * (double) partition)); // Aggregate

            trail_sz_sum = 0;
            decl_cnt_sum = 0;
        }

        partition = solver->nbclausesbeforereduce/2;
        if (solver->conflicts % partition == 0) {
            // Push
            lbd_hist_q_recent.push(lbd_hist_recent);
            //  Reset
            std::fill(lbd_hist_recent.begin(), lbd_hist_recent.end(), 0.0);
        }

        partition = solver->nbclausesbeforereduce/1;
        if (solver->conflicts % partition == 0)
            lbd_hist_q_total.push(lbd_hist_total);
    }

    void updateLBDHist(int lbd, int update) {
        if (lbd <= 0 || lbd_range < lbd) return; // disregard LBD = 0

        lbd_hist_recent[lbd - 1] += std::max(update, 0);
        lbd_hist_total[lbd - 1] += update;

        if (lbd_hist_total[lbd - 1] < 0) {
            printf("Error: Sum of LBD's should be >= 0. (lbd_total[%i] = %f)\n",
                lbd, lbd_hist_total[lbd - 1]);
            void *array[10];
            size_t size;

            // get void*'s for all entries on the stack
            size = backtrace(array, 10);

            // print out all the frames to stderr
            backtrace_symbols_fd(array, size, STDERR_FILENO);
            exit(1);
        }
    }

    int cl_lable_size() {
        // id, num_used, size, lbd, activity, locked, learnt, tagged, deleted
        return 9;
    }

private:
    inline bool isWatch(Var v) {
        return isWatch(mkLit(v, false)) || isWatch(mkLit(v, true));
    }

    inline bool isWatch(Lit l) {
        return solver->watches[l].size() > 0;
    }

    double avgLBD() {
        double avgLBD = 0;
        int counter = 0;
        vec<CRef> &learnts = solver->learnts;

        for (int j = 0; j < learnts.size(); j++){
            Clause& c = ca[learnts[j]];
            avgLBD += c.lbd();
            counter++;
        }

        return avgLBD / counter;
    }
};

}
#endif
