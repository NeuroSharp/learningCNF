#ifndef SharpSAT_ORACLE_H_
#define SharpSAT_ORACLE_H_

#include "solver.h"
#include "../common_types.h" // TODO: Fix the top directory (should be SharpSAT)
#include "component_types/component.h"

#include <functional>
#include <vector>
#include <deque>

namespace SharpSAT {

class BranchingOracle {

public:
    BranchingOracle(std::function<void(CVIG)> capture_cb,
                    std::function<int*(CVIG)> branching_cb) :
        capture_cb(capture_cb),
        branching_cb(branching_cb)
    {}

    void capture(u_vec rows_arr, u_vec cols_arr, std::vector<u_vec> data_arr) {

        // int max_variable_id_ = literals.end_lit().var() - 1;
        // int max_clause_id_ = 0;

        // auto curr_cl_ofs = lit_pool.begin();
        // for (auto it_lit = lit_pool.begin(); it_lit < lit_pool.end(); it_lit++) {
        //     if (*it_lit == SENTINEL_LIT) {
        //         if (it_lit + 1 == lit_pool.end())
        //             break;

        //         max_clause_id_++;
        //         it_lit += ClauseHeader::overheadInLits();
        //         curr_cl_ofs = it_lit + 1;

        //     } else {
        //         assert(it_lit->var() <= max_variable_id_);
        //         assert(max_clause_id_ == getHeaderOf(curr_cl_ofs).id());

        //         rows_arr.push_back(max_clause_id_); // index of clause in CVIG
        //         cols_arr.push_back(*lt);
        //         data_arr.push_back(1);

        //     }
        // }

        BranchingOracle::capture_cb(std::make_tuple(rows_arr, cols_arr, data_arr));
    }

    int* decideLiteral(u_vec rows_arr, u_vec cols_arr, std::vector<u_vec> data_arr) {
        // // Estimate of the number of literals in the component (Each clause is connected to at least 3 lits)
        // int init_size = 3 * comp.numLongClauses();
        // int max_clause_id_ = 0;

        // u_vec rows_arr(init_size, 0);
        // u_vec cols_arr(init_size, 0);

        // for (auto it_cla = comp.clsBegin(); *it_cla != clsSENTINEL; it_cla++) {
        //     max_clause_id_++;

        //     for (auto lit_it = solver->beginOf(*it_cla); *lit_it != SENTINEL_LIT; lit_it++) {
        //         if (isActive(*lit_it)) {
        //             rows_arr.push_back(max_clause_id_); // index of clause in CVIG
        //             cols_arr.push_back(lit_it->raw());
        //         }
        //     }
        // }
        // assert(max_clause_id_ == clsBegin.numLongClauses());

        // // Begin adding binary clauses
        // for (auto var_it = comp.varsBegin(); *var_it != varsSENTINEL; var_it++) {
        //     // for both positive and negative literal of this var
        //     for (auto sign : {false, true}) {
        //         Literal l = solver->literal(LiteralID(*var_it, sign));
        //         if (l.hasBinaryLinks()) {
        //             for (auto lit_it = l.binary_links_.begin(); *lit_it != SENTINEL_LIT; lit_it++) {
        //                 assert(isActive(*lit_it));
        //                 max_clause_id_++;

        //                 rows_arr.push_back(max_clause_id_); // current lit
        //                 cols_arr.push_back(LiteralID(*var_it, sign).raw());

        //                 rows_arr.push_back(max_clause_id_); // its binary link
        //                 cols_arr.push_back(lit_it->raw());
        //             }
        //         }
        //     }
        // }

        // u_vec data_arr(rows_arr.size(), 1);

        return BranchingOracle::branching_cb(std::make_tuple(rows_arr, cols_arr, data_arr));

    }

    // CVIG cl_adj(Component &comp)
    // {
    //     int nLongCla = comp->numLongClauses();
    //     int nVars = comp->num_variables();

    //     u_vec rows_arr(nVars*2, 0);
    //     u_vec cols_arr(nVars*2, 0);
    //     u_vec data_arr(nVars*2, 0);

    //     int i = 0;
    //     for (auto itCl = comp.clsBegin(); *itCl != clsSENTINEL; itCl++){
    //         for (auto lt = beginOf(*itCl); *lt != SENTINEL_LIT; lt++)
    //             if (isActive(*lt)) {
    //                 rows_arr.push_back(i++); // index of clause in CVIG
    //                 cols_arr.push_back(*lt);
    //                 data_arr.push_back(c.id());
    //             }
    //     }

    //     // TODO: Add binary clauses
    //     for (auto it = comp.varsBegin(); *it != varsSENTINEL; it++) {
    //         literal(LiteralID(*it, true)).activity_score_ *= 0.5;
    //         literal(LiteralID(*it, false)).activity_score_ *= 0.5;
    //     }

    //     return std::make_tuple(rows_arr, cols_arr, data_arr);
    // }

private:
    std::function<void(CVIG)> capture_cb;
    std::function<int*(CVIG)> branching_cb;
};
}

#endif
