/*
 * solver.cpp
 *
 *  Created on: Aug 23, 2012
 *      Author: marc
 */
#include "solver.h"
#include <deque>
#include <set>

#include <algorithm>

using namespace SharpSAT;

StopWatch::StopWatch() {
    interval_length_.tv_sec = 60;
    gettimeofday(&last_interval_start_, NULL);
    start_time_ = stop_time_ = last_interval_start_;
}

timeval StopWatch::getElapsedTime() {
    timeval r;
    timeval other_time = stop_time_;
    if (stop_time_.tv_sec == start_time_.tv_sec
            && stop_time_.tv_usec == start_time_.tv_usec)
        gettimeofday(&other_time, NULL);
    long int ad = 0;
    long int bd = 0;

    if (other_time.tv_usec < start_time_.tv_usec) {
        ad = 1;
        bd = 1000000;
    }
    r.tv_sec = other_time.tv_sec - ad - start_time_.tv_sec;
    r.tv_usec = other_time.tv_usec + bd - start_time_.tv_usec;
    return r;
}



void Solver::print(std::vector<LiteralID> &vec) {
    for (auto l : vec)
        std::cout << l.toInt() << " ";
    std::cout << std::endl;
}

void Solver::print(std::vector<unsigned> &vec) {
    for (auto l : vec)
        std::cout << l << " ";
    std::cout << std::endl;
}

bool Solver::simplePreProcess() {

    if (!config_.perform_pre_processing)
        return true;
    assert(literal_stack_.size() == 0);
    unsigned start_ofs = 0;
//BEGIN process unit clauses
    for (auto lit : unit_clauses_)
        setLiteralIfFree(lit);
//END process unit clauses
    bool succeeded = BCP(start_ofs);

    if (succeeded)
        succeeded &= prepFailedLiteralTest();

    if (succeeded) {
        HardWireAndCompact();
        incidentGraph();
    }
    return succeeded;
}

bool Solver::prepFailedLiteralTest() {
    unsigned last_size;
    do {
        last_size = literal_stack_.size();
        for (unsigned v = 1; v < variables_.size(); v++)
            if (isActive(v)) {
                unsigned sz = literal_stack_.size();
                setLiteralIfFree(LiteralID(v, true));
                bool res = BCP(sz);
                while (literal_stack_.size() > sz) {
                    unSet(literal_stack_.back());
                    literal_stack_.pop_back();
                }

                if (!res) {
                    sz = literal_stack_.size();
                    setLiteralIfFree(LiteralID(v, false));
                    if (!BCP(sz))
                        return false;
                } else {

                    sz = literal_stack_.size();
                    setLiteralIfFree(LiteralID(v, false));
                    bool resb = BCP(sz);
                    while (literal_stack_.size() > sz) {
                        unSet(literal_stack_.back());
                        literal_stack_.pop_back();
                    }
                    if (!resb) {
                        sz = literal_stack_.size();
                        setLiteralIfFree(LiteralID(v, true));
                        if (!BCP(sz))
                            return false;
                    }
                }
            }
    } while (literal_stack_.size() > last_size);

    return true;
}

void Solver::incidentGraph() {
    unsigned max_clause_id_ = 0;
    std::set<unsigned> processed_cls;

    u_vec rows_arr;
    u_vec cols_arr;
    for (auto currentLit = LiteralID(1, false); currentLit != literals_.end_lit(); currentLit.inc()) {
        // Begin adding long clauses

        for (auto cl_ofs : occurrence_lists_[currentLit]) {
            unsigned cl_id = getHeaderOf(cl_ofs).id();
            if (0 < processed_cls.count(cl_id)) continue; // This clause has been processed before

            bool inc_clause_id = false;
            // Include the active literals of this clause that appear in the component
            for (auto lit_it = beginOf(cl_ofs); *lit_it != SENTINEL_LIT; lit_it++)
                if (isActive(*lit_it)) {
                    rows_arr.push_back(max_clause_id_);
                    cols_arr.push_back(lookupDiamacsLitID(*lit_it).toInt());

                    inc_clause_id = true;
                }

            if (inc_clause_id) max_clause_id_++;
            processed_cls.insert(cl_id);
        }

        if (!literal(currentLit).hasBinaryLinks()) continue;

        // Begin adding binary clauses
        for (auto lit_it = literal(currentLit).binary_links_.begin(); *lit_it != SENTINEL_LIT; lit_it++) {
            // A trick to avoid counting binary clauses twice (once for each literal)
            if (currentLit < *lit_it) continue;

            if (isActive(*lit_it)) {
                rows_arr.push_back(max_clause_id_); // current lit
                cols_arr.push_back(lookupDiamacsLitID(currentLit).toInt());

                rows_arr.push_back(max_clause_id_); // its binary link
                cols_arr.push_back(lookupDiamacsLitID(*lit_it).toInt());

                max_clause_id_++;

                if (currentLit.toIndex() < 0 || lit_it->toIndex() < 0)
                    printf("Wrong Lit in Binary Clause!\n");
            }
        }
    }

    std::vector<u_vec> data_arr(rows_arr.size(), u_vec(1, 0));

    oracle_->capture(rows_arr, cols_arr, data_arr);
}

void Solver::HardWireAndCompact() {
    compactClauses();
    compactVariables();
    for (auto lit : literal_stack_)
        level_zero_vars.push_back(lit.toInt());

    literal_stack_.clear();

    for (auto l = LiteralID(1, false); l != literals_.end_lit(); l.inc()) {
        literal(l).activity_score_ = literal(l).binary_links_.size() - 1;
        literal(l).activity_score_ += occurrence_lists_[l].size();
    }

    statistics_.num_unit_clauses_ = unit_clauses_.size();

    statistics_.num_original_binary_clauses_ = statistics_.num_binary_clauses_;
    statistics_.num_original_unit_clauses_ = statistics_.num_unit_clauses_ =
                unit_clauses_.size();
    initStack(num_variables());
    original_lit_pool_size_ = literal_pool_.size();
}

void Solver::solve(const std::string &file_name) {
    stopwatch_.start();
    statistics_.input_file_ = file_name;

    if (!config_.quiet)
        std::cout << "Reading file...\t" << std::endl;

    createfromFile(file_name);

    if (!config_.quiet)
        std::cout << "Done!\t" << std::endl;

    initStack(num_variables());

    if (!config_.quiet) {
        std::cout << "Solving " << file_name << std::endl;
        statistics_.printShortFormulaInfo();
    }
    if (!config_.quiet)
        std::cout << std::endl << "Preprocessing .." << std::flush;
    bool notfoundUNSAT = simplePreProcess();
    if (!config_.quiet)
        std::cout << " DONE" << std::endl;

    if (notfoundUNSAT) {

        if (!config_.quiet) {
            statistics_.printShortFormulaInfo();
        }

        last_ccl_deletion_time_ = last_ccl_cleanup_time_ =
                                      statistics_.getTime();

        violated_clause.reserve(num_variables());

        comp_manager_.initialize(literals_, literal_pool_);

        statistics_.exit_state_ = countSAT();

        statistics_.set_final_solution_count(stack_.top().getTotalModelCount());
        statistics_.num_long_conflict_clauses_ = num_conflict_clauses();

    } else {
        statistics_.exit_state_ = SUCCESS;
        statistics_.set_final_solution_count(0.0);
        std::cout << std::endl << " FOUND UNSAT DURING PREPROCESSING " << std::endl;
    }

    stopwatch_.stop();
    statistics_.time_elapsed_ = stopwatch_.getElapsedSeconds();

    comp_manager_.gatherStatistics();
    statistics_.writeToFile("data.out");
    if (!config_.quiet)
        statistics_.printShort();
}

SOLVER_StateT Solver::countSAT() {
    retStateT state = RESOLVED;

    while (true) {
        while (comp_manager_.findNextRemainingComponentOf(stack_.top())) {
            decideLiteral();
            if (stopwatch_.timeBoundBroken() || asynch_interrupt)
                return TIMEOUT;
            if (stopwatch_.interval_tick())
                printOnlineStats();

            while (!bcp()) {
                state = resolveConflict();
                if (state == BACKTRACK)
                    break;
            }
            if (state == BACKTRACK)
                break;
        }

        state = backtrack();
        if (state == EXIT)
            return SUCCESS;
        while (state != PROCESS_COMPONENT && !bcp()) {
            state = resolveConflict();
            if (state == BACKTRACK) {
                state = backtrack();
                if (state == EXIT)
                    return SUCCESS;
            }
        }
    }
    return SUCCESS;
}

std::vector<int> Solver::getLitStack() {
    std::vector<int> lit_stack;
    for (auto lit : literal_stack_)
        lit_stack.push_back(lookupDiamacsLitID(lit).toInt());

    return lit_stack;
}

std::vector<int> Solver::getBranchingSequence() {
    return branching_sequence;
}

std::vector<int> Solver::getLevelZeroVars() {
    return level_zero_vars;
}

std::vector<std::vector<double>> Solver::getLitLabels() {
    // Candidates to add:
    //      literal(l).binary_links_.size()
    //      occurrence_lists_[l].size()
    int lit_lable_size = 4; // 2 features plus 2 indexes (id and dimacs id)
    std::vector<double> row(lit_lable_size, 0.0);
    std::vector<std::vector<double>> lit_labels(statistics_.num_variables_ * 2, row);

    for (unsigned v = 1; v < variables_.size(); v++) {
        float var_score = scoreOf(v);
        // for both positive and negative literal of this var
        for (auto sign : {false, true}) {
            LiteralID lit_id = LiteralID(v, sign);
            Literal lit = literal(lit_id);
            std::vector<double> row(lit_lable_size, 0.0);
            row[0] = lit_id.toIndex();
            row[1] = lookupDiamacsLitID(lit_id).toInt();
            row[2] = lit.activity_score_;
            row[3] = var_score;
            lit_labels[lit_id.toIndex()] = row;
        }
    }


    // Component &comp = comp_manager_.superComponentOf(stack_.top());
    // for (auto var_it = comp.varsBegin(); *var_it != varsSENTINEL; var_it++) {
    //     float var_score = scoreOf(*var_it);
    //     // for both positive and negative literal of this var
    //     for (auto sign : {false, true}) {
    //         Literal lit = literal(LiteralID(*var_it, sign));
    //         std::vector<double> row(lit_lable_size, 0.0);
    //         row[0] = lit.activity_score_;
    //         row[1] = var_score;
    //         lit_labels[LiteralID(*var_it, sign).toIndex()] = row;
    //     }
    // }

    return lit_labels;
}

LiteralID Solver::branch(Component &comp) {
    LiteralID ret = NOT_A_LIT;
    unsigned max_clause_id_ = 0;

    u_vec rows_arr;
    u_vec cols_arr;

    std::set<unsigned> processed_cls;
    std::set<unsigned> vars_in_comp;

    for (auto var_it = comp.varsBegin(); *var_it != varsSENTINEL; var_it++)
        vars_in_comp.insert(*var_it);

    for (auto var_it = comp.varsBegin(); *var_it != varsSENTINEL; var_it++) {
        // for both positive and negative literal of this var
        for (auto sign : {false, true}) {
            LiteralID currentLit = LiteralID(*var_it, sign);

            // Begin adding long clauses
            for (auto cl_ofs : occurrence_lists_[currentLit]) {
                unsigned cl_id = getHeaderOf(cl_ofs).id();
                if (0 < processed_cls.count(cl_id)) continue; // This clause has been processed before

                bool inc_clause_id = false;
                // Include the active literals of this clause that appear in the component
                for (auto lit_it = beginOf(cl_ofs); *lit_it != SENTINEL_LIT; lit_it++)
                    if (isActive(*lit_it) && 0 < vars_in_comp.count(lit_it->var())) {
                        rows_arr.push_back(max_clause_id_);
                        cols_arr.push_back(lit_it->toIndex());

                        inc_clause_id = true;
                    }

                if (inc_clause_id) max_clause_id_++;
                processed_cls.insert(cl_id);
            }

            if (!literal(currentLit).hasBinaryLinks()) continue;

            // Begin adding binary clauses
            for (auto lit_it = literal(currentLit).binary_links_.begin(); *lit_it != SENTINEL_LIT; lit_it++) {
                // A trick to avoid counting binary clauses twice (once for each literal)
                if (currentLit < *lit_it) continue;

                if (isActive(*lit_it) && 0 < vars_in_comp.count(lit_it->var())) {
                    rows_arr.push_back(max_clause_id_); // current lit
                    cols_arr.push_back(currentLit.toIndex());

                    rows_arr.push_back(max_clause_id_); // its binary link
                    cols_arr.push_back(lit_it->toIndex());

                    max_clause_id_++;

                    if (currentLit.toIndex() < 0 || lit_it->toIndex() < 0)
                        printf("Wrong Lit in Binary Clause!\n");
                }
            }
        }
    }

    if (!config_.quiet) {
        std::cout << "component size (var/cla_bin/cla_long/cla_total) \t";
        std::cout << comp.num_variables() << "/" <<
            max_clause_id_ - comp.numLongClauses() << "/" <<
            comp.numLongClauses() << "/" <<
            max_clause_id_ << std::endl;
    }

    std::vector<u_vec> data_arr(rows_arr.size(), u_vec(1, 0));

    int lit_ind = *(oracle_->decideLiteral(rows_arr, cols_arr, data_arr));
    if (0 <= lit_ind)
        ret.copyRaw(lit_ind + 2);


    return ret;
}

void Solver::decideLiteral() {
    // establish another decision stack level
    stack_.push_back(
        StackLevel(stack_.top().currentRemainingComponent(),
                   literal_stack_.size(),
                   comp_manager_.component_stack_size()));


    LiteralID theLit = branch(comp_manager_.superComponentOf(stack_.top()));
    if (asynch_interrupt) return;

    if (theLit == NOT_A_LIT) { // Default Branching

        float max_score = -1;
        float score;
        unsigned max_score_var = 0;
        for (auto it =
                    comp_manager_.superComponentOf(stack_.top()).varsBegin();
                *it != varsSENTINEL; it++) {
            score = scoreOf(*it);
            if (score > max_score) {
                max_score = score;
                max_score_var = *it;
            }
        }
        // this assert should always hold,
        // if not then there is a bug in the logic of countSAT();
        assert(max_score_var != 0);

        theLit = LiteralID(max_score_var,
                     literal(LiteralID(max_score_var, true)).activity_score_
                     > literal(LiteralID(max_score_var, false)).activity_score_);
    }

    setLiteralIfFree(theLit);
    statistics_.num_decisions_++;
    branching_sequence.push_back(lookupDiamacsLitID(theLit).toInt());

    if (statistics_.num_decisions_ % 128 == 0)
//    if (statistics_.num_conflicts_ % 128 == 0)
        decayActivities();
    // decayActivitiesOf(comp_manager_.superComponentOf(stack_.top()));
    assert(
        stack_.top().remaining_components_ofs() <= comp_manager_.component_stack_size());
}

retStateT Solver::backtrack() {
    assert(
        stack_.top().remaining_components_ofs() <= comp_manager_.component_stack_size());
    do {
        if (stack_.top().branch_found_unsat())
            comp_manager_.removeAllCachePollutionsOf(stack_.top());
        else if (stack_.top().anotherCompProcessible())
            return PROCESS_COMPONENT;

        if (!stack_.top().isSecondBranch()) {
            LiteralID aLit = TOS_decLit();
            assert(stack_.get_decision_level() > 0);
            stack_.top().changeBranch();
            reactivateTOS();
            setLiteralIfFree(aLit.neg(), NOT_A_CLAUSE);
            return RESOLVED;
        }
        // OTHERWISE:  backtrack further
        comp_manager_.cacheModelCountOf(stack_.top().super_component(),
                                        stack_.top().getTotalModelCount());

        if (stack_.get_decision_level() <= 0)
            break;
        reactivateTOS();

        assert(stack_.size() >= 2);
        (stack_.end() - 2)->includeSolution(stack_.top().getTotalModelCount());
        stack_.pop_back();
        // step to the next component not yet processed
        stack_.top().nextUnprocessedComponent();

        assert(
            stack_.top().remaining_components_ofs() < comp_manager_.component_stack_size() + 1);

    } while (stack_.get_decision_level() >= 0);
    return EXIT;
}

retStateT Solver::resolveConflict() {
    recordLastUIPCauses();

    if (statistics_.num_clauses_learned_ - last_ccl_deletion_time_
            > statistics_.clause_deletion_interval()) {
        deleteConflictClauses();
        last_ccl_deletion_time_ = statistics_.num_clauses_learned_;
    }

    if (statistics_.num_clauses_learned_ - last_ccl_cleanup_time_ > 100000) {
        compactConflictLiteralPool();
        last_ccl_cleanup_time_ = statistics_.num_clauses_learned_;
    }

    statistics_.num_conflicts_++;

    assert(
        stack_.top().remaining_components_ofs() <= comp_manager_.component_stack_size());

    assert(uip_clauses_.size() == 1);

    // DEBUG
    if (uip_clauses_.back().size() == 0)
        std::cout << " EMPTY CLAUSE FOUND" << std::endl;
    // END DEBUG

    stack_.top().mark_branch_unsat();
    //BEGIN Backtracking
    // maybe the other branch had some solutions
    if (stack_.top().isSecondBranch()) {
        return BACKTRACK;
    }

    Antecedent ant(NOT_A_CLAUSE);
    // this has to be checked since using implicit BCP
    // and checking literals there not exhaustively
    // we cannot guarantee that uip_clauses_.back().front() == TOS_decLit().neg()
    // this is because we might have checked a literal
    // during implict BCP which has been a failed literal
    // due only to assignments made at lower decision levels
    if (uip_clauses_.back().front() == TOS_decLit().neg()) {
        assert(TOS_decLit().neg() == uip_clauses_.back()[0]);
        var(TOS_decLit().neg()).ante = addUIPConflictClause(
                                           uip_clauses_.back());
        ant = var(TOS_decLit()).ante;
    }
//  // RRR
//  else if(var(uip_clauses_.back().front()).decision_level
//          < stack_.get_decision_level()
//          && assertion_level_ <  stack_.get_decision_level()){
//         stack_.top().set_both_branches_unsat();
//         return BACKTRACK;
//  }
//
//
//  // RRR
    assert(stack_.get_decision_level() > 0);
    assert(stack_.top().branch_found_unsat());

    // we do not have to remove pollutions here,
    // since conflicts only arise directly before
    // remaining components are stored
    // hence
    assert(
        stack_.top().remaining_components_ofs() == comp_manager_.component_stack_size());

    stack_.top().changeBranch();
    LiteralID lit = TOS_decLit();
    reactivateTOS();
    setLiteralIfFree(lit.neg(), ant);
//END Backtracking
    return RESOLVED;
}

bool Solver::bcp() {
// the asserted literal has been set, so we start
// bcp on that literal
    unsigned start_ofs = literal_stack_.size() - 1;

//BEGIN process unit clauses
    for (auto lit : unit_clauses_)
        setLiteralIfFree(lit);
//END process unit clauses

    bool bSucceeded = BCP(start_ofs);

    if (config_.perform_failed_lit_test && bSucceeded) {
        bSucceeded = implicitBCP();
    }
    return bSucceeded;
}

bool Solver::BCP(unsigned start_at_stack_ofs) {
    for (unsigned int i = start_at_stack_ofs; i < literal_stack_.size(); i++) {
        LiteralID unLit = literal_stack_[i].neg();
        //BEGIN Propagate Bin Clauses
        for (auto bt = literal(unLit).binary_links_.begin();
                *bt != SENTINEL_LIT; bt++) {
            // if (unLit.toInt() == -84) printf("[%i, -84]\n", (*bt).toInt());
            if (isResolved(*bt)) {
                setConflictState(unLit, *bt);
                return false;
            }
            setLiteralIfFree(*bt, Antecedent(unLit));
        }
        //END Propagate Bin Clauses
        for (auto itcl = literal(unLit).watch_list_.rbegin();
                *itcl != SENTINEL_CL; itcl++) {
            bool isLitA = (*beginOf(*itcl) == unLit);
            auto p_watchLit = beginOf(*itcl) + 1 - isLitA;
            auto p_otherLit = beginOf(*itcl) + isLitA;

            if (isSatisfied(*p_otherLit))
                continue;
            auto itL = beginOf(*itcl) + 2;
            while (isResolved(*itL))
                itL++;
            // either we found a free or satisfied lit
            if (*itL != SENTINEL_LIT) {
                literal(*itL).addWatchLinkTo(*itcl);
                std::swap(*itL, *p_watchLit);
                *itcl = literal(unLit).watch_list_.back();
                literal(unLit).watch_list_.pop_back();
            } else {
                // or p_unLit stays resolved
                // and we have hence no free literal left
                // for p_otherLit remain poss: Active or Resolved
                if (setLiteralIfFree(*p_otherLit, Antecedent(*itcl))) { // implication
                    if (isLitA)
                        std::swap(*p_otherLit, *p_watchLit);
                } else {
                    setConflictState(*itcl);
                    return false;
                }
            }
        }
    }
    return true;
}

//bool Solver::implicitBCP() {
//  static std::vector<LiteralID> test_lits(num_variables());
//  static LiteralIndexedVector<unsigned char> viewed_lits(num_variables() + 1,
//      0);
//
//  unsigned stack_ofs = stack_.top().literal_stack_ofs();
//  while (stack_ofs < literal_stack_.size()) {
//    test_lits.clear();
//    for (auto it = literal_stack_.begin() + stack_ofs;
//        it != literal_stack_.end(); it++) {
//      for (auto cl_ofs : occurrence_lists_[it->neg()])
//        if (!isSatisfied(cl_ofs)) {
//          for (auto lt = beginOf(cl_ofs); *lt != SENTINEL_LIT; lt++)
//            if (isActive(*lt) && !viewed_lits[lt->neg()]) {
//              test_lits.push_back(lt->neg());
//              viewed_lits[lt->neg()] = true;
//
//            }
//        }
//    }
//
//    stack_ofs = literal_stack_.size();
//    for (auto jt = test_lits.begin(); jt != test_lits.end(); jt++)
//      viewed_lits[*jt] = false;
//
//    statistics_.num_failed_literal_tests_ += test_lits.size();
//
//    for (auto lit : test_lits)
//      if (isActive(lit)) {
//        unsigned sz = literal_stack_.size();
//        // we increase the decLev artificially
//        // s.t. after the tentative BCP call, we can learn a conflict clause
//        // relative to the assignment of *jt
//        stack_.startFailedLitTest();
//        setLiteralIfFree(lit);
//
//        assert(!hasAntecedent(lit));
//
//        bool bSucceeded = BCP(sz);
//        if (!bSucceeded)
//          recordAllUIPCauses();
//
//        stack_.stopFailedLitTest();
//
//        while (literal_stack_.size() > sz) {
//          unSet(literal_stack_.back());
//          literal_stack_.pop_back();
//        }
//
//        if (!bSucceeded) {
//          statistics_.num_failed_literals_detected_++;
//          sz = literal_stack_.size();
//          for (auto it = uip_clauses_.rbegin(); it != uip_clauses_.rend();
//              it++) {
//            setLiteralIfFree(it->front(), addUIPConflictClause(*it));
//          }
//          if (!BCP(sz))
//            return false;
//        }
//      }
//  }
//  return true;
//}

// this is IBCP 30.08
bool Solver::implicitBCP() {
    static std::vector<LiteralID> test_lits(num_variables());
    static LiteralIndexedVector<unsigned char> viewed_lits(num_variables() + 1,
            0);

    unsigned stack_ofs = stack_.top().literal_stack_ofs();
    unsigned num_curr_lits = 0;
    while (stack_ofs < literal_stack_.size()) {
        test_lits.clear();
        for (auto it = literal_stack_.begin() + stack_ofs;
                it != literal_stack_.end(); it++) {
            for (auto cl_ofs : occurrence_lists_[it->neg()])
                if (!isSatisfied(cl_ofs)) {
                    for (auto lt = beginOf(cl_ofs); *lt != SENTINEL_LIT; lt++)
                        if (isActive(*lt) && !viewed_lits[lt->neg()]) {
                            test_lits.push_back(lt->neg());
                            viewed_lits[lt->neg()] = true;

                        }
                }
        }
        num_curr_lits = literal_stack_.size() - stack_ofs;
        stack_ofs = literal_stack_.size();
        for (auto jt = test_lits.begin(); jt != test_lits.end(); jt++)
            viewed_lits[*jt] = false;

        std::vector<float> scores;
        scores.clear();
        for (auto jt = test_lits.begin(); jt != test_lits.end(); jt++) {
            scores.push_back(literal(*jt).activity_score_);
        }
        sort(scores.begin(), scores.end());
        num_curr_lits = 10 + num_curr_lits / 20;
        float threshold = 0.0;
        if (scores.size() > num_curr_lits) {
            threshold = scores[scores.size() - num_curr_lits];
        }

        statistics_.num_failed_literal_tests_ += test_lits.size();

        for (auto lit : test_lits)
            if (isActive(lit) && threshold <= literal(lit).activity_score_) {
                unsigned sz = literal_stack_.size();
                // we increase the decLev artificially
                // s.t. after the tentative BCP call, we can learn a conflict clause
                // relative to the assignment of *jt
                stack_.startFailedLitTest();
                setLiteralIfFree(lit);

                assert(!hasAntecedent(lit));

                bool bSucceeded = BCP(sz);
                if (!bSucceeded)
                    recordAllUIPCauses();

                stack_.stopFailedLitTest();

                while (literal_stack_.size() > sz) {
                    unSet(literal_stack_.back());
                    literal_stack_.pop_back();
                }

                if (!bSucceeded) {
                    statistics_.num_failed_literals_detected_++;
                    sz = literal_stack_.size();
                    for (auto it = uip_clauses_.rbegin();
                            it != uip_clauses_.rend(); it++) {
                        // DEBUG
                        if (it->size() == 0)
                            std::cout << "EMPTY CLAUSE FOUND" << std::endl;
                        // END DEBUG
                        setLiteralIfFree(it->front(),
                                         addUIPConflictClause(*it));
                    }
                    if (!BCP(sz))
                        return false;
                }
            }
    }

    // BEGIN TEST
//  float max_score = -1;
//  float score;
//  unsigned max_score_var = 0;
//  for (auto it =
//          component_analyzer_.superComponentOf(stack_.top()).varsBegin();
//          *it != varsSENTINEL; it++)
//      if (isActive(*it)) {
//          score = scoreOf(*it);
//          if (score > max_score) {
//              max_score = score;
//              max_score_var = *it;
//          }
//      }
//  LiteralID theLit(max_score_var,
//          literal(LiteralID(max_score_var, true)).activity_score_
//                  > literal(LiteralID(max_score_var, false)).activity_score_);
//  if (!fail_test(theLit.neg())) {
//      std::cout << ".";
//
//      statistics_.num_failed_literals_detected_++;
//      unsigned sz = literal_stack_.size();
//      for (auto it = uip_clauses_.rbegin(); it != uip_clauses_.rend(); it++) {
//          setLiteralIfFree(it->front(), addUIPConflictClause(*it));
//      }
//      if (!BCP(sz))
//          return false;
//
//  }
    // END
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////
// BEGIN module conflictAnalyzer
///////////////////////////////////////////////////////////////////////////////////////////////

void Solver::minimizeAndStoreUIPClause(LiteralID uipLit,
                                       std::vector<LiteralID> & tmp_clause, bool seen[]) {
    static std::deque<LiteralID> clause;
    clause.clear();
    assertion_level_ = 0;
    for (auto lit : tmp_clause) {
        if (existsUnitClauseOf(lit.var()))
            continue;
        bool resolve_out = false;
        if (hasAntecedent(lit)) {
            resolve_out = true;
            if (getAntecedent(lit).isAClause()) {
                for (auto it = beginOf(getAntecedent(lit).asCl()) + 1;
                        *it != SENTINEL_CL; it++)
                    if (!seen[it->var()]) {
                        resolve_out = false;
                        break;
                    }
            } else if (!seen[getAntecedent(lit).asLit().var()]) {
                resolve_out = false;
            }
        }

        if (!resolve_out) {
            // uipLit should be the sole literal of this Decision Level
            if (var(lit).decision_level >= assertion_level_) {
                assertion_level_ = var(lit).decision_level;
                clause.push_front(lit);
            } else
                clause.push_back(lit);
        }
    }

    if (uipLit.var())
        assert(var(uipLit).decision_level == stack_.get_decision_level());

    //assert(uipLit.var() != 0);
    if (uipLit.var() != 0)
        clause.push_front(uipLit);
    uip_clauses_.push_back(std::vector<LiteralID>(clause.begin(), clause.end()));
}

void Solver::recordLastUIPCauses() {
// note:
// variables of lower dl: if seen we dont work with them anymore
// variables of this dl: if seen we incorporate their
// antecedent and set to unseen
    bool seen[num_variables() + 1];
    memset(seen, false, sizeof(bool) * (num_variables() + 1));

    static std::vector<LiteralID> tmp_clause;
    tmp_clause.clear();

    assertion_level_ = 0;
    uip_clauses_.clear();

    unsigned lit_stack_ofs = literal_stack_.size();
    int DL = stack_.get_decision_level();
    unsigned lits_at_current_dl = 0;

    for (auto l : violated_clause) {
        if (var(l).decision_level == 0 || existsUnitClauseOf(l.var()))
            continue;
        if (var(l).decision_level < DL)
            tmp_clause.push_back(l);
        else
            lits_at_current_dl++;
        literal(l).increaseActivity();
        seen[l.var()] = true;
    }

    LiteralID curr_lit;
    while (lits_at_current_dl) {
        assert(lit_stack_ofs != 0);
        curr_lit = literal_stack_[--lit_stack_ofs];

        if (!seen[curr_lit.var()])
            continue;

        seen[curr_lit.var()] = false;

        if (lits_at_current_dl-- == 1) {
            // perform UIP stuff
            if (!hasAntecedent(curr_lit)) {
                // this should be the decision literal when in first branch
                // or it is a literal decided to explore in failed literal testing
                //assert(stack_.TOS_decLit() == curr_lit);
//              std::cout << "R" << curr_lit.toInt() << "S"
//                   << var(curr_lit).ante.isAnt() << " "  << std::endl;
                break;
            }
        }

        assert(hasAntecedent(curr_lit));

        //std::cout << "{" << curr_lit.toInt() << "}";
        if (getAntecedent(curr_lit).isAClause()) {
            updateActivities(getAntecedent(curr_lit).asCl());
            assert(curr_lit == *beginOf(getAntecedent(curr_lit).asCl()));

            for (auto it = beginOf(getAntecedent(curr_lit).asCl()) + 1;
                    *it != SENTINEL_CL; it++) {
                if (seen[it->var()] || (var(*it).decision_level == 0)
                        || existsUnitClauseOf(it->var()))
                    continue;
                if (var(*it).decision_level < DL)
                    tmp_clause.push_back(*it);
                else
                    lits_at_current_dl++;
                seen[it->var()] = true;
            }
        } else {
            LiteralID alit = getAntecedent(curr_lit).asLit();
            literal(alit).increaseActivity();
            literal(curr_lit).increaseActivity();
            if (!seen[alit.var()] && !(var(alit).decision_level == 0)
                    && !existsUnitClauseOf(alit.var())) {
                if (var(alit).decision_level < DL)
                    tmp_clause.push_back(alit);
                else
                    lits_at_current_dl++;
                seen[alit.var()] = true;
            }
        }
        curr_lit = NOT_A_LIT;
    }

//  std::cout << "T" << curr_lit.toInt() << "U "
//     << var(curr_lit).decision_level << ", " << stack_.get_decision_level() << std::endl;
//  std::cout << "V"  << var(curr_lit).ante.isAnt() << " "  << std::endl;
    minimizeAndStoreUIPClause(curr_lit.neg(), tmp_clause, seen);

//  if (var(curr_lit).decision_level > assertion_level_)
//      assertion_level_ = var(curr_lit).decision_level;
}

void Solver::recordAllUIPCauses() {
// note:
// variables of lower dl: if seen we dont work with them anymore
// variables of this dl: if seen we incorporate their
// antecedent and set to unseen
    bool seen[num_variables() + 1];
    memset(seen, false, sizeof(bool) * (num_variables() + 1));

    static std::vector<LiteralID> tmp_clause;
    tmp_clause.clear();

    assertion_level_ = 0;
    uip_clauses_.clear();

    unsigned lit_stack_ofs = literal_stack_.size();
    int DL = stack_.get_decision_level();
    unsigned lits_at_current_dl = 0;

    for (auto l : violated_clause) {
        if (var(l).decision_level == 0 || existsUnitClauseOf(l.var()))
            continue;
        if (var(l).decision_level < DL)
            tmp_clause.push_back(l);
        else
            lits_at_current_dl++;
        literal(l).increaseActivity();
        seen[l.var()] = true;
    }
    unsigned n = 0;
    LiteralID curr_lit;
    while (lits_at_current_dl) {
        assert(lit_stack_ofs != 0);
        curr_lit = literal_stack_[--lit_stack_ofs];

        if (!seen[curr_lit.var()])
            continue;

        seen[curr_lit.var()] = false;

        if (lits_at_current_dl-- == 1) {
            n++;
            if (!hasAntecedent(curr_lit)) {
                // this should be the decision literal when in first branch
                // or it is a literal decided to explore in failed literal testing
                //assert(stack_.TOS_decLit() == curr_lit);
                break;
            }
            // perform UIP stuff
            minimizeAndStoreUIPClause(curr_lit.neg(), tmp_clause, seen);
        }

        assert(hasAntecedent(curr_lit));

        if (getAntecedent(curr_lit).isAClause()) {
            updateActivities(getAntecedent(curr_lit).asCl());
            assert(curr_lit == *beginOf(getAntecedent(curr_lit).asCl()));

            for (auto it = beginOf(getAntecedent(curr_lit).asCl()) + 1;
                    *it != SENTINEL_CL; it++) {
                if (seen[it->var()] || (var(*it).decision_level == 0)
                        || existsUnitClauseOf(it->var()))
                    continue;
                if (var(*it).decision_level < DL)
                    tmp_clause.push_back(*it);
                else
                    lits_at_current_dl++;
                seen[it->var()] = true;
            }
        } else {
            LiteralID alit = getAntecedent(curr_lit).asLit();
            literal(alit).increaseActivity();
            literal(curr_lit).increaseActivity();
            if (!seen[alit.var()] && !(var(alit).decision_level == 0)
                    && !existsUnitClauseOf(alit.var())) {
                if (var(alit).decision_level < DL)
                    tmp_clause.push_back(alit);
                else
                    lits_at_current_dl++;
                seen[alit.var()] = true;
            }
        }
    }
    if (!hasAntecedent(curr_lit)) {
        minimizeAndStoreUIPClause(curr_lit.neg(), tmp_clause, seen);
    }
//  if (var(curr_lit).decision_level > assertion_level_)
//      assertion_level_ = var(curr_lit).decision_level;
}

void Solver::printOnlineStats() {
    if (config_.quiet)
        return;

    std::cout << std::endl;
    std::cout << "time elapsed: " << stopwatch_.getElapsedSeconds() << "s" << std::endl;
    if (config_.verbose) {
        std::cout << "conflict clauses (all / bin / unit) \t";
        std::cout << num_conflict_clauses();
        std::cout << "/" << statistics_.num_binary_conflict_clauses_ << "/"
                  << unit_clauses_.size() << std::endl;
        std::cout << "failed literals found by implicit BCP \t "
                  << statistics_.num_failed_literals_detected_ << std::endl;
        ;

        std::cout << "implicit BCP miss rate \t "
                  << statistics_.implicitBCP_miss_rate() * 100 << "%";
        std::cout << std::endl;

        comp_manager_.gatherStatistics();

        std::cout << "cache size " << statistics_.cache_MB_memory_usage() << "MB" << std::endl;
        std::cout << "components (stored / hits) \t\t"
                  << statistics_.cached_component_count() << "/"
                  << statistics_.cache_hits() << std::endl;
        std::cout << "avg. variable count (stored / hits) \t"
                  << statistics_.getAvgComponentSize() << "/"
                  << statistics_.getAvgCacheHitSize();
        std::cout << std::endl;
        std::cout << "cache miss rate " << statistics_.cache_miss_rate() * 100 << "%"
                  << std::endl;
    }
}

