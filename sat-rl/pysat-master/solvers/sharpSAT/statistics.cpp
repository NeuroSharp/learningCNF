/*
 * statistics.cpp
 *
 *  Created on: Feb 13, 2013
 *      Author: mthurley
 */

#include "statistics.h"

#include <iostream>
#include <fstream>

using namespace SharpSAT;

void DataAndStatistics::print_final_solution_count() {
    std::cout << final_solution_count_.get_str();
}

void DataAndStatistics::writeToFile(const std::string & file_name) {
    std::ofstream out(file_name, std::ios_base::app);
    unsigned pos = input_file_.find_last_of("/\\");
    out << "<tr>" << std::endl;
    out << "<td>" << input_file_.substr(pos + 1) << "</td>" << std::endl;
    out << "<td>" << num_original_variables_ << "</td>" << std::endl;
    out << "<td>" << num_original_clauses_ << "</td>" << std::endl;
    out << "<td>" << num_decisions_ << "</td>" << std::endl;
    out << "<td>" << time_elapsed_ << "</td>" << std::endl;

    std::string s = final_solution_count_.get_str();
    if (final_solution_count_ == 0)
        s = "UNSAT";
    out << "<td>" << s << "</td>" << std::endl;
    out << "</tr>" << std::endl;
}

void DataAndStatistics::printShort() {
    if (exit_state_ == TIMEOUT) {
        std::cout << std::endl << " TIMEOUT !" << std::endl;
        return;
    }
    std::cout << std::endl << std::endl;
    std::cout << "variables (total / active / free)\t" << num_variables_ << "/"
              << num_used_variables_ << "/" << num_variables_ - num_used_variables_
              << std::endl;
    std::cout << "clauses (removed) \t\t\t" << num_original_clauses_ << " ("
              << num_original_clauses_ - num_clauses() << ")" << std::endl;
    std::cout << "decisions \t\t\t\t" << num_decisions_ << std::endl;
    std::cout << "conflicts \t\t\t\t" << num_conflicts_ << std::endl;
    std::cout << "conflict clauses (all/bin/unit) \t";
    std::cout << num_conflict_clauses();
    std::cout << "/" << num_binary_conflict_clauses_ << "/" << num_unit_clauses_
              << std::endl << std::endl;
    std::cout << "failed literals found by implicit BCP \t "
              << num_failed_literals_detected_ << std::endl;


    std::cout << "implicit BCP miss rate \t " << implicitBCP_miss_rate() * 100 << "%";
    std::cout << std::endl;
    std::cout << "bytes cache size     \t" << cache_bytes_memory_usage()  << "\t"
              << std::endl;

    std::cout << "bytes cache (overall) \t" << overall_cache_bytes_memory_stored()
              << "" << std::endl;
    std::cout << "bytes cache (infra / comps) "
              << (cache_infrastructure_bytes_memory_usage_) << "/"
              << sum_bytes_cached_components_  << "\t" << std::endl;

    std::cout << "bytes pure comp data (curr)    " << sum_bytes_pure_cached_component_data_  << "" << std::endl;
    std::cout << "bytes pure comp data (overall) " << overall_bytes_pure_stored_component_data_ << "" << std::endl;

    std::cout << "bytes cache with sysoverh (curr)    " << sys_overhead_sum_bytes_cached_components_  << "" << std::endl;
    std::cout << "bytes cache with sysoverh (overall) " << sys_overhead_overall_bytes_components_stored_ << "" << std::endl;


    std::cout << "cache (stores / hits) \t\t\t" << num_cached_components_ << "/"
              << num_cache_hits_ << std::endl;
    std::cout << "cache miss rate " << cache_miss_rate() * 100 << "%" << std::endl;
    std::cout << " avg. variable count (stores / hits) \t" << getAvgComponentSize()
              << "/" << getAvgCacheHitSize() << std::endl << std::endl;
    std::cout << "\n# solutions " << std::endl;

    print_final_solution_count();

    std::cout << "\n# END" << std::endl << std::endl;
    std::cout << "time: " << time_elapsed_ << "s\n\n";
}

void DataAndStatistics::getStats(StatsMap& stats)
{
    stats["num_vars_total"] = num_variables_;
    stats["num_vars_active"] = num_used_variables_;
    stats["num_vars_free"] = num_variables_ - num_used_variables_;

    stats["num_conflict_clause_bin"] = num_binary_conflict_clauses_;
    stats["num_conflict_clause_long"] = num_long_conflict_clauses_;

    stats["num_clauses_original"] = num_original_clauses_;
    stats["num_clauses_removed"] = num_original_clauses_ - num_clauses();

    stats["num_decisions"] = num_decisions_;
    stats["num_conflicts"] = num_conflicts_;

    stats["implicit_BCP_miss_rate"]    = implicitBCP_miss_rate() * 100;
    stats["bytes_cache_size"] = cache_bytes_memory_usage();
    stats["bytes_cache_overall"] = overall_cache_bytes_memory_stored();
    stats["num_cashed_stored"] = num_cached_components_;
    stats["num_cashed_hits"] = num_cache_hits_;
    stats["num_cashed_lookups"] = num_cache_look_ups_;
    stats["cache_miss_rate"] = cache_miss_rate() * 100;
    stats["avg_comp_size_stored"] = getAvgComponentSize();
    stats["avg_comp_size_hit"] = getAvgCacheHitSize();
}
