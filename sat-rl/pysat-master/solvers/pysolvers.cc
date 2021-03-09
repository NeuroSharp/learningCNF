/*
 * pysolvers.cc
 *
 *  Created on: Nov 26, 2016
 *      Author: aign
 */

#define PY_SSIZE_T_CLEAN

// #define PY_ARRAY_UNIQUE_SYMBOL PYSAT_ARRAY_API
// #define PYSAT_IMPORTING_PY_ARRAY

#include <Python.h>
#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <functional>
#include <vector>
#include <deque>
#include <numpy/arrayobject.h>

#include <execinfo.h>
#include <stdlib.h>
#include <unistd.h>

#include "common_types.h"

#ifdef WITH_SHARPSAT
#include "sharpSAT/solver.h"
#include "sharpSAT/branching_oracle.h"
#include "sharpSAT/primitive_types.h"
#endif

#ifdef WITH_GLUCOSE30
#include "glucose30/core/Solver.h"
#include "glucose30/oracles/gc/GCOracle.h"
#include "glucose30/oracles/gc/DefaultOracle.h"
#include "glucose30/oracles/gc/PercentageOracle.h"
#include "glucose30/oracles/gc/CounterFactualOracle.h"
#include "glucose30/oracles/gc/ThreeValuedOracle.h"
#include "glucose30/oracles/gc/LBDThresholdOracle.h"
#include "glucose30/oracles/branching/BranchOracle.h"
#include "glucose30/oracles/FeatureExtractor.h"
#endif

#ifdef WITH_GLUCOSE41
#include "glucose41/core/Solver.h"
#endif

#ifdef WITH_LINGELING
#include "lingeling/lglib.h"
#endif

#ifdef WITH_MINICARD
#include "minicard/core/Solver.h"
#endif

#ifdef WITH_MINISAT22
#include "minisat22/core/Solver.h"
#include "minisat22/oracles/Oracle.h"
#endif

#ifdef WITH_MINISATGH
#include "minisatgh/core/Solver.h"
#endif

using namespace std;

// docstrings
//=============================================================================
static char     module_docstring[] = "This module provides a wrapper interface "
				"for several SAT solvers.";
static char        new_docstring[] = "Create a new solver object.";
static char      addcl_docstring[] = "Add a clause to formula.";
static char      addam_docstring[] = "Add an atmost constraint to formula "
				"(for Minicard only).";
static char      solve_docstring[] = "Solve a given CNF instance.";
static char        lim_docstring[] = "Solve a given CNF instance within a budget.";
static char       prop_docstring[] = "Propagate a given set of literals.";
static char     phases_docstring[] = "Set variable polarities.";
static char    tbudget_docstring[] = "Set time limit on sat solving (CPU time).";
static char    cbudget_docstring[] = "Set limit on the number of conflicts.";
static char    pbudget_docstring[] = "Set limit on the number of propagations.";
static char    setincr_docstring[] = "Set incremental mode (for Glucose3 only).";
static char    tracepr_docstring[] = "Trace resolution proof.";
static char       core_docstring[] = "Get an unsatisfiable core if formula is UNSAT.";
static char      model_docstring[] = "Get a model if formula is SAT.";
static char        ngc_docstring[] = "Get number of GC calls.";
static char      nvars_docstring[] = "Get number of variables used by the solver.";
static char       ncls_docstring[] = "Get number of clauses used by the solver.";
static char        del_docstring[] = "Delete a previously created solver object.";
static char  gc_oracle_docstring[] = "Sets the Python callback for GC heuristic on the solver object.";
static char      stats_docstring[] = "Returns the solver specific statistics.";
// static char branch_oracle_docstring[] = "Sets the Python callback for branching heuristic on the solver object.";
static char branching_oracle_docstring[] = "Sets the Python callback for component selection heuristic on the solver object.";
static char  lit_stack_docstring[] = "Returns the literals currently on the trail of the solver.";
static char branching_seq_docstring[] = "Returns the sequence of branching literals taken by the current run of the solver.";
static char lev_zero_vars_docstring[] = "Return the initial unit clauses and thei implications (for sharpSAT).";
static char      rbase_docstring[] = "Set reduce_base attribute of Minisat";
static char     cl_arr_docstring[] = "Gets the current set of clauses from the solver object.";
static char	 cl_labels_docstring[] = "Gets the features (labels) for the input/learnt clauses.";
static char var_labels_docstring[] = "Gets the variable features (labels)";
static char lit_labels_docstring[] = "Gets the literal features (labels)";
static char 	   gss_docstring[] = "Get the global solver state (features)";
static char  terminate_docstring[] = "Terminate the current solving process.";
static char     reward_docstring[] = "Returnt the current calculated reward for the RL agent.";

static PyObject *SATError;
static jmp_buf env;

// function declaration for functions available in module
//=============================================================================
extern "C" {
#ifdef WITH_SHARPSAT
    static PyObject *py_sharpsat_new              (PyObject *, PyObject *);
    static PyObject *py_sharpsat_solve            (PyObject *, PyObject *);
    static PyObject *py_sharpsat_nof_vars         (PyObject *, PyObject *);
    static PyObject *py_sharpsat_nof_cls          (PyObject *, PyObject *);
    static PyObject *py_sharpsat_del              (PyObject *, PyObject *);
    static PyObject *py_sharpsat_branching_oracle (PyObject *, PyObject *);
    static PyObject *py_sharpsat_lit_stack        (PyObject *, PyObject *);
    static PyObject *py_sharpsat_branching_seq    (PyObject *, PyObject *);
    static PyObject *py_sharpsat_lit_labels       (PyObject *, PyObject *);
    static PyObject *py_sharpsat_terminate        (PyObject *, PyObject *);
    static PyObject *py_sharpsat_reward           (PyObject *, PyObject *);
    static PyObject *py_sharpsat_lev_zero_vars    (PyObject *, PyObject *);
    static PyObject *py_sharpsat_stats            (PyObject *, PyObject *);
#endif
#ifdef WITH_GLUCOSE30
	static PyObject *py_glucose3_new       (PyObject *, PyObject *);
	static PyObject *py_glucose3_add_cl    (PyObject *, PyObject *);
	static PyObject *py_glucose3_solve     (PyObject *, PyObject *);
	static PyObject *py_glucose3_solve_lim (PyObject *, PyObject *);
	static PyObject *py_glucose3_propagate (PyObject *, PyObject *);
	static PyObject *py_glucose3_setphases (PyObject *, PyObject *);
    static PyObject *py_glucose3_tbudget   (PyObject *, PyObject *);
	static PyObject *py_glucose3_cbudget   (PyObject *, PyObject *);
	static PyObject *py_glucose3_pbudget   (PyObject *, PyObject *);
	static PyObject *py_glucose3_setincr   (PyObject *, PyObject *);
	static PyObject *py_glucose3_tracepr   (PyObject *, PyObject *);
	static PyObject *py_glucose3_core      (PyObject *, PyObject *);
	static PyObject *py_glucose3_model     (PyObject *, PyObject *);
    static PyObject *py_glucose3_nof_gc    (PyObject *, PyObject *);
	static PyObject *py_glucose3_nof_vars  (PyObject *, PyObject *);
	static PyObject *py_glucose3_nof_cls   (PyObject *, PyObject *);
	static PyObject *py_glucose3_del       (PyObject *, PyObject *);
    static PyObject *py_glucose3_gc_oracle (PyObject *, PyObject *);
    static PyObject *py_glucose3_branching_oracle (PyObject *, PyObject *);
    static PyObject *py_glucose3_stats     (PyObject *, PyObject *);
    static PyObject *py_glucose3_rbase     (PyObject *, PyObject *);
    static PyObject *py_glucose3_cl_arr    (PyObject *, PyObject *);
    static PyObject *py_glucose3_cl_labels (PyObject *, PyObject *);
    static PyObject *py_glucose3_var_labels(PyObject *, PyObject *);
    static PyObject *py_glucose3_lit_labels(PyObject *, PyObject *);
    static PyObject *py_glucose3_gss       (PyObject *, PyObject *);
    static PyObject *py_glucose3_terminate (PyObject *, PyObject *);
    static PyObject *py_glucose3_reward    (PyObject *, PyObject *);
#endif
#ifdef WITH_GLUCOSE41
	static PyObject *py_glucose41_new       (PyObject *, PyObject *);
	static PyObject *py_glucose41_add_cl    (PyObject *, PyObject *);
	static PyObject *py_glucose41_solve     (PyObject *, PyObject *);
	static PyObject *py_glucose41_solve_lim (PyObject *, PyObject *);
	static PyObject *py_glucose41_propagate (PyObject *, PyObject *);
	static PyObject *py_glucose41_setphases (PyObject *, PyObject *);
	static PyObject *py_glucose41_cbudget   (PyObject *, PyObject *);
	static PyObject *py_glucose41_pbudget   (PyObject *, PyObject *);
	static PyObject *py_glucose41_setincr   (PyObject *, PyObject *);
	static PyObject *py_glucose41_tracepr   (PyObject *, PyObject *);
	static PyObject *py_glucose41_core      (PyObject *, PyObject *);
	static PyObject *py_glucose41_model     (PyObject *, PyObject *);
	static PyObject *py_glucose41_nof_vars  (PyObject *, PyObject *);
	static PyObject *py_glucose41_nof_cls   (PyObject *, PyObject *);
	static PyObject *py_glucose41_del       (PyObject *, PyObject *);
#endif
#ifdef WITH_LINGELING
	static PyObject *py_lingeling_new       (PyObject *, PyObject *);
	static PyObject *py_lingeling_add_cl    (PyObject *, PyObject *);
	static PyObject *py_lingeling_solve     (PyObject *, PyObject *);
	static PyObject *py_lingeling_setphases (PyObject *, PyObject *);
	static PyObject *py_lingeling_tracepr   (PyObject *, PyObject *);
	static PyObject *py_lingeling_core      (PyObject *, PyObject *);
	static PyObject *py_lingeling_model     (PyObject *, PyObject *);
	static PyObject *py_lingeling_nof_vars  (PyObject *, PyObject *);
	static PyObject *py_lingeling_nof_cls   (PyObject *, PyObject *);
	static PyObject *py_lingeling_del       (PyObject *, PyObject *);
#endif
#ifdef WITH_MINICARD
	static PyObject *py_minicard_new       (PyObject *, PyObject *);
	static PyObject *py_minicard_add_cl    (PyObject *, PyObject *);
	static PyObject *py_minicard_add_am    (PyObject *, PyObject *);
	static PyObject *py_minicard_solve     (PyObject *, PyObject *);
	static PyObject *py_minicard_solve_lim (PyObject *, PyObject *);
	static PyObject *py_minicard_propagate (PyObject *, PyObject *);
	static PyObject *py_minicard_setphases (PyObject *, PyObject *);
	static PyObject *py_minicard_cbudget   (PyObject *, PyObject *);
	static PyObject *py_minicard_pbudget   (PyObject *, PyObject *);
	static PyObject *py_minicard_core      (PyObject *, PyObject *);
	static PyObject *py_minicard_model     (PyObject *, PyObject *);
	static PyObject *py_minicard_nof_vars  (PyObject *, PyObject *);
	static PyObject *py_minicard_nof_cls   (PyObject *, PyObject *);
	static PyObject *py_minicard_del       (PyObject *, PyObject *);
#endif
#ifdef WITH_MINISAT22
	static PyObject *py_minisat22_new       (PyObject *, PyObject *);
	static PyObject *py_minisat22_add_cl    (PyObject *, PyObject *);
	static PyObject *py_minisat22_solve     (PyObject *, PyObject *);
	static PyObject *py_minisat22_solve_lim (PyObject *, PyObject *);
	static PyObject *py_minisat22_propagate (PyObject *, PyObject *);
	static PyObject *py_minisat22_setphases (PyObject *, PyObject *);
	static PyObject *py_minisat22_cbudget   (PyObject *, PyObject *);
	static PyObject *py_minisat22_pbudget   (PyObject *, PyObject *);
	static PyObject *py_minisat22_core      (PyObject *, PyObject *);
	static PyObject *py_minisat22_model     (PyObject *, PyObject *);
	static PyObject *py_minisat22_nof_vars  (PyObject *, PyObject *);
	static PyObject *py_minisat22_nof_cls   (PyObject *, PyObject *);
	static PyObject *py_minisat22_del       (PyObject *, PyObject *);
	static PyObject *py_minisat22_gc_oracle (PyObject *, PyObject *);
	static PyObject *py_minisat22_rbase     (PyObject *, PyObject *);
	static PyObject *py_minisat22_cl_arr    (PyObject *, PyObject *);
	static PyObject *py_minisat22_cl_labels (PyObject *, PyObject *);
	static PyObject *py_minisat22_var_labels(PyObject *, PyObject *);
	static PyObject *py_minisat22_gss       (PyObject *, PyObject *);
	static PyObject *py_minisat22_terminate (PyObject *, PyObject *);
	static PyObject *py_minisat22_reward    (PyObject *, PyObject *);
#endif
#ifdef WITH_MINISATGH
	static PyObject *py_minisatgh_new       (PyObject *, PyObject *);
	static PyObject *py_minisatgh_add_cl    (PyObject *, PyObject *);
	static PyObject *py_minisatgh_solve     (PyObject *, PyObject *);
	static PyObject *py_minisatgh_solve_lim (PyObject *, PyObject *);
	static PyObject *py_minisatgh_propagate (PyObject *, PyObject *);
	static PyObject *py_minisatgh_setphases (PyObject *, PyObject *);
	static PyObject *py_minisatgh_cbudget   (PyObject *, PyObject *);
	static PyObject *py_minisatgh_pbudget   (PyObject *, PyObject *);
	static PyObject *py_minisatgh_core      (PyObject *, PyObject *);
	static PyObject *py_minisatgh_model     (PyObject *, PyObject *);
	static PyObject *py_minisatgh_nof_vars  (PyObject *, PyObject *);
	static PyObject *py_minisatgh_nof_cls   (PyObject *, PyObject *);
	static PyObject *py_minisatgh_del       (PyObject *, PyObject *);
#endif
}

// module specification
//=============================================================================
static PyMethodDef module_methods[] = {
#ifdef WITH_SHARPSAT
    { "sharpsat_new",              py_sharpsat_new,              METH_VARARGS,              new_docstring },
    { "sharpsat_solve",            py_sharpsat_solve,            METH_VARARGS,            solve_docstring },
    { "sharpsat_nof_vars",         py_sharpsat_nof_vars,         METH_VARARGS,            nvars_docstring },
    { "sharpsat_nof_cls",          py_sharpsat_nof_cls,          METH_VARARGS,             ncls_docstring },
    { "sharpsat_del",              py_sharpsat_del,              METH_VARARGS,              del_docstring },
    { "sharpsat_branching_oracle", py_sharpsat_branching_oracle, METH_VARARGS, branching_oracle_docstring },
    { "sharpsat_lit_stack",        py_sharpsat_lit_stack,        METH_VARARGS,        lit_stack_docstring },
    { "sharpsat_branching_seq",    py_sharpsat_branching_seq,    METH_VARARGS,    branching_seq_docstring },
    { "sharpsat_lit_labels",       py_sharpsat_lit_labels,       METH_VARARGS,       lit_labels_docstring },
    { "sharpsat_terminate",        py_sharpsat_terminate,        METH_VARARGS,        terminate_docstring },
    { "sharpsat_reward",           py_sharpsat_reward,           METH_VARARGS,           reward_docstring },
    { "sharpsat_lev_zero_vars",    py_sharpsat_lev_zero_vars,    METH_VARARGS,    lev_zero_vars_docstring },
    { "sharpsat_stats",            py_sharpsat_stats,            METH_VARARGS,            stats_docstring },

#endif
#ifdef WITH_GLUCOSE30
	{ "glucose3_new",        py_glucose3_new,         METH_VARARGS,        new_docstring },
    { "glucose3_add_cl",     py_glucose3_add_cl,      METH_VARARGS,      addcl_docstring },
    { "glucose3_solve",      py_glucose3_solve,       METH_VARARGS,      solve_docstring },
    { "glucose3_solve_lim",  py_glucose3_solve_lim,   METH_VARARGS,        lim_docstring },
    { "glucose3_propagate",  py_glucose3_propagate,   METH_VARARGS,       prop_docstring },
    { "glucose3_setphases",  py_glucose3_setphases,   METH_VARARGS,     phases_docstring },
    { "glucose3_tbudget",    py_glucose3_tbudget,     METH_VARARGS,    tbudget_docstring },
    { "glucose3_cbudget",    py_glucose3_cbudget,     METH_VARARGS,    cbudget_docstring },
    { "glucose3_pbudget",    py_glucose3_pbudget,     METH_VARARGS,    pbudget_docstring },
    { "glucose3_setincr",    py_glucose3_setincr,     METH_VARARGS,    setincr_docstring },
    { "glucose3_tracepr",    py_glucose3_tracepr,     METH_VARARGS,    tracepr_docstring },
    { "glucose3_core",       py_glucose3_core,        METH_VARARGS,       core_docstring },
    { "glucose3_model",      py_glucose3_model,       METH_VARARGS,      model_docstring },
    { "glucose3_nof_gc",     py_glucose3_nof_gc,      METH_VARARGS,        ngc_docstring },
    { "glucose3_nof_vars",   py_glucose3_nof_vars,    METH_VARARGS,      nvars_docstring },
    { "glucose3_nof_cls",    py_glucose3_nof_cls,     METH_VARARGS,       ncls_docstring },
	{ "glucose3_del",        py_glucose3_del,         METH_VARARGS,        del_docstring },
    { "glucose3_gc_oracle",  py_glucose3_gc_oracle,   METH_VARARGS,  gc_oracle_docstring },
    { "glucose3_branching_oracle", py_glucose3_branching_oracle,   METH_VARARGS,  gc_oracle_docstring },
    { "glucose3_stats",      py_glucose3_stats,       METH_VARARGS,      stats_docstring },
    { "glucose3_rbase",      py_glucose3_rbase,       METH_VARARGS,      rbase_docstring },
    { "glucose3_cl_arr",     py_glucose3_cl_arr,      METH_VARARGS,     cl_arr_docstring },
    { "glucose3_cl_labels",  py_glucose3_cl_labels,   METH_VARARGS,  cl_labels_docstring },
    { "glucose3_var_labels", py_glucose3_var_labels,  METH_VARARGS, var_labels_docstring },
    { "glucose3_lit_labels", py_glucose3_lit_labels,  METH_VARARGS, lit_labels_docstring },
    { "glucose3_gss",        py_glucose3_gss,         METH_VARARGS,        gss_docstring },
    { "glucose3_terminate",  py_glucose3_terminate,   METH_VARARGS,  terminate_docstring },
    { "glucose3_reward",     py_glucose3_reward,      METH_VARARGS,     reward_docstring },
#endif
#ifdef WITH_GLUCOSE41
	{ "glucose41_new",       py_glucose41_new,       METH_VARARGS,     new_docstring },
    { "glucose41_add_cl",    py_glucose41_add_cl,    METH_VARARGS,   addcl_docstring },
    { "glucose41_solve",     py_glucose41_solve,     METH_VARARGS,   solve_docstring },
    { "glucose41_solve_lim", py_glucose41_solve_lim, METH_VARARGS,     lim_docstring },
    { "glucose41_propagate", py_glucose41_propagate, METH_VARARGS,    prop_docstring },
    { "glucose41_setphases", py_glucose41_setphases, METH_VARARGS,  phases_docstring },
    { "glucose41_cbudget",   py_glucose41_cbudget,   METH_VARARGS, cbudget_docstring },
    { "glucose41_pbudget",   py_glucose41_pbudget,   METH_VARARGS, pbudget_docstring },
    { "glucose41_setincr",   py_glucose41_setincr,   METH_VARARGS, setincr_docstring },
    { "glucose41_tracepr",   py_glucose41_tracepr,   METH_VARARGS, tracepr_docstring },
    { "glucose41_core",      py_glucose41_core,      METH_VARARGS,    core_docstring },
    { "glucose41_model",     py_glucose41_model,     METH_VARARGS,   model_docstring },
    { "glucose41_nof_vars",  py_glucose41_nof_vars,  METH_VARARGS,   nvars_docstring },
    { "glucose41_nof_cls",   py_glucose41_nof_cls,   METH_VARARGS,    ncls_docstring },
	{ "glucose41_del",       py_glucose41_del,       METH_VARARGS,     del_docstring },
#endif
#ifdef WITH_LINGELING
	{ "lingeling_new",       py_lingeling_new,       METH_VARARGS,     new_docstring },
    { "lingeling_add_cl",    py_lingeling_add_cl,    METH_VARARGS,   addcl_docstring },
    { "lingeling_solve",     py_lingeling_solve,     METH_VARARGS,   solve_docstring },
    { "lingeling_setphases", py_lingeling_setphases, METH_VARARGS,  phases_docstring },
    { "lingeling_tracepr",   py_lingeling_tracepr,   METH_VARARGS, tracepr_docstring },
    { "lingeling_core",      py_lingeling_core,      METH_VARARGS,    core_docstring },
    { "lingeling_model",     py_lingeling_model,     METH_VARARGS,   model_docstring },
    { "lingeling_nof_vars",  py_lingeling_nof_vars,  METH_VARARGS,   nvars_docstring },
    { "lingeling_nof_cls",   py_lingeling_nof_cls,   METH_VARARGS,    ncls_docstring },
	{ "lingeling_del",       py_lingeling_del,       METH_VARARGS,     del_docstring },
#endif
#ifdef WITH_MINICARD
	{ "minicard_new",       py_minicard_new,       METH_VARARGS,     new_docstring },
    { "minicard_add_cl",    py_minicard_add_cl,    METH_VARARGS,   addcl_docstring },
    { "minicard_solve",     py_minicard_solve,     METH_VARARGS,   solve_docstring },
    { "minicard_solve_lim", py_minicard_solve_lim, METH_VARARGS,     lim_docstring },
    { "minicard_propagate", py_minicard_propagate, METH_VARARGS,    prop_docstring },
    { "minicard_setphases", py_minicard_setphases, METH_VARARGS,  phases_docstring },
    { "minicard_cbudget",   py_minicard_cbudget,   METH_VARARGS, cbudget_docstring },
    { "minicard_pbudget",   py_minicard_pbudget,   METH_VARARGS, pbudget_docstring },
    { "minicard_core",      py_minicard_core,      METH_VARARGS,    core_docstring },
    { "minicard_model",     py_minicard_model,     METH_VARARGS,   model_docstring },
    { "minicard_nof_vars",  py_minicard_nof_vars,  METH_VARARGS,   nvars_docstring },
    { "minicard_nof_cls",   py_minicard_nof_cls,   METH_VARARGS,    ncls_docstring },
	{ "minicard_del",       py_minicard_del,       METH_VARARGS,     del_docstring },
    { "minicard_add_am",    py_minicard_add_am,    METH_VARARGS,   addam_docstring },
#endif
#ifdef WITH_MINISAT22
	{ "minisat22_new",        py_minisat22_new,        METH_VARARGS,        new_docstring },
    { "minisat22_add_cl",     py_minisat22_add_cl,     METH_VARARGS,      addcl_docstring },
    { "minisat22_solve",      py_minisat22_solve,      METH_VARARGS,      solve_docstring },
    { "minisat22_solve_lim",  py_minisat22_solve_lim,  METH_VARARGS,        lim_docstring },
    { "minisat22_propagate",  py_minisat22_propagate,  METH_VARARGS,       prop_docstring },
    { "minisat22_setphases",  py_minisat22_setphases,  METH_VARARGS,     phases_docstring },
    { "minisat22_cbudget",    py_minisat22_cbudget,    METH_VARARGS,    cbudget_docstring },
    { "minisat22_pbudget",    py_minisat22_pbudget,    METH_VARARGS,    pbudget_docstring },
    { "minisat22_core",       py_minisat22_core,       METH_VARARGS,       core_docstring },
    { "minisat22_model",      py_minisat22_model,      METH_VARARGS,      model_docstring },
    { "minisat22_nof_vars",   py_minisat22_nof_vars,   METH_VARARGS,      nvars_docstring },
    { "minisat22_nof_cls",    py_minisat22_nof_cls,    METH_VARARGS,       ncls_docstring },
    { "minisat22_del",        py_minisat22_del,        METH_VARARGS,        del_docstring },
    { "minisat22_gc_oracle",  py_minisat22_gc_oracle,  METH_VARARGS,  gc_oracle_docstring },
    { "minisat22_rbase",      py_minisat22_rbase,      METH_VARARGS,      rbase_docstring },
    { "minisat22_cl_arr",     py_minisat22_cl_arr,     METH_VARARGS,     cl_arr_docstring },
    { "minisat22_cl_labels",  py_minisat22_cl_labels,  METH_VARARGS,  cl_labels_docstring },
    { "minisat22_var_labels", py_minisat22_var_labels, METH_VARARGS, var_labels_docstring },
    { "minisat22_gss", 		  py_minisat22_gss, 	   METH_VARARGS, 		gss_docstring },
    { "minisat22_terminate",  py_minisat22_terminate,  METH_VARARGS,  terminate_docstring },
    { "minisat22_reward",     py_minisat22_reward, 	   METH_VARARGS,     reward_docstring },
#endif
#ifdef WITH_MINISATGH
	{ "minisatgh_new",       py_minisatgh_new,       METH_VARARGS,     new_docstring },
    { "minisatgh_add_cl",    py_minisatgh_add_cl,    METH_VARARGS,   addcl_docstring },
    { "minisatgh_solve",     py_minisatgh_solve,     METH_VARARGS,   solve_docstring },
    { "minisatgh_solve_lim", py_minisatgh_solve_lim, METH_VARARGS,     lim_docstring },
    { "minisatgh_propagate", py_minisatgh_propagate, METH_VARARGS,    prop_docstring },
    { "minisatgh_setphases", py_minisatgh_setphases, METH_VARARGS,  phases_docstring },
    { "minisatgh_cbudget",   py_minisatgh_cbudget,   METH_VARARGS, cbudget_docstring },
    { "minisatgh_pbudget",   py_minisatgh_pbudget,   METH_VARARGS, pbudget_docstring },
    { "minisatgh_core",      py_minisatgh_core,      METH_VARARGS,    core_docstring },
    { "minisatgh_model",     py_minisatgh_model,     METH_VARARGS,   model_docstring },
    { "minisatgh_nof_vars",  py_minisatgh_nof_vars,  METH_VARARGS,   nvars_docstring },
    { "minisatgh_nof_cls",   py_minisatgh_nof_cls,   METH_VARARGS,    ncls_docstring },
	{ "minisatgh_del",       py_minisatgh_del,       METH_VARARGS,     del_docstring },
#endif
	{ NULL, NULL, 0, NULL }
};

extern "C" {

// signal handler for SIGINT
//=============================================================================
static void sigint_handler(int signum)
{
	longjmp(env, -1);
}

// signal handler for SIGSEGV
//=============================================================================
static void sigsegv_handler(int signum)
{
	void *array[10];
	size_t size;

	// get void*'s for all entries on the stack
	size = backtrace(array, 10);

	// print out all the frames to stderr
	fprintf(stderr, "Error: signal %d:\n", signum);
	backtrace_symbols_fd(array, size, STDERR_FILENO);
	exit(1);
}

#if PY_MAJOR_VERSION >= 3
// PyInt_asLong()
//=============================================================================
static int pyint_to_cint(PyObject *i_obj)
{
	return PyLong_AsLong(i_obj);
}

// PyInt_fromLong()
//=============================================================================
static PyObject *pyint_from_cint(int i)
{
	return PyLong_FromLong(i);
}

// PyInt_Check()
//=============================================================================
static int pyint_check(PyObject *i_obj)
{
	return PyLong_Check(i_obj);
}

// PyCapsule_New()
//=============================================================================
static PyObject *void_to_pyobj(void *ptr)
{
	return PyCapsule_New(ptr, NULL, NULL);
}

// PyCapsule_GetPointer()
//=============================================================================
static void *pyobj_to_void(PyObject *obj)
{
	return PyCapsule_GetPointer(obj, NULL);
}

// module initialization
//=============================================================================
static struct PyModuleDef module_def = {
	PyModuleDef_HEAD_INIT,
	"pysolvers",       /* m_name */
	module_docstring,  /* m_doc */
	-1,                /* m_size */
	module_methods,    /* m_methods */
	NULL,              /* m_reload */
	NULL,              /* m_traverse */
	NULL,              /* m_clear */
	NULL,              /* m_free */
};

PyMODINIT_FUNC PyInit_pysolvers(void)
{
	PyObject *m = PyModule_Create(&module_def);

	if (m == NULL)
		return NULL;

	SATError = PyErr_NewException((char *)"pysolvers.error", NULL, NULL);
	Py_INCREF(SATError);

	if (PyModule_AddObject(m, "error", SATError) < 0) {
		Py_DECREF(SATError);
		return NULL;
	}

    if (!PyArray_API) { // Make sure to call import_array only once
        _import_array();
    }

	return m;
}
#else

// PyInt_asLong()
//=============================================================================
static int pyint_to_cint(PyObject *i_obj)
{
	return PyInt_AsLong(i_obj);
}

// PyInt_fromLong()
//=============================================================================
static PyObject *pyint_from_cint(int i)
{
	return PyInt_FromLong(i);
}

// PyInt_Check()
//=============================================================================
static int pyint_check(PyObject *i_obj)
{
	return PyInt_Check(i_obj);
}

// PyCObject_FromVoidPtr()
//=============================================================================
static PyObject *void_to_pyobj(void *ptr)
{
	return PyCObject_FromVoidPtr(ptr, NULL);
}

// PyCObject_AsVoidPtr()
//=============================================================================
static void *pyobj_to_void(PyObject *obj)
{
	return PyCObject_AsVoidPtr(obj);
}

// module initialization
//=============================================================================
PyMODINIT_FUNC initpysolvers(void)
{
	PyObject *m = Py_InitModule3("pysolvers", module_methods,
			module_docstring);

	if (m == NULL)
		return;

	SATError = PyErr_NewException((char *)"pysolvers.error", NULL, NULL);
	Py_INCREF(SATError);
	PyModule_AddObject(m, "error", SATError);

    if (!PyArray_API) { // Make sure to call import_array only once
        _import_array();
    }
}
#endif

}


template<class T>
static PyArrayObject* vec_to_nparray(const vector<T>& q, int type_num = PyArray_DOUBLE){
    // rows not empty
    if( !q.empty() ){
       size_t nRows = q.size();
       npy_intp dims[] = { static_cast<npy_intp>(nRows)};

       PyArrayObject* q_array = (PyArrayObject *) PyArray_SimpleNew(1, dims, type_num);
       T *q_array_pointer = (T*) PyArray_DATA(q_array);

       copy(q.begin(),q.end(),q_array_pointer);

       return q_array;

    // no data at all
    } else {
        npy_intp dims[] = {0};
        return (PyArrayObject*) PyArray_ZEROS(1, dims, type_num, 0);
    }
}

template<class T>
static PyArrayObject* vec_to_nparray(const vector<vector<T>>& q, int type_num = PyArray_DOUBLE){

   // rows not empty
   if( !q.empty() ){

      // column not empty
      if( !q[0].empty() ){

        size_t nRows = q.size();
        size_t nCols = q[0].size();
        npy_intp dims[2] = {static_cast<npy_intp>(nRows),
                                static_cast<npy_intp>(nCols)};
        PyArrayObject* q_array = (PyArrayObject *) PyArray_SimpleNew(2, dims, type_num);

        T *vec_array_pointer = (T*) PyArray_DATA(q_array);

        // copy vector line by line ... maybe could be done at one
        for (size_t iRow=0; iRow < q.size(); ++iRow){

          if( q[iRow].size() != nCols){
             Py_DECREF(q_array); // delete
             throw(string("Can not convert vector<vector<T>> to np.array, since c++ matrix shape is not uniform."));
          }

          copy(q[iRow].begin(),q[iRow].end(),vec_array_pointer+iRow*nCols);
        }

        return q_array;

     // Empty columns
     } else {
        npy_intp dims[2] = {static_cast<npy_intp>(q.size()), 0};
        return (PyArrayObject*) PyArray_ZEROS(2, dims, type_num, 0);
     }


   // no data at all
   } else {
      npy_intp dims[2] = {0, 0};
      return (PyArrayObject*) PyArray_ZEROS(2, dims, type_num, 0);
   }

}

extern "C" {

static PyObject *civg_to_pyobject(CVIG matrix) {
    PyArrayObject* rows = vec_to_nparray(std::get<0>(matrix), PyArray_INT);
    PyArrayObject* cols = vec_to_nparray(std::get<1>(matrix), PyArray_INT);
    PyArrayObject* data = vec_to_nparray(std::get<2>(matrix), PyArray_INT);


    PyObject *ret = Py_BuildValue("(OOO)", rows, cols, data);

    return ret;
}

// C to Numpy Conversion functions
//=============================================================================
void* callback_wrapper(PyObject *pyCallback, PyObject *args=NULL)
{
    PyObject *result = PyEval_CallObject(pyCallback, args);
    PyArrayObject *res_array = NULL;
    // Note that the return type can technically even be an NPY_NOTYPE, but we only ever need double
    // so we use NPY_DOUBLE here for simplicity, this way the client can pass int without any problem
    // and those ints will get converted to double
    if (result)
        res_array = (PyArrayObject*) PyArray_FROM_OTF(result, NPY_NOTYPE, NPY_IN_ARRAY);

    // Check for erros in the callback
    if (result == NULL || res_array == NULL) {
        /* cannot return NULL... */
        printf("Error in callback:\n");
        PyErr_Print();

        // TODO: Check if we can fail more gracefully instead of exiting.
        exit(1);
    }

    Py_XDECREF(result);

    if (PyArray_SIZE(res_array) == 0)
        return NULL;
    else
        return (void*) PyArray_DATA(res_array);

}

// C to Numpy Conversion functions
//=============================================================================
void void_callback_wrapper(PyObject *pyCallback, PyObject *args=NULL)
{
    PyObject *result = PyEval_CallObject(pyCallback, args);

    // Check for erros in the callback
    if (result == NULL) {
        /* cannot return NULL... */
        printf("Error in callback:\n");
        PyErr_Print();

        // TODO: Check if we can fail more gracefully instead of exiting.
        exit(1);
    }

    Py_XDECREF(result);
}

// API for SharpSAT
//=============================================================================
#ifdef WITH_SHARPSAT

static PyObject *py_sharpsat_new(PyObject *self, PyObject *args)
{
    SharpSAT::Solver *s = new SharpSAT::Solver();
    int b_obj, time_obj;

    if (!PyArg_ParseTuple(args, "ii", &b_obj, &time_obj))
        return NULL;

    if (s == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
            "Cannot create a new solver.");
        return NULL;
    }

    if (time_obj > 0)
        s->setTimeBound(time_obj);
    s->config().quiet = (b_obj == 0); // verbose = ~ quiet

    return void_to_pyobj((void *)s);
}

//
//=============================================================================
static PyObject *py_sharpsat_del(PyObject *self, PyObject *args)
{
    PyObject *s_obj;

    if (!PyArg_ParseTuple(args, "O", &s_obj))
        return NULL;

    // get pointer to solver
    SharpSAT::Solver *s = (SharpSAT::Solver *)pyobj_to_void(s_obj);

    delete s;

    PyObject *ret = Py_BuildValue("");
    return ret;
}

//
//=============================================================================
static PyObject *py_sharpsat_solve(PyObject *self, PyObject *args)
{
    signal(SIGINT,  sigint_handler);
    signal(SIGSEGV, sigsegv_handler);

    PyObject *s_obj;
    char     *f_obj;  // file path

    if (!PyArg_ParseTuple(args, "Os", &s_obj, &f_obj))
        return NULL;

    // get pointer to solver
    SharpSAT::Solver *s = (SharpSAT::Solver *)pyobj_to_void(s_obj);

    if (setjmp(env) != 0) {
        PyErr_SetString(SATError, "Caught keyboard interrupt");
        return NULL;
    }

    s->solve(std::string(f_obj));
    std::string restr = s->statistics().final_solution_count().get_str();

    PyObject *ret;
    if (s->statistics().exit_state_ == SharpSAT::TIMEOUT)
        ret = Py_BuildValue("");
    else
        ret = PyLong_FromString(restr.c_str(), NULL, 10);

    return ret;
}

//
//=============================================================================
static PyObject *py_sharpsat_nof_vars(PyObject *self, PyObject *args)
{
    PyObject *s_obj;

    if (!PyArg_ParseTuple(args, "O", &s_obj))
        return NULL;

    // get pointer to solver
    SharpSAT::Solver *s = (SharpSAT::Solver *)pyobj_to_void(s_obj);

    int nof_vars = s->statistics().num_variables_;

    PyObject *ret = Py_BuildValue("n", (Py_ssize_t)nof_vars);
    return ret;
}

//
//=============================================================================
static PyObject *py_sharpsat_nof_cls(PyObject *self, PyObject *args)
{

    PyObject *s_obj;

    if (!PyArg_ParseTuple(args, "O", &s_obj))
        return NULL;

    // get pointer to solver
    SharpSAT::Solver *s = (SharpSAT::Solver *)pyobj_to_void(s_obj);

    int nof_cls = s->statistics().num_non_unit_clauses();

    PyObject *ret = Py_BuildValue("n", (Py_ssize_t)nof_cls);
    return ret;
}



// Function that set the python callback on the Oracle instace
//=============================================================================
static PyObject *py_sharpsat_branching_oracle(PyObject *self, PyObject *args)
{
    PyObject *s_obj;
    PyObject *captureCallback;
    PyObject *branchingCallback;

    if (!PyArg_ParseTuple(args, "OOO:capture_cbO:branching_cb",
                            &s_obj, &captureCallback, &branchingCallback))
        return NULL;
    if (!PyCallable_Check(captureCallback) || !PyCallable_Check(branchingCallback)) {
        PyErr_SetString(PyExc_TypeError, "parameter must be callable");
        return NULL;
    }

    // get pointer to solver
    SharpSAT::Solver *s = (SharpSAT::Solver *)pyobj_to_void(s_obj);

    // Add a reference to new callbacks
    Py_XINCREF(captureCallback);
    Py_XINCREF(branchingCallback);

    // Create a new Oracle and set it on the solver instance
    std::function<void(CVIG)> capture_cb = [=](CVIG matirx) {
        PyObject *arg = civg_to_pyobject(matirx);
        return void_callback_wrapper(captureCallback, arg);
    };

    std::function<int*(CVIG)> branching_cb = [=](CVIG matirx) {
        PyObject *arg = civg_to_pyobject(matirx);
        return (int*) callback_wrapper(branchingCallback, arg);
    };

    SharpSAT::BranchingOracle *oracle =
                    new SharpSAT::BranchingOracle(capture_cb, branching_cb);

    s->setBranchingOracle(oracle);

    // Conforming to the the way the rest of the interface returns null
    PyObject *ret = Py_BuildValue("");

    return ret;
}

static PyObject *py_sharpsat_reward(PyObject *self, PyObject *args)
{
    PyObject *s_obj;

    if (!PyArg_ParseTuple(args, "O", &s_obj))
        return NULL;

    // get pointer to solver
    SharpSAT::Solver *s = (SharpSAT::Solver *)pyobj_to_void(s_obj);
    uint64_t reward = s->statistics().getTime(); // This is num_decisions_

    PyObject *ret = PyLong_FromUnsignedLong(reward);
    return ret;
}

// A function that retunrs the stats for glucose
//=============================================================================
static PyObject *py_sharpsat_stats(PyObject *self, PyObject *args)
{
    StatsMap stats;
    PyObject *s_obj;

    if (!PyArg_ParseTuple(args, "O", &s_obj))
        return NULL;

    // get pointer to solver
    SharpSAT::Solver *s = (SharpSAT::Solver *)pyobj_to_void(s_obj);
    SharpSAT::DataAndStatistics dns = s->statistics();

    PyObject *ret = Py_BuildValue("");

    dns.getStats(stats);

    PyObject *pyStats = PyDict_New();
    for (auto item : stats)
        PyDict_SetItemString(pyStats,
            item.first.c_str(), Py_BuildValue("d", (double) item.second));

    ret = Py_BuildValue("O", pyStats);
    Py_DECREF(pyStats);

    return ret;
}

// A method to asynchronous termination of the current solving process
//=============================================================================
static PyObject *py_sharpsat_terminate(PyObject *self, PyObject *args)
{
    PyObject *s_obj;

    if (!PyArg_ParseTuple(args, "O", &s_obj))
        return NULL;

    // get pointer to solver
    SharpSAT::Solver *s = (SharpSAT::Solver *)pyobj_to_void(s_obj);
    s->interrupt();

    PyObject *ret = Py_BuildValue("");
    return ret;
}

// A function that retunrs the labels (features) for all literals
//=============================================================================
static PyObject *py_sharpsat_lit_labels(PyObject *self, PyObject *args)
{
    PyObject *s_obj;

    if (!PyArg_ParseTuple(args, "O", &s_obj))
        return NULL;

    // get pointer to solver
    SharpSAT::Solver *s = (SharpSAT::Solver *)pyobj_to_void(s_obj);
    std::vector<std::vector<double>> var_labels = s->getLitLabels();

    PyArrayObject* PyVar = vec_to_nparray(var_labels);

    PyObject *ret = Py_BuildValue("O", PyVar);
    Py_DECREF(PyVar);

    return ret;
}

// A function that retunrs the list of lits on the trail
//=============================================================================
static PyObject *py_sharpsat_lit_stack(PyObject *self, PyObject *args)
{
    PyObject *s_obj;

    if (!PyArg_ParseTuple(args, "O", &s_obj))
        return NULL;

    // get pointer to solver
    SharpSAT::Solver *s = (SharpSAT::Solver *)pyobj_to_void(s_obj);
    std::vector<int> lit_stack = s->getLitStack();

    PyArrayObject* PyVar = vec_to_nparray(lit_stack, PyArray_INT);

    PyObject *ret = Py_BuildValue("O", PyVar);
    Py_DECREF(PyVar);

    return ret;
}

// Returns the sequence of branching literals taken by the current run of the solver.
//=============================================================================
static PyObject *py_sharpsat_branching_seq(PyObject *self, PyObject *args)
{
    PyObject *s_obj;

    if (!PyArg_ParseTuple(args, "O", &s_obj))
        return NULL;

    // get pointer to solver
    SharpSAT::Solver *s = (SharpSAT::Solver *)pyobj_to_void(s_obj);
    std::vector<int> seq = s->getBranchingSequence(); // This is num_decisions_

    PyArrayObject *PyVar = vec_to_nparray(seq, PyArray_INT);

    PyObject *ret = Py_BuildValue("O", PyVar);
    Py_DECREF(PyVar);
    return ret;
}

// A function that retunrs the list of problem units clauses
//=============================================================================
static PyObject *py_sharpsat_lev_zero_vars(PyObject *self, PyObject *args)
{
    PyObject *s_obj;

    if (!PyArg_ParseTuple(args, "O", &s_obj))
        return NULL;

    // get pointer to solver
    SharpSAT::Solver *s = (SharpSAT::Solver *)pyobj_to_void(s_obj);
    std::vector<int> lit_stack = s->getLevelZeroVars();

    PyArrayObject* PyVar = vec_to_nparray(lit_stack, PyArray_INT);

    PyObject *ret = Py_BuildValue("O", PyVar);
    Py_DECREF(PyVar);

    return ret;
}

#endif

// API for Glucose 3.0
//=============================================================================
#ifdef WITH_GLUCOSE30

// GC policy
enum class GCPolicy {GLUCOSE, LBD_THRSHLD, PERCENTAGE, COUNTER_FACTUAL, THREE_VAL};

static PyObject *py_glucose3_new(PyObject *self, PyObject *args)
{
    uint64_t reduce_base;
    int      gc_freq;

    if (!PyArg_ParseTuple(args, "Li", &reduce_base, &gc_freq))
        return NULL;

    Glucose30::Solver *s = new Glucose30::Solver(reduce_base);
    s->setGCFreq(static_cast<Glucose30::GCFreq>(gc_freq));

    if (s == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
            "Cannot create a new solver.");
        return NULL;
    }

    return void_to_pyobj((void *)s);
}

// Function that set the python callback on the GC Oracle instace
//=============================================================================
static PyObject *py_glucose3_gc_oracle(PyObject *self, PyObject *args)
{
    PyObject *s_obj;
    PyObject *pyCallback;
    int      gcPolicy;
    int      gcStats;

    if (!PyArg_ParseTuple(args, "OiiO:set_callback", &s_obj, &gcPolicy, &gcStats, &pyCallback))
        return NULL;
    if (!PyCallable_Check(pyCallback)) {
        PyErr_SetString(PyExc_TypeError, "parameter must be callable");
        return NULL;
    }

    // get pointer to solver
    Glucose30::Solver *s = (Glucose30::Solver *)pyobj_to_void(s_obj);

    // Add a reference to new pyCallback
    Py_XINCREF(pyCallback);

    // Dispose of the previous oracle
    Glucose30::GCOracle *oracle = s->getGCOracle();
    if (oracle)
        delete oracle;

    // Create a new Oracle and set it on the solver instance
    std::function<double*()> callback = [=]() {
        return (double*) callback_wrapper(pyCallback);
    };

    bool collectStats = (gcStats != 0);
    switch((GCPolicy) gcPolicy) {
        case GCPolicy::LBD_THRSHLD: {
            oracle = new Glucose30::LBDThresholdOracle(s, callback, collectStats);
            break;
        }
        case GCPolicy::PERCENTAGE: {
            oracle = new Glucose30::PercentageOracle(s, callback, collectStats);
            break;
        }
        case GCPolicy::COUNTER_FACTUAL: {
            oracle = new Glucose30::CounterFactualOracle(s, callback, collectStats);
            break;
        }
        case GCPolicy::THREE_VAL: {
            oracle = new Glucose30::ThreeValuedOracle(s, callback, collectStats);
            break;
        }
        default: { // GCPolicy::GLUCOSE
            oracle = new Glucose30::DefaultOracle(s, callback, collectStats);
        }
    }
    s->setGCOracle(oracle);

    // Conforming to the the way the rest of the interface returns null
    PyObject *ret = Py_BuildValue("");

    return ret;
}

// Function that set the python callback on the Branching Oracle instace
//=============================================================================
static PyObject *py_glucose3_branching_oracle(PyObject *self, PyObject *args)
{
    PyObject *s_obj;
    PyObject *pyCallback;
    int      brTrigger;
    int      brFreq;

    if (!PyArg_ParseTuple(args, "OiiO:set_callback", &s_obj, &brTrigger, &brFreq, &pyCallback))
        return NULL;
    if (!PyCallable_Check(pyCallback)) {
        PyErr_SetString(PyExc_TypeError, "parameter must be callable");
        return NULL;
    }

    // get pointer to solver
    Glucose30::Solver *s = (Glucose30::Solver *)pyobj_to_void(s_obj);

    // Add a reference to new pyCallback
    Py_XINCREF(pyCallback);

    // Dispose of the previous oracle
    Glucose30::BranchOracle *oracle = s->getBranchOracle();
    if (oracle)
        delete oracle;

    // Create a new Oracle and set it on the solver instance
    std::function<double*()> callback = [=]() {
        return (double*) callback_wrapper(pyCallback);
    };

    oracle = new Glucose30::BranchOracle(s, callback,
        static_cast<Glucose30::BRTrigger>(brTrigger), brFreq);
    s->setBranchOracle(oracle);

    // Conforming to the the way the rest of the interface returns null
    PyObject *ret = Py_BuildValue("");

    return ret;
}

// A function that retunrs the stats for glucose
//=============================================================================
static PyObject *py_glucose3_stats(PyObject *self, PyObject *args)
{
    StatsMap stats;
    PyObject *s_obj;

    if (!PyArg_ParseTuple(args, "O", &s_obj))
        return NULL;

    // get pointer to solver
    Glucose30::Solver *s = (Glucose30::Solver *)pyobj_to_void(s_obj);
    Glucose30::GCOracle *oracle = s->getGCOracle();

    PyObject *ret = Py_BuildValue("");
    if (oracle) {
        oracle->getStats(stats);

        PyObject *pyStats = PyDict_New();
        for (auto item : stats)
            PyDict_SetItemString(pyStats,
                item.first.c_str(), Py_BuildValue("d", (double) item.second));

        ret = Py_BuildValue("O", pyStats);
        Py_DECREF(pyStats);
    }

    return ret;
}

// set the reduce_base for Minisat, in Glucose30 that variable is renamed to: specialIncReduceDB
// Note that the reduce_base is now set on the instructor of Solver. This implementation
// is only kept for reference. The sizes of stats queues are init'ed according to
// reduce_base so if reduce_base is changed half-way through, those changes
// will not be reflected on the sizes of those queues.
//=============================================================================
static PyObject *py_glucose3_rbase(PyObject *self, PyObject *args)
{
    PyObject *s_obj;
    uint64_t reduce_base;

    if (!PyArg_ParseTuple(args, "OL", &s_obj, &reduce_base))
        return NULL;

    // get pointer to solver
    Glucose30::Solver *s = (Glucose30::Solver *)pyobj_to_void(s_obj);
    s->setReduceBase(reduce_base);

    PyObject *ret = Py_BuildValue("");
    return ret;
}

// auxiliary function for declaring new variables
//=============================================================================
static inline void glucose3_declare_vars(Glucose30::Solver *s, const int max_id)
{
	while (s->nVars() < max_id + 1)
		s->newVar();
}

//
//=============================================================================
static PyObject *py_glucose3_add_cl(PyObject *self, PyObject *args)
{
    PyObject *s_obj;
    PyObject *c_obj;

    if (!PyArg_ParseTuple(args, "OO", &s_obj, &c_obj))
        return NULL;

    // get pointer to solver
    Glucose30::Solver *s = (Glucose30::Solver *)pyobj_to_void(s_obj);
    Glucose30::vec<Glucose30::Lit> cl;
    int vari = 0;

    // clause iterator
    PyObject *i_obj = PyObject_GetIter(c_obj);
    if (i_obj == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "Clause does not seem to be an iterable object.");
        return NULL;
    }

    PyObject *l_obj;
    while ((l_obj = PyIter_Next(i_obj)) != NULL) {
        if (!pyint_check(l_obj)) {
            Py_DECREF(l_obj);
            Py_DECREF(i_obj);
            PyErr_SetString(PyExc_TypeError, "integer expected");
            return NULL;
        }

        int parsed_lit = pyint_to_cint(l_obj);
        Py_DECREF(l_obj);

        if (parsed_lit == 0) {
            Py_DECREF(i_obj);
            PyErr_SetString(PyExc_ValueError, "non-zero integer expected");
            return NULL;
        }

        vari = abs(parsed_lit)-1;
        while (vari >= s->nVars())
            s->newVar();
        cl.push((parsed_lit > 0) ? Glucose30::mkLit(vari) : ~Glucose30::mkLit(vari));
    }

    Py_DECREF(i_obj);

    bool res = s->addClause(cl);

    PyObject *ret = PyBool_FromLong((long)res);
    return ret;
}

// A function that retunrs the list of problem clauses as a sparse matrix
//=============================================================================
static PyObject *py_glucose3_cl_arr(PyObject *self, PyObject *args)
{
    PyObject *s_obj;
    int b_obj;

    if (!PyArg_ParseTuple(args, "Oi", &s_obj, &b_obj))
        return NULL;

    // get pointer to solver
    Glucose30::Solver *s = (Glucose30::Solver *)pyobj_to_void(s_obj);
    Glucose30::FeatureExtractor *fext = s->getFeatExtractor();

    bool learnts = b_obj!=0;
    // CVIG claVarMatrix = learnts? fext->cl_adj() : fext->clause2SPArray(false);
    CVIG claVarMatrix = fext->cl_adj();
    PyObject *ret = civg_to_pyobject(claVarMatrix);

    return ret;
}

// A function that retunrs the labels (features) for input/learnt clauses
//=============================================================================
static PyObject *py_glucose3_cl_labels(PyObject *self, PyObject *args)
{
    PyObject *s_obj;
    int b_obj;

    if (!PyArg_ParseTuple(args, "Oi", &s_obj, &b_obj))
        return NULL;

    // get pointer to solver
    Glucose30::Solver *s = (Glucose30::Solver *)pyobj_to_void(s_obj);
    Glucose30::FeatureExtractor *fext = s->getFeatExtractor();

    bool learnts = b_obj != 0;
    std::vector<std::vector<double>> cl_labels = fext->getClaLabels(learnts);
    PyArrayObject* PyCla = vec_to_nparray(cl_labels);

    PyObject *ret = Py_BuildValue("O", PyCla);
    Py_DECREF(PyCla);

    return ret;
}

// A function that retunrs the labels (features) for all literals
//=============================================================================
static PyObject *py_glucose3_lit_labels(PyObject *self, PyObject *args)
{
    PyObject *s_obj;

    if (!PyArg_ParseTuple(args, "O", &s_obj))
        return NULL;

    // get pointer to solver
    Glucose30::Solver *s = (Glucose30::Solver *)pyobj_to_void(s_obj);
    Glucose30::FeatureExtractor *fext = s->getFeatExtractor();
    std::vector<std::vector<double>> var_labels = fext->getLitLabels();
    PyArrayObject* PyVar = vec_to_nparray(var_labels);

    PyObject *ret = Py_BuildValue("O", PyVar);
    Py_DECREF(PyVar);

    return ret;
}

// A function that retunrs the labels (features) for all problem variables
//=============================================================================
static PyObject *py_glucose3_var_labels(PyObject *self, PyObject *args)
{
    PyObject *s_obj;

    if (!PyArg_ParseTuple(args, "O", &s_obj))
        return NULL;

    // get pointer to solver
    Glucose30::Solver *s = (Glucose30::Solver *)pyobj_to_void(s_obj);
    Glucose30::FeatureExtractor *fext = s->getFeatExtractor();
    std::vector<std::vector<double>> var_labels = fext->getVarLabels();
    PyArrayObject* PyVar = vec_to_nparray(var_labels);

    PyObject *ret = Py_BuildValue("O", PyVar);
    Py_DECREF(PyVar);

    return ret;
}

// A function that retunrs the global solver state (features)
//=============================================================================
static PyObject *py_glucose3_gss(PyObject *self, PyObject *args)
{
    PyObject *s_obj;

    if (!PyArg_ParseTuple(args, "O", &s_obj))
        return NULL;

    // get pointer to solver
    Glucose30::Solver *s = (Glucose30::Solver *)pyobj_to_void(s_obj);
    Glucose30::FeatureExtractor *fext = s->getFeatExtractor();
    std::unordered_map<std::string, std::vector<double>> gss = fext->getGSS();

    PyObject *pyGSS = PyDict_New();
    for (auto item : gss)
        PyDict_SetItemString(pyGSS,
            item.first.c_str(), (PyObject *) vec_to_nparray(item.second));

    PyObject *ret = Py_BuildValue("O", pyGSS);
    Py_DECREF(pyGSS);

    return ret;
}

// A method to asynchronous termination of the current solving process
//=============================================================================
static PyObject *py_glucose3_terminate(PyObject *self, PyObject *args)
{
    PyObject *s_obj;

    if (!PyArg_ParseTuple(args, "O", &s_obj))
        return NULL;

    // get pointer to solver
    Glucose30::Solver *s = (Glucose30::Solver *)pyobj_to_void(s_obj);
    s->interrupt();

    PyObject *ret = Py_BuildValue("");
    return ret;
}

// Return the current reward for the RL agent
// step_cnt: If True, the number of branching steps is returned,
//     otherwise the op_cnt.
//=============================================================================
static PyObject *py_glucose3_reward(PyObject *self, PyObject *args)
{
    PyObject *s_obj;
    int b_obj;

    if (!PyArg_ParseTuple(args, "Oi", &s_obj, &b_obj))
        return NULL;

    bool step_cnt = b_obj != 0;

    // get pointer to solver
    Glucose30::Solver *s = (Glucose30::Solver *)pyobj_to_void(s_obj);
    uint64_t reward = s->getReward(step_cnt);

    PyObject *ret = PyLong_FromUnsignedLongLong(reward);
    return ret;
}

//
//=============================================================================
static PyObject *py_glucose3_solve(PyObject *self, PyObject *args)
{
	signal(SIGINT,  sigint_handler);
    signal(SIGSEGV, sigsegv_handler);

	PyObject *s_obj;
	PyObject *a_obj;  // assumptions

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &a_obj))
		return NULL;

	// get pointer to solver
	Glucose30::Solver *s = (Glucose30::Solver *)pyobj_to_void(s_obj);

	int size = (int)PyList_Size(a_obj);
	Glucose30::vec<Glucose30::Lit> a((int)size);

	int max_var = -1;
	for (int i = 0; i < size; ++i) {
		PyObject *l_obj = PyList_GetItem(a_obj, i);
		int l = pyint_to_cint(l_obj);
		a[i] = (l > 0) ? Glucose30::mkLit(l, false) : Glucose30::mkLit(-l, true);

		if (abs(l) > max_var)
			max_var = abs(l);
	}

	if (max_var > 0)
		glucose3_declare_vars(s, max_var);

	if (setjmp(env) != 0) {
		PyErr_SetString(SATError, "Caught keyboard interrupt");
		return NULL;
	}

    bool res = s->solve(a);

    PyObject *ret = PyBool_FromLong((long)res);
    return ret;
}

//
//=============================================================================
static PyObject *py_glucose3_solve_lim(PyObject *self, PyObject *args)
{
	signal(SIGINT, sigint_handler);

	PyObject *s_obj;
	PyObject *a_obj;  // assumptions

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &a_obj))
		return NULL;

	// get pointer to solver
	Glucose30::Solver *s = (Glucose30::Solver *)pyobj_to_void(s_obj);

	int size = (int)PyList_Size(a_obj);
	Glucose30::vec<Glucose30::Lit> a((int)size);

	int max_var = -1;
	for (int i = 0; i < size; ++i) {
		PyObject *l_obj = PyList_GetItem(a_obj, i);
		int l = pyint_to_cint(l_obj);
		a[i] = (l > 0) ? Glucose30::mkLit(l, false) : Glucose30::mkLit(-l, true);

		if (abs(l) > max_var)
			max_var = abs(l);
	}

	if (max_var > 0)
		glucose3_declare_vars(s, max_var);

	if (setjmp(env) != 0) {
		PyErr_SetString(SATError, "Caught keyboard interrupt");
		return NULL;
	}

	Glucose30::lbool res = s->solveLimited(a);

	PyObject *ret;
	if (res != Glucose30::lbool((uint8_t)2))  // l_Undef
		ret = PyBool_FromLong((long)!(Glucose30::toInt(res)));
	else
		ret = Py_BuildValue("");  // return Python's None if l_Undef

	return ret;
}

//
//=============================================================================
static PyObject *py_glucose3_propagate(PyObject *self, PyObject *args)
{
	signal(SIGINT, sigint_handler);

	PyObject *s_obj;
	PyObject *a_obj;  // assumptions
	int save_phases;

	if (!PyArg_ParseTuple(args, "OOi", &s_obj, &a_obj, &save_phases))
		return NULL;

	// get pointer to solver
	Glucose30::Solver *s = (Glucose30::Solver *)pyobj_to_void(s_obj);

	int size = (int)PyList_Size(a_obj);
	Glucose30::vec<Glucose30::Lit> a((int)size);

	int max_var = -1;
	for (int i = 0; i < size; ++i) {
		PyObject *l_obj = PyList_GetItem(a_obj, i);
		int l = pyint_to_cint(l_obj);
		a[i] = (l > 0) ? Glucose30::mkLit(l, false) : Glucose30::mkLit(-l, true);

		if (abs(l) > max_var)
			max_var = abs(l);
	}

	if (max_var > 0)
		glucose3_declare_vars(s, max_var);

	if (setjmp(env) != 0) {
		PyErr_SetString(SATError, "Caught keyboard interrupt");
		return NULL;
	}

	Glucose30::vec<Glucose30::Lit> p;
	bool res = s->prop_check(a, p, save_phases);

	PyObject *propagated = PyList_New(p.size());
	for (int i = 0; i < p.size(); ++i) {
		int l = Glucose30::var(p[i]) * (Glucose30::sign(p[i]) ? -1 : 1);
		PyObject *lit = pyint_from_cint(l);
		PyList_SetItem(propagated, i, lit);
	}

	PyObject *ret = Py_BuildValue("nO", (Py_ssize_t)res, propagated);
	Py_DECREF(propagated);

	return ret;
}

//
//=============================================================================
static PyObject *py_glucose3_setphases(PyObject *self, PyObject *args)
{
	signal(SIGINT, sigint_handler);

	PyObject *s_obj;
	PyObject *p_obj;  // polarities given as a list of integer literals

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &p_obj))
		return NULL;

	// get pointer to solver
	Glucose30::Solver *s = (Glucose30::Solver *)pyobj_to_void(s_obj);

	int size = (int)PyList_Size(p_obj);
	vector<int> p(size);

	int max_var = -1;
	for (int i = 0; i < size; ++i) {
		PyObject *l_obj = PyList_GetItem(p_obj, i);
		p[i] = pyint_to_cint(l_obj);

		if (abs(p[i]) > max_var)
			max_var = abs(p[i]);
	}

	if (max_var > 0)
		glucose3_declare_vars(s, max_var);

	for (int i = 0; i < size; ++i)
		s->setPolarity(abs(p[i]), p[i] < 0);

	PyObject *ret = Py_BuildValue("");
	return ret;
}

//
//=============================================================================
static PyObject *py_glucose3_tbudget(PyObject *self, PyObject *args)
{
    PyObject *s_obj;
    int budget;

    if (!PyArg_ParseTuple(args, "Oi", &s_obj, &budget))
        return NULL;

    // get pointer to solver
    Glucose30::Solver *s = (Glucose30::Solver *)pyobj_to_void(s_obj);

    if (budget != 0 && budget != -1)  // it is 0 by default
        s->setTimeBudget(budget);
    else
        s->budgetOff();

    PyObject *ret = Py_BuildValue("");
    return ret;
}

//
//=============================================================================
static PyObject *py_glucose3_cbudget(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	int64_t budget;

	if (!PyArg_ParseTuple(args, "Ol", &s_obj, &budget))
		return NULL;

	// get pointer to solver
	Glucose30::Solver *s = (Glucose30::Solver *)pyobj_to_void(s_obj);

	if (budget != 0 && budget != -1)  // it is 0 by default
		s->setConfBudget(budget);
	else
		s->budgetOff();

	PyObject *ret = Py_BuildValue("");
	return ret;
}

//
//=============================================================================
static PyObject *py_glucose3_pbudget(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	int64_t budget;

	if (!PyArg_ParseTuple(args, "Ol", &s_obj, &budget))
		return NULL;

	// get pointer to solver
	Glucose30::Solver *s = (Glucose30::Solver *)pyobj_to_void(s_obj);

	if (budget != 0 && budget != -1)  // it is 0 by default
		s->setPropBudget(budget);
	else
		s->budgetOff();

	PyObject *ret = Py_BuildValue("");
	return ret;
}

//
//=============================================================================
static PyObject *py_glucose3_setincr(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	Glucose30::Solver *s = (Glucose30::Solver *)pyobj_to_void(s_obj);

	s->setIncrementalMode();

	PyObject *ret = Py_BuildValue("");
	return ret;
}

//
//=============================================================================
static PyObject *py_glucose3_tracepr(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	PyObject *p_obj;

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &p_obj))
		return NULL;

	// get pointer to solver
#if PY_MAJOR_VERSION < 3
	Glucose30::Solver *s = (Glucose30::Solver *)PyCObject_AsVoidPtr(s_obj);

	s->certifiedOutput = PyFile_AsFile(p_obj);
	PyFile_IncUseCount((PyFileObject *)p_obj);
#else
	Glucose30::Solver *s = (Glucose30::Solver *)PyCapsule_GetPointer(s_obj, NULL);

	int fd = PyObject_AsFileDescriptor(p_obj);
	if (fd == -1) {
		PyErr_SetString(SATError, "Cannot create proof file descriptor!");
		return NULL;
	}

	s->certifiedOutput = fdopen(fd, "w+");
	if (s->certifiedOutput == 0) {
		PyErr_SetString(SATError, "Cannot create proof file pointer!");
		return NULL;
	}

	setlinebuf(s->certifiedOutput);
	Py_INCREF(p_obj);
#endif

	s->certifiedUNSAT  = true;
	s->certifiedPyFile = (void *)p_obj;

	PyObject *ret = Py_BuildValue("");
	return ret;
}

//
//=============================================================================
static PyObject *py_glucose3_core(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	Glucose30::Solver *s = (Glucose30::Solver *)pyobj_to_void(s_obj);

	Glucose30::vec<Glucose30::Lit> *c = &(s->conflict);  // minisat's conflict

	PyObject *core = PyList_New(c->size());
	for (int i = 0; i < c->size(); ++i) {
		int l = Glucose30::var((*c)[i]) * (Glucose30::sign((*c)[i]) ? 1 : -1);
		PyObject *lit = pyint_from_cint(l);
		PyList_SetItem(core, i, lit);
	}

	PyObject *ret = Py_None;

	if (c->size())
		ret = Py_BuildValue("O", core);

	Py_DECREF(core);
	return ret;
}

//
//=============================================================================
static PyObject *py_glucose3_model(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	Glucose30::Solver *s = (Glucose30::Solver *)pyobj_to_void(s_obj);

	// minisat's model
	Glucose30::vec<Glucose30::lbool> *m = &(s->model);

	if (m->size()) {
		// l_True fails to work
		Glucose30::lbool True = Glucose30::lbool((uint8_t)0);

		PyObject *model = PyList_New(m->size());
		for (int i = 0; i < m->size(); ++i) {
			int l = (i + 1) * ((*m)[i] == True ? 1 : -1);
			PyObject *lit = pyint_from_cint(l);
			PyList_SetItem(model, i, lit);
		}

		PyObject *ret = Py_BuildValue("O", model);
		Py_DECREF(model);
		return ret;
    }

	Py_RETURN_NONE;
}

//
//=============================================================================
static PyObject *py_glucose3_nof_gc(PyObject *self, PyObject *args)
{
    PyObject *s_obj;

    if (!PyArg_ParseTuple(args, "O", &s_obj))
        return NULL;

    // get pointer to solver
    Glucose30::Solver *s = (Glucose30::Solver *)pyobj_to_void(s_obj);

    int nof_gc = s->nbReduceDB;  // 0 is a dummy variable

    PyObject *ret = Py_BuildValue("n", (Py_ssize_t)nof_gc);
    return ret;
}

//
//=============================================================================
static PyObject *py_glucose3_nof_vars(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	Glucose30::Solver *s = (Glucose30::Solver *)pyobj_to_void(s_obj);

	int nof_vars = s->nVars() - 1;  // 0 is a dummy variable

	PyObject *ret = Py_BuildValue("n", (Py_ssize_t)nof_vars);
	return ret;
}

//
//=============================================================================
static PyObject *py_glucose3_nof_cls(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
    int b_obj;

    if (!PyArg_ParseTuple(args, "Oi", &s_obj, &b_obj))
        return NULL;

    // get pointer to solver
    Glucose30::Solver *s = (Glucose30::Solver *)pyobj_to_void(s_obj);

    int nof_cls;
    if(b_obj == 0) // original clauses
        nof_cls = s->nClauses();
    else // current learnt clauses
        nof_cls = s->nLearnts();

	PyObject *ret = Py_BuildValue("n", (Py_ssize_t)nof_cls);
	return ret;
}

//
//=============================================================================
static PyObject *py_glucose3_del(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
#if PY_MAJOR_VERSION < 3
	Glucose30::Solver *s = (Glucose30::Solver *)PyCObject_AsVoidPtr(s_obj);

	if (s->certifiedUNSAT == true)
		PyFile_DecUseCount((PyFileObject *)(s->certifiedPyFile));
#else
	Glucose30::Solver *s = (Glucose30::Solver *)PyCapsule_GetPointer(s_obj, NULL);

	if (s->certifiedUNSAT == true)
		Py_DECREF((PyObject *)s->certifiedPyFile);
#endif

	delete s;

	PyObject *ret = Py_BuildValue("");
	return ret;
}
#endif  // WITH_GLUCOSE30

// API for Glucose 4.1
//=============================================================================
#ifdef WITH_GLUCOSE41
static PyObject *py_glucose41_new(PyObject *self, PyObject *args)
{
	Glucose41::Solver *s = new Glucose41::Solver();

	if (s == NULL) {
		PyErr_SetString(PyExc_RuntimeError,
				"Cannot create a new solver.");
		return NULL;
	}

	return void_to_pyobj((void *)s);
}

// auxiliary function for declaring new variables
//=============================================================================
static inline void glucose41_declare_vars(Glucose41::Solver *s, const int max_id)
{
	while (s->nVars() < max_id + 1)
		s->newVar();
}

//
//=============================================================================
static PyObject *py_glucose41_add_cl(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	PyObject *c_obj;

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &c_obj))
		return NULL;

	// get pointer to solver
	Glucose41::Solver *s = (Glucose41::Solver *)pyobj_to_void(s_obj);
	Glucose41::vec<Glucose41::Lit> cl;
	int max_var = -1;

	// clause iterator
	PyObject *i_obj = PyObject_GetIter(c_obj);
	if (i_obj == NULL) {
		PyErr_SetString(PyExc_RuntimeError,
				"Clause does not seem to be an iterable object.");
		return NULL;
	}

	PyObject *l_obj;
	while ((l_obj = PyIter_Next(i_obj)) != NULL) {
		if (!pyint_check(l_obj)) {
			Py_DECREF(l_obj);
			Py_DECREF(i_obj);
			PyErr_SetString(PyExc_TypeError, "integer expected");
			return NULL;
		}

		int l = pyint_to_cint(l_obj);
		Py_DECREF(l_obj);

		if (l == 0) {
			Py_DECREF(i_obj);
			PyErr_SetString(PyExc_ValueError, "non-zero integer expected");
			return NULL;
		}

		cl.push((l > 0) ? Glucose41::mkLit(l, false) : Glucose41::mkLit(-l, true));

		if (abs(l) > max_var)
			max_var = abs(l);
	}

	Py_DECREF(i_obj);

	if (max_var > 0)
		glucose41_declare_vars(s, max_var);

	bool res = s->addClause(cl);

	PyObject *ret = PyBool_FromLong((long)res);
	return ret;
}

//
//=============================================================================
static PyObject *py_glucose41_solve(PyObject *self, PyObject *args)
{
	signal(SIGINT, sigint_handler);

	PyObject *s_obj;
	PyObject *a_obj;  // assumptions

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &a_obj))
		return NULL;

	// get pointer to solver
	Glucose41::Solver *s = (Glucose41::Solver *)pyobj_to_void(s_obj);

	int size = (int)PyList_Size(a_obj);
	Glucose41::vec<Glucose41::Lit> a((int)size);

	int max_var = -1;
	for (int i = 0; i < size; ++i) {
		PyObject *l_obj = PyList_GetItem(a_obj, i);
		int l = pyint_to_cint(l_obj);
		a[i] = (l > 0) ? Glucose41::mkLit(l, false) : Glucose41::mkLit(-l, true);

		if (abs(l) > max_var)
			max_var = abs(l);
	}

	if (max_var > 0)
		glucose41_declare_vars(s, max_var);

	if (setjmp(env) != 0) {
		PyErr_SetString(SATError, "Caught keyboard interrupt");
		return NULL;
	}

	bool res = s->solve(a);

	PyObject *ret = PyBool_FromLong((long)res);
	return ret;
}

//
//=============================================================================
static PyObject *py_glucose41_solve_lim(PyObject *self, PyObject *args)
{
	signal(SIGINT, sigint_handler);

	PyObject *s_obj;
	PyObject *a_obj;  // assumptions

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &a_obj))
		return NULL;

	// get pointer to solver
	Glucose41::Solver *s = (Glucose41::Solver *)pyobj_to_void(s_obj);

	int size = (int)PyList_Size(a_obj);
	Glucose41::vec<Glucose41::Lit> a((int)size);

	int max_var = -1;
	for (int i = 0; i < size; ++i) {
		PyObject *l_obj = PyList_GetItem(a_obj, i);
		int l = pyint_to_cint(l_obj);
		a[i] = (l > 0) ? Glucose41::mkLit(l, false) : Glucose41::mkLit(-l, true);

		if (abs(l) > max_var)
			max_var = abs(l);
	}

	if (max_var > 0)
		glucose41_declare_vars(s, max_var);

	if (setjmp(env) != 0) {
		PyErr_SetString(SATError, "Caught keyboard interrupt");
		return NULL;
	}

	Glucose41::lbool res = s->solveLimited(a);

	PyObject *ret;
	if (res != Glucose41::lbool((uint8_t)2))  // l_Undef
		ret = PyBool_FromLong((long)!(Glucose41::toInt(res)));
	else
		ret = Py_BuildValue("");  // return Python's None if l_Undef

	return ret;
}

//
//=============================================================================
static PyObject *py_glucose41_propagate(PyObject *self, PyObject *args)
{
	signal(SIGINT, sigint_handler);

	PyObject *s_obj;
	PyObject *a_obj;  // assumptions
	int save_phases;

	if (!PyArg_ParseTuple(args, "OOi", &s_obj, &a_obj, &save_phases))
		return NULL;

	// get pointer to solver
	Glucose41::Solver *s = (Glucose41::Solver *)pyobj_to_void(s_obj);

	int size = (int)PyList_Size(a_obj);
	Glucose41::vec<Glucose41::Lit> a((int)size);

	int max_var = -1;
	for (int i = 0; i < size; ++i) {
		PyObject *l_obj = PyList_GetItem(a_obj, i);
		int l = pyint_to_cint(l_obj);
		a[i] = (l > 0) ? Glucose41::mkLit(l, false) : Glucose41::mkLit(-l, true);

		if (abs(l) > max_var)
			max_var = abs(l);
	}

	if (max_var > 0)
		glucose41_declare_vars(s, max_var);

	if (setjmp(env) != 0) {
		PyErr_SetString(SATError, "Caught keyboard interrupt");
		return NULL;
	}

	Glucose41::vec<Glucose41::Lit> p;
	bool res = s->prop_check(a, p, save_phases);

	PyObject *propagated = PyList_New(p.size());
	for (int i = 0; i < p.size(); ++i) {
		int l = Glucose41::var(p[i]) * (Glucose41::sign(p[i]) ? -1 : 1);
		PyObject *lit = pyint_from_cint(l);
		PyList_SetItem(propagated, i, lit);
	}

	PyObject *ret = Py_BuildValue("nO", (Py_ssize_t)res, propagated);
	Py_DECREF(propagated);

	return ret;
}

//
//=============================================================================
static PyObject *py_glucose41_setphases(PyObject *self, PyObject *args)
{
	signal(SIGINT, sigint_handler);

	PyObject *s_obj;
	PyObject *p_obj;  // polarities given as a list of integer literals

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &p_obj))
		return NULL;

	// get pointer to solver
	Glucose41::Solver *s = (Glucose41::Solver *)pyobj_to_void(s_obj);

	int size = (int)PyList_Size(p_obj);
	vector<int> p(size);

	int max_var = -1;
	for (int i = 0; i < size; ++i) {
		PyObject *l_obj = PyList_GetItem(p_obj, i);
		p[i] = pyint_to_cint(l_obj);

		if (abs(p[i]) > max_var)
			max_var = abs(p[i]);
	}

	if (max_var > 0)
		glucose41_declare_vars(s, max_var);

	for (int i = 0; i < size; ++i)
		s->setPolarity(abs(p[i]), p[i] < 0);

	PyObject *ret = Py_BuildValue("");
	return ret;
}

//
//=============================================================================
static PyObject *py_glucose41_cbudget(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	int64_t budget;

	if (!PyArg_ParseTuple(args, "Ol", &s_obj, &budget))
		return NULL;

	// get pointer to solver
	Glucose41::Solver *s = (Glucose41::Solver *)pyobj_to_void(s_obj);

	if (budget != 0 && budget != -1)  // it is 0 by default
		s->setConfBudget(budget);
	else
		s->budgetOff();

	PyObject *ret = Py_BuildValue("");
	return ret;
}

//
//=============================================================================
static PyObject *py_glucose41_pbudget(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	int64_t budget;

	if (!PyArg_ParseTuple(args, "Ol", &s_obj, &budget))
		return NULL;

	// get pointer to solver
	Glucose41::Solver *s = (Glucose41::Solver *)pyobj_to_void(s_obj);

	if (budget != 0 && budget != -1)  // it is 0 by default
		s->setPropBudget(budget);
	else
		s->budgetOff();

	PyObject *ret = Py_BuildValue("");
	return ret;
}

//
//=============================================================================
static PyObject *py_glucose41_setincr(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	Glucose41::Solver *s = (Glucose41::Solver *)pyobj_to_void(s_obj);

	s->setIncrementalMode();

	PyObject *ret = Py_BuildValue("");
	return ret;
}

//
//=============================================================================
static PyObject *py_glucose41_tracepr(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	PyObject *p_obj;

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &p_obj))
		return NULL;

	// get pointer to solver
#if PY_MAJOR_VERSION < 3
	Glucose41::Solver *s = (Glucose41::Solver *)PyCObject_AsVoidPtr(s_obj);

	s->certifiedOutput = PyFile_AsFile(p_obj);
	PyFile_IncUseCount((PyFileObject *)p_obj);
#else
	Glucose41::Solver *s = (Glucose41::Solver *)PyCapsule_GetPointer(s_obj, NULL);

	int fd = PyObject_AsFileDescriptor(p_obj);
	if (fd == -1) {
		PyErr_SetString(SATError, "Cannot create proof file descriptor!");
		return NULL;
	}

	s->certifiedOutput = fdopen(fd, "w+");
	if (s->certifiedOutput == 0) {
		PyErr_SetString(SATError, "Cannot create proof file pointer!");
		return NULL;
	}

	setlinebuf(s->certifiedOutput);
	Py_INCREF(p_obj);
#endif

	s->certifiedUNSAT  = true;
	s->certifiedPyFile = (void *)p_obj;

	PyObject *ret = Py_BuildValue("");
	return ret;
}

//
//=============================================================================
static PyObject *py_glucose41_core(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	Glucose41::Solver *s = (Glucose41::Solver *)pyobj_to_void(s_obj);

	Glucose41::vec<Glucose41::Lit> *c = &(s->conflict);  // minisat's conflict

	PyObject *core = PyList_New(c->size());
	for (int i = 0; i < c->size(); ++i) {
		int l = Glucose41::var((*c)[i]) * (Glucose41::sign((*c)[i]) ? 1 : -1);
		PyObject *lit = pyint_from_cint(l);
		PyList_SetItem(core, i, lit);
	}

	PyObject *ret = Py_None;

	if (c->size())
		ret = Py_BuildValue("O", core);

	Py_DECREF(core);
	return ret;
}

//
//=============================================================================
static PyObject *py_glucose41_model(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	PyObject *ret = Py_None;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	Glucose41::Solver *s = (Glucose41::Solver *)pyobj_to_void(s_obj);

	// minisat's model
	Glucose41::vec<Glucose41::lbool> *m = &(s->model);

	if (m->size()) {
		// l_True fails to work
		Glucose41::lbool True = Glucose41::lbool((uint8_t)0);

		PyObject *model = PyList_New(m->size() - 1);
		for (int i = 1; i < m->size(); ++i) {
			int l = i * ((*m)[i] == True ? 1 : -1);
			PyObject *lit = pyint_from_cint(l);
			PyList_SetItem(model, i - 1, lit);
		}

		ret = Py_BuildValue("O", model);
		Py_DECREF(model);
	}

	return ret;
}

//
//=============================================================================
static PyObject *py_glucose41_nof_vars(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	Glucose41::Solver *s = (Glucose41::Solver *)pyobj_to_void(s_obj);

	int nof_vars = s->nVars() - 1;  // 0 is a dummy variable

	PyObject *ret = Py_BuildValue("n", (Py_ssize_t)nof_vars);
	return ret;
}

//
//=============================================================================
static PyObject *py_glucose41_nof_cls(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	Glucose41::Solver *s = (Glucose41::Solver *)pyobj_to_void(s_obj);

	int nof_cls = s->nClauses();

	PyObject *ret = Py_BuildValue("n", (Py_ssize_t)nof_cls);
	return ret;
}

//
//=============================================================================
static PyObject *py_glucose41_del(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
#if PY_MAJOR_VERSION < 3
	Glucose41::Solver *s = (Glucose41::Solver *)PyCObject_AsVoidPtr(s_obj);

	if (s->certifiedUNSAT == true)
		PyFile_DecUseCount((PyFileObject *)(s->certifiedPyFile));
#else
	Glucose41::Solver *s = (Glucose41::Solver *)PyCapsule_GetPointer(s_obj, NULL);

	if (s->certifiedUNSAT == true)
		Py_DECREF((PyObject *)s->certifiedPyFile);
#endif

	delete s;

	PyObject *ret = Py_BuildValue("");
	return ret;
}
#endif  // WITH_GLUCOSE41

// API for Lingeling
//=============================================================================
#ifdef WITH_LINGELING
static PyObject *py_lingeling_new(PyObject *self, PyObject *args)
{
	LGL *s = lglinit();
	int t_lim = 5000;

	if (s == NULL) {
		PyErr_SetString(PyExc_RuntimeError,
				"Cannot create a new solver.");
		return NULL;
	}

	if (PyArg_ParseTuple(args, "i", &t_lim)){
		lglsettimeout(s, t_lim);
	}

	lglsetopt(s, "simplify", 0);


	return void_to_pyobj((void *)s);
}

//
//=============================================================================
static PyObject *py_lingeling_add_cl(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	PyObject *c_obj;

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &c_obj))
		return NULL;

	// get pointer to solver
	LGL *s = (LGL *)pyobj_to_void(s_obj);

	// clause iterator
	PyObject *i_obj = PyObject_GetIter(c_obj);
	if (i_obj == NULL) {
		PyErr_SetString(PyExc_RuntimeError,
				"Clause does not seem to be an iterable object.");
		return NULL;
	}

	PyObject *l_obj;
	while ((l_obj = PyIter_Next(i_obj)) != NULL) {
		if (!pyint_check(l_obj)) {
			Py_DECREF(l_obj);
			Py_DECREF(i_obj);
			PyErr_SetString(PyExc_TypeError, "integer expected");
			return NULL;
		}

		int l = pyint_to_cint(l_obj);
		Py_DECREF(l_obj);

		if (l == 0) {
			Py_DECREF(i_obj);
			PyErr_SetString(PyExc_ValueError, "non-zero integer expected");
			return NULL;
		}

		lgladd(s, l);
		lglfreeze(s, abs(l));
	}

	lgladd(s, 0);
	Py_DECREF(i_obj);

	PyObject *ret = PyBool_FromLong((long)true);
	return ret;
}

//
//=============================================================================
static PyObject *py_lingeling_tracepr(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	PyObject *p_obj;

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &p_obj))
		return NULL;

	// get pointer to solver
#if PY_MAJOR_VERSION < 3
	LGL *s = (LGL *)PyCObject_AsVoidPtr(s_obj);

	lglsetrace(s, PyFile_AsFile(p_obj));
	PyFile_IncUseCount((PyFileObject *)p_obj);
#else
	LGL *s = (LGL *)PyCapsule_GetPointer(s_obj, NULL);

	int fd = PyObject_AsFileDescriptor(p_obj);
	if (fd == -1) {
		PyErr_SetString(SATError, "Cannot create proof file descriptor!");
		return NULL;
	}

	FILE *lgl_trace_fp = fdopen(fd, "w+");
	if (lgl_trace_fp == NULL) {
		PyErr_SetString(SATError, "Cannot create proof file pointer!");
		return NULL;
	}

	setlinebuf(lgl_trace_fp);
	lglsetrace(s, lgl_trace_fp);
	Py_INCREF(p_obj);
#endif

	lglsetopt (s, "druplig", 1);
	lglsetopt (s, "drupligtrace", 2);

	PyObject *ret = Py_BuildValue("");
	return ret;
}

//
//=============================================================================
static PyObject *py_lingeling_solve(PyObject *self, PyObject *args)
{
	signal(SIGINT, sigint_handler);

	PyObject *s_obj;
	PyObject *a_obj;  // assumptions

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &a_obj))
		return NULL;

	// get pointer to solver
	LGL *s = (LGL *)pyobj_to_void(s_obj);

	int size = (int)PyList_Size(a_obj);

	for (int i = 0; i < size; ++i) {
		PyObject *l_obj = PyList_GetItem(a_obj, i);
		int l = pyint_to_cint(l_obj);

		lglassume(s, l);
	}

	if (setjmp(env) != 0) {
		PyErr_SetString(SATError, "Caught keyboard interrupt");
		return NULL;
	}
	int res = -1;
	// Py_BEGIN_ALLOW_THREADS
	res = lglsat(s); //== 10 ? true : false;
	// Py_END_ALLOW_THREADS

	// PyObject *ret = PyBool_FromLong((long)res);

	PyObject *ret = Py_BuildValue("l", res);
	return ret;
}

//
//=============================================================================
static PyObject *py_lingeling_setphases(PyObject *self, PyObject *args)
{
	signal(SIGINT, sigint_handler);

	PyObject *s_obj;
	PyObject *p_obj;  // polarities given as a list of integer literals

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &p_obj))
		return NULL;

	// get pointer to solver
	LGL *s = (LGL *)pyobj_to_void(s_obj);

	int size = (int)PyList_Size(p_obj);

	for (int i = 0; i < size; ++i) {
		PyObject *l_obj = PyList_GetItem(p_obj, i);
		int lit = pyint_to_cint(l_obj);
		lglsetphase(s, lit);
	}

	PyObject *ret = Py_BuildValue("");
	return ret;
}

//
//=============================================================================
static PyObject *py_lingeling_core(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	PyObject *a_obj;  // assumptions

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &a_obj))
		return NULL;

	// get pointer to solver
	LGL *s = (LGL *)pyobj_to_void(s_obj);

	int size = (int)PyList_Size(a_obj);

	vector<int> c;
	for (int i = 0; i < size; ++i) {
		PyObject *l_obj = PyList_GetItem(a_obj, i);
		int l = pyint_to_cint(l_obj);

		if (lglfailed(s, l))
			c.push_back(l);
	}

	PyObject *core = PyList_New(c.size());
	for (size_t i = 0; i < c.size(); ++i) {
		PyObject *lit = pyint_from_cint(c[i]);
		PyList_SetItem(core, i, lit);
	}

	PyObject *ret = Py_None;

	if (c.size())
		ret = Py_BuildValue("O", core);

	Py_DECREF(core);
	return ret;
}

//
//=============================================================================
static PyObject *py_lingeling_model(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	PyObject *ret = Py_None;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	LGL *s = (LGL *)pyobj_to_void(s_obj);

	int maxvar = lglmaxvar(s);
	if (maxvar) {
		PyObject *model = PyList_New(maxvar);
		for (int i = 1; i <= maxvar; ++i) {
			int l = lglderef(s, i) > 0 ? i : -i;

			PyObject *lit = pyint_from_cint(l);
			PyList_SetItem(model, i - 1, lit);
		}

		ret = Py_BuildValue("O", model);
		Py_DECREF(model);
	}

	return ret;
}

//
//=============================================================================
static PyObject *py_lingeling_nof_vars(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	LGL *s = (LGL *)pyobj_to_void(s_obj);

	int nof_vars = lglmaxvar(s);

	PyObject *ret = Py_BuildValue("n", (Py_ssize_t)nof_vars);
	return ret;
}

//
//=============================================================================
static PyObject *py_lingeling_nof_cls(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	LGL *s = (LGL *)pyobj_to_void(s_obj);

	int nof_cls = lglnclauses(s);

	PyObject *ret = Py_BuildValue("n", (Py_ssize_t)nof_cls);
	return ret;
}

//
//=============================================================================
static PyObject *py_lingeling_del(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	PyObject *p_obj;

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &p_obj))
		return NULL;

	// get pointer to solver
#if PY_MAJOR_VERSION < 3
	LGL *s = (LGL *)PyCObject_AsVoidPtr(s_obj);

	if (p_obj != Py_None)
		PyFile_DecUseCount((PyFileObject *)p_obj);
#else
	LGL *s = (LGL *)PyCapsule_GetPointer(s_obj, NULL);

	if (p_obj != Py_None)
		Py_DECREF(p_obj);
#endif

	lglrelease(s);

	PyObject *ret = Py_BuildValue("");
	return ret;
}
#endif  // WITH_LINGELING

// API for Minicard
//=============================================================================
#ifdef WITH_MINICARD
static PyObject *py_minicard_new(PyObject *self, PyObject *args)
{
	Minicard::Solver *s = new Minicard::Solver();

	if (s == NULL) {
		PyErr_SetString(PyExc_RuntimeError,
				"Cannot create a new solver.");
		return NULL;
	}

	return void_to_pyobj((void *)s);
}

// auxiliary function for declaring new variables
//=============================================================================
static inline void minicard_declare_vars(Minicard::Solver *s, const int max_id)
{
	while (s->nVars() < max_id + 1)
		s->newVar();
}

//
//=============================================================================
static PyObject *py_minicard_add_cl(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	PyObject *c_obj;

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &c_obj))
		return NULL;

	// get pointer to solver
	Minicard::Solver *s = (Minicard::Solver *)pyobj_to_void(s_obj);
	Minicard::vec<Minicard::Lit> cl;
	int max_var = -1;

	// clause iterator
	PyObject *i_obj = PyObject_GetIter(c_obj);
	if (i_obj == NULL) {
		PyErr_SetString(PyExc_RuntimeError,
				"Clause does not seem to be an iterable object.");
		return NULL;
	}

	PyObject *l_obj;
	while ((l_obj = PyIter_Next(i_obj)) != NULL) {
		if (!pyint_check(l_obj)) {
			Py_DECREF(l_obj);
			Py_DECREF(i_obj);
			PyErr_SetString(PyExc_TypeError, "integer expected");
			return NULL;
		}

		int l = pyint_to_cint(l_obj);
		Py_DECREF(l_obj);

		if (l == 0) {
			Py_DECREF(i_obj);
			PyErr_SetString(PyExc_ValueError, "non-zero integer expected");
			return NULL;
		}

		cl.push((l > 0) ? Minicard::mkLit(l, false) : Minicard::mkLit(-l, true));

		if (abs(l) > max_var)
			max_var = abs(l);
	}

	Py_DECREF(i_obj);

	if (max_var > 0)
		minicard_declare_vars(s, max_var);

	bool res = s->addClause(cl);

	PyObject *ret = PyBool_FromLong((long)res);
	return ret;
}

//
//=============================================================================
static PyObject *py_minicard_add_am(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	PyObject *c_obj;
	int64_t rhs;

	if (!PyArg_ParseTuple(args, "OOl", &s_obj, &c_obj, &rhs))
		return NULL;

	// get pointer to solver
	Minicard::Solver *s = (Minicard::Solver *)pyobj_to_void(s_obj);

	int size = (int)PyList_Size(c_obj);
	Minicard::vec<Minicard::Lit> cl((int)size);

	int max_var = -1;
	for (int i = 0; i < size; ++i) {
		PyObject *l_obj = PyList_GetItem(c_obj, i);
		int l = pyint_to_cint(l_obj);
		cl[i] = (l > 0) ? Minicard::mkLit(l, false) : Minicard::mkLit(-l, true);

		if (abs(l) > max_var)
			max_var = abs(l);
	}

	if (max_var > 0)
		minicard_declare_vars(s, max_var);

	bool res = s->addAtMost(cl, rhs);

	PyObject *ret = PyBool_FromLong((long)res);
	return ret;
}

//
//=============================================================================
static PyObject *py_minicard_solve(PyObject *self, PyObject *args)
{
	signal(SIGINT, sigint_handler);

	PyObject *s_obj;
	PyObject *a_obj;  // assumptions

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &a_obj))
		return NULL;

	// get pointer to solver
	Minicard::Solver *s = (Minicard::Solver *)pyobj_to_void(s_obj);

	int size = (int)PyList_Size(a_obj);
	Minicard::vec<Minicard::Lit> a((int)size);

	int max_var = -1;
	for (int i = 0; i < size; ++i) {
		PyObject *l_obj = PyList_GetItem(a_obj, i);
		int l = pyint_to_cint(l_obj);
		a[i] = (l > 0) ? Minicard::mkLit(l, false) : Minicard::mkLit(-l, true);

		if (abs(l) > max_var)
			max_var = abs(l);
	}

	if (max_var > 0)
		minicard_declare_vars(s, max_var);

	if (setjmp(env) != 0) {
		PyErr_SetString(SATError, "Caught keyboard interrupt");
		return NULL;
	}

	bool res = s->solve(a);

	PyObject *ret = PyBool_FromLong((long)res);
	return ret;
}

//
//=============================================================================
static PyObject *py_minicard_solve_lim(PyObject *self, PyObject *args)
{
	signal(SIGINT, sigint_handler);

	PyObject *s_obj;
	PyObject *a_obj;  // assumptions

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &a_obj))
		return NULL;

	// get pointer to solver
	Minicard::Solver *s = (Minicard::Solver *)pyobj_to_void(s_obj);

	int size = (int)PyList_Size(a_obj);
	Minicard::vec<Minicard::Lit> a((int)size);

	int max_var = -1;
	for (int i = 0; i < size; ++i) {
		PyObject *l_obj = PyList_GetItem(a_obj, i);
		int l = pyint_to_cint(l_obj);
		a[i] = (l > 0) ? Minicard::mkLit(l, false) : Minicard::mkLit(-l, true);

		if (abs(l) > max_var)
			max_var = abs(l);
	}

	if (max_var > 0)
		minicard_declare_vars(s, max_var);

	if (setjmp(env) != 0) {
		PyErr_SetString(SATError, "Caught keyboard interrupt");
		return NULL;
	}

	Minicard::lbool res = s->solveLimited(a);

	PyObject *ret;
	if (res != Minicard::lbool((uint8_t)2))  // l_Undef
		ret = PyBool_FromLong((long)!(Minicard::toInt(res)));
	else
		ret = Py_BuildValue("");  // return Python's None if l_Undef

	return ret;
}

//
//=============================================================================
static PyObject *py_minicard_propagate(PyObject *self, PyObject *args)
{
	signal(SIGINT, sigint_handler);

	PyObject *s_obj;
	PyObject *a_obj;  // assumptions
	int save_phases;

	if (!PyArg_ParseTuple(args, "OOi", &s_obj, &a_obj, &save_phases))
		return NULL;

	// get pointer to solver
	Minicard::Solver *s = (Minicard::Solver *)pyobj_to_void(s_obj);

	int size = (int)PyList_Size(a_obj);
	Minicard::vec<Minicard::Lit> a((int)size);

	int max_var = -1;
	for (int i = 0; i < size; ++i) {
		PyObject *l_obj = PyList_GetItem(a_obj, i);
		int l = pyint_to_cint(l_obj);
		a[i] = (l > 0) ? Minicard::mkLit(l, false) : Minicard::mkLit(-l, true);

		if (abs(l) > max_var)
			max_var = abs(l);
	}

	if (max_var > 0)
		minicard_declare_vars(s, max_var);

	if (setjmp(env) != 0) {
		PyErr_SetString(SATError, "Caught keyboard interrupt");
		return NULL;
	}

	Minicard::vec<Minicard::Lit> p;
	bool res = s->prop_check(a, p, save_phases);

	PyObject *propagated = PyList_New(p.size());
	for (int i = 0; i < p.size(); ++i) {
		int l = Minicard::var(p[i]) * (Minicard::sign(p[i]) ? -1 : 1);
		PyObject *lit = pyint_from_cint(l);
		PyList_SetItem(propagated, i, lit);
	}

	PyObject *ret = Py_BuildValue("nO", (Py_ssize_t)res, propagated);
	Py_DECREF(propagated);

	return ret;
}

//
//=============================================================================
static PyObject *py_minicard_setphases(PyObject *self, PyObject *args)
{
	signal(SIGINT, sigint_handler);

	PyObject *s_obj;
	PyObject *p_obj;  // polarities given as a list of integer literals

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &p_obj))
		return NULL;

	// get pointer to solver
	Minicard::Solver *s = (Minicard::Solver *)pyobj_to_void(s_obj);

	int size = (int)PyList_Size(p_obj);
	vector<int> p(size);

	int max_var = -1;
	for (int i = 0; i < size; ++i) {
		PyObject *l_obj = PyList_GetItem(p_obj, i);
		p[i] = pyint_to_cint(l_obj);

		if (abs(p[i]) > max_var)
			max_var = abs(p[i]);
	}

	if (max_var > 0)
		minicard_declare_vars(s, max_var);

	for (int i = 0; i < size; ++i)
		s->setPolarity(abs(p[i]), p[i] < 0);

	PyObject *ret = Py_BuildValue("");
	return ret;
}

//
//=============================================================================
static PyObject *py_minicard_cbudget(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	int64_t budget;

	if (!PyArg_ParseTuple(args, "Ol", &s_obj, &budget))
		return NULL;

	// get pointer to solver
	Minicard::Solver *s = (Minicard::Solver *)pyobj_to_void(s_obj);

	if (budget != 0 && budget != -1)  // it is 0 by default
		s->setConfBudget(budget);
	else
		s->budgetOff();

	PyObject *ret = Py_BuildValue("");
	return ret;
}

//
//=============================================================================
static PyObject *py_minicard_pbudget(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	int64_t budget;

	if (!PyArg_ParseTuple(args, "Ol", &s_obj, &budget))
		return NULL;

	// get pointer to solver
	Minicard::Solver *s = (Minicard::Solver *)pyobj_to_void(s_obj);

	if (budget != 0 && budget != -1)  // it is 0 by default
		s->setPropBudget(budget);
	else
		s->budgetOff();

	PyObject *ret = Py_BuildValue("");
	return ret;
}

//
//=============================================================================
static PyObject *py_minicard_core(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	Minicard::Solver *s = (Minicard::Solver *)pyobj_to_void(s_obj);

	Minicard::vec<Minicard::Lit> *c = &(s->conflict);  // minisat's conflict

	PyObject *core = PyList_New(c->size());
	for (int i = 0; i < c->size(); ++i) {
		int l = Minicard::var((*c)[i]) * (Minicard::sign((*c)[i]) ? 1 : -1);
		PyObject *lit = pyint_from_cint(l);
		PyList_SetItem(core, i, lit);
	}

	PyObject *ret = Py_None;

	if (c->size())
		ret = Py_BuildValue("O", core);

	Py_DECREF(core);
	return ret;
}

//
//=============================================================================
static PyObject *py_minicard_model(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	PyObject *ret = Py_None;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	Minicard::Solver *s = (Minicard::Solver *)pyobj_to_void(s_obj);

	// minisat's model
	Minicard::vec<Minicard::lbool> *m = &(s->model);

	if (m->size()) {
		// l_True fails to work
		Minicard::lbool True = Minicard::lbool((uint8_t)0);

		PyObject *model = PyList_New(m->size() - 1);
		for (int i = 1; i < m->size(); ++i) {
			int l = i * ((*m)[i] == True ? 1 : -1);
			PyObject *lit = pyint_from_cint(l);
			PyList_SetItem(model, i - 1, lit);
		}

		ret = Py_BuildValue("O", model);
		Py_DECREF(model);
	}

	return ret;
}

//
//=============================================================================
static PyObject *py_minicard_nof_vars(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	Minicard::Solver *s = (Minicard::Solver *)pyobj_to_void(s_obj);

	int nof_vars = s->nVars() - 1;  // 0 is a dummy variable

	PyObject *ret = Py_BuildValue("n", (Py_ssize_t)nof_vars);
	return ret;
}

//
//=============================================================================
static PyObject *py_minicard_nof_cls(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	Minicard::Solver *s = (Minicard::Solver *)pyobj_to_void(s_obj);

	int nof_cls = s->nClauses();

	PyObject *ret = Py_BuildValue("n", (Py_ssize_t)nof_cls);
	return ret;
}

//
//=============================================================================
static PyObject *py_minicard_del(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	Minicard::Solver *s = (Minicard::Solver *)pyobj_to_void(s_obj);

	delete s;

	PyObject *ret = Py_BuildValue("");
	return ret;
}
#endif  // WITH_MINICARD

// API for MiniSat 2.2
//=============================================================================
#ifdef WITH_MINISAT22
static PyObject *py_minisat22_new(PyObject *self, PyObject *args)
{
	uint64_t reduce_base;
	int 	 gc_freq;

	if (!PyArg_ParseTuple(args, "Li", &reduce_base, &gc_freq))
		return NULL;

	Minisat22::Solver *s = new Minisat22::Solver(reduce_base);
	s->setGCFreq(static_cast<Minisat22::GCFreq>(gc_freq));

	if (s == NULL) {
		PyErr_SetString(PyExc_RuntimeError,
				"Cannot create a new solver.");
		return NULL;
	}

	return void_to_pyobj((void *)s);
}

// Function that set the python callback on the Oracle instace
//=============================================================================
static PyObject *py_minisat22_gc_oracle(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	PyObject *callback;
    int      gcPolicy;

    // Minisat version does not support other gcPolicies
	if (!PyArg_ParseTuple(args, "OiO:set_callback", &s_obj, &gcPolicy, &callback))
		return NULL;
	if (!PyCallable_Check(callback)) {
        PyErr_SetString(PyExc_TypeError, "parameter must be callable");
        return NULL;
    }

    // get pointer to solver
	Minisat22::Solver *s = (Minisat22::Solver *)pyobj_to_void(s_obj);

	// Add a reference to new callback
	Py_XINCREF(callback);

    // Dispose of previous oracle
    Minisat22::Oracle *oracle = s->getGCOracle();
    if (oracle)
    	delete oracle;

    // Create a new Oracle and set it on the solver instance
    oracle = new Minisat22::Oracle(s, callback);
    s->setGCOracle(oracle);

    /* Boilerplate to return "None" */
    // Py_INCREF(Py_None);
    // result = Py_None;
    // Conforming to the the way the rest of the interface returns null
    PyObject *ret = Py_BuildValue("");

	return ret;
}

// set the reduce_base attribute of Minisat
// Note that the reduce_base is now set on the instructor of Solver. This implementation
// is only kept for reference. The sizes of stats queues are init'ed according to
// reduce_base so if reduce_base is changed half-way through, those changes
// will not be reflected on the sizes of those queues.
//=============================================================================
static PyObject *py_minisat22_rbase(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	uint64_t reduce_base;

	if (!PyArg_ParseTuple(args, "OL", &s_obj, &reduce_base))
		return NULL;

	// get pointer to solver
	Minisat22::Solver *s = (Minisat22::Solver *)pyobj_to_void(s_obj);
	s->setReduceBase(reduce_base);

	PyObject *ret = Py_BuildValue("");
	return ret;
}

// auxiliary function for declaring new variables
//=============================================================================
static inline void minisat22_declare_vars(Minisat22::Solver *s, const int max_id)
{
	while (s->nVars() < max_id + 1)
		s->newVar();
}

//
//=============================================================================
static PyObject *py_minisat22_add_cl(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	PyObject *c_obj;

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &c_obj))
		return NULL;

	// get pointer to solver
	Minisat22::Solver *s = (Minisat22::Solver *)pyobj_to_void(s_obj);
	Minisat22::vec<Minisat22::Lit> cl;
	int vari = 0;

	// clause iterator
	PyObject *i_obj = PyObject_GetIter(c_obj);
	if (i_obj == NULL) {
		PyErr_SetString(PyExc_RuntimeError,
				"Clause does not seem to be an iterable object.");
		return NULL;
	}

	PyObject *l_obj;
	while ((l_obj = PyIter_Next(i_obj)) != NULL) {
		if (!pyint_check(l_obj)) {
			Py_DECREF(l_obj);
			Py_DECREF(i_obj);
			PyErr_SetString(PyExc_TypeError, "integer expected");
			return NULL;
		}

		int parsed_lit = pyint_to_cint(l_obj);
		Py_DECREF(l_obj);

		if (parsed_lit == 0) {
			Py_DECREF(i_obj);
			PyErr_SetString(PyExc_ValueError, "non-zero integer expected");
			return NULL;
		}

		vari = abs(parsed_lit)-1;
		while (vari >= s->nVars()) s->newVar();
		Minisat22::Lit lit2push = (parsed_lit > 0) ? Minisat22::mkLit(vari) : ~Minisat22::mkLit(vari);

		cl.push(lit2push);
	}

	Py_DECREF(i_obj);

	bool res = s->addClause(cl);

	PyObject *ret = PyBool_FromLong((long)res);
	return ret;
}

// A function that retunrs the list of problem clauses as a sparse matrix
//=============================================================================
static PyObject *py_minisat22_cl_arr(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	int b_obj;

	if (!PyArg_ParseTuple(args, "Oi", &s_obj, &b_obj))
		return NULL;

	// get pointer to solver
	Minisat22::Solver *s = (Minisat22::Solver *)pyobj_to_void(s_obj);
	Minisat22::Oracle *oracle = s->getGCOracle();

	bool learnts = b_obj!=0;
	return oracle->clause2SPArray(learnts);
}

// A function that retunrs the labels (features) for input/learnt clauses
//=============================================================================
static PyObject *py_minisat22_cl_labels(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	int b_obj;

	if (!PyArg_ParseTuple(args, "Oi", &s_obj, &b_obj))
		return NULL;

	// get pointer to solver
	Minisat22::Solver *s = (Minisat22::Solver *)pyobj_to_void(s_obj);
	Minisat22::Oracle *oracle = s->getGCOracle();

	bool learnts = b_obj!=0;
	return oracle->getClaLabels(learnts);
}

// A function that retunrs the labels (features) for all problem variables
//=============================================================================
static PyObject *py_minisat22_var_labels(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	Minisat22::Solver *s = (Minisat22::Solver *)pyobj_to_void(s_obj);
	Minisat22::Oracle *oracle = s->getGCOracle();

	return oracle->getVarLabels();
}

// A function that retunrs the global solver state (features)
//=============================================================================
static PyObject *py_minisat22_gss(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	Minisat22::Solver *s = (Minisat22::Solver *)pyobj_to_void(s_obj);
	Minisat22::Oracle *oracle = s->getGCOracle();

	return oracle->getGSS();
}

// A method to asynchronous termination of the current solving process
//=============================================================================
static PyObject *py_minisat22_terminate(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	Minisat22::Solver *s = (Minisat22::Solver *)pyobj_to_void(s_obj);
	s->interrupt();

	PyObject *ret = Py_BuildValue("");
	return ret;
}

// Return the current reward for the RL agent
//=============================================================================
static PyObject *py_minisat22_reward(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	Minisat22::Solver *s = (Minisat22::Solver *)pyobj_to_void(s_obj);
	uint64_t reward = s->getReward();

    PyObject *ret = PyLong_FromUnsignedLongLong(reward);
	return ret;
}

//
//=============================================================================
static PyObject *py_minisat22_solve(PyObject *self, PyObject *args)
{
	signal(SIGINT,  sigint_handler);
	signal(SIGSEGV, sigsegv_handler);

	PyObject *s_obj;
	PyObject *a_obj;  // assumptions

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &a_obj))
		return NULL;

	// get pointer to solver
	Minisat22::Solver *s = (Minisat22::Solver *)pyobj_to_void(s_obj);

	int size = (int)PyList_Size(a_obj);
	Minisat22::vec<Minisat22::Lit> a((int)size);

	int max_var = -1;
	for (int i = 0; i < size; ++i) {
		PyObject *l_obj = PyList_GetItem(a_obj, i);
		int l = pyint_to_cint(l_obj);
		a[i] = (l > 0) ? Minisat22::mkLit(l, false) : Minisat22::mkLit(-l, true);

		if (abs(l) > max_var)
			max_var = abs(l);
	}

	if (max_var > 0)
		minisat22_declare_vars(s, max_var);

	if (setjmp(env) != 0) {
		PyErr_SetString(SATError, "Caught keyboard interrupt");
		return NULL;
	}

	bool res = s->solve(a);

	PyObject *ret = PyBool_FromLong((long)res);
	return ret;
}

//
//=============================================================================
static PyObject *py_minisat22_solve_lim(PyObject *self, PyObject *args)
{
	signal(SIGINT, sigint_handler);

	PyObject *s_obj;
	PyObject *a_obj;  // assumptions

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &a_obj))
		return NULL;

	// get pointer to solver
	Minisat22::Solver *s = (Minisat22::Solver *)pyobj_to_void(s_obj);

	int size = (int)PyList_Size(a_obj);
	Minisat22::vec<Minisat22::Lit> a((int)size);

	int max_var = -1;
	for (int i = 0; i < size; ++i) {
		PyObject *l_obj = PyList_GetItem(a_obj, i);
		int l = pyint_to_cint(l_obj);
		a[i] = (l > 0) ? Minisat22::mkLit(l, false) : Minisat22::mkLit(-l, true);

		if (abs(l) > max_var)
			max_var = abs(l);
	}

	if (max_var > 0)
		minisat22_declare_vars(s, max_var);

	if (setjmp(env) != 0) {
		PyErr_SetString(SATError, "Caught keyboard interrupt");
		return NULL;
	}

	Minisat22::lbool res = s->solveLimited(a);

	PyObject *ret;
	if (res != Minisat22::lbool((uint8_t)2))  // l_Undef
		ret = PyBool_FromLong((long)!(Minisat22::toInt(res)));
	else
		ret = Py_BuildValue("");  // return Python's None if l_Undef

	return ret;
}

//
//=============================================================================
static PyObject *py_minisat22_propagate(PyObject *self, PyObject *args)
{
	signal(SIGINT, sigint_handler);

	PyObject *s_obj;
	PyObject *a_obj;  // assumptions
	int save_phases;

	if (!PyArg_ParseTuple(args, "OOi", &s_obj, &a_obj, &save_phases))
		return NULL;

	// get pointer to solver
	Minisat22::Solver *s = (Minisat22::Solver *)pyobj_to_void(s_obj);

	int size = (int)PyList_Size(a_obj);
	Minisat22::vec<Minisat22::Lit> a((int)size);

	int max_var = -1;
	for (int i = 0; i < size; ++i) {
		PyObject *l_obj = PyList_GetItem(a_obj, i);
		int l = pyint_to_cint(l_obj);
		a[i] = (l > 0) ? Minisat22::mkLit(l, false) : Minisat22::mkLit(-l, true);

		if (abs(l) > max_var)
			max_var = abs(l);
	}

	if (max_var > 0)
		minisat22_declare_vars(s, max_var);

	if (setjmp(env) != 0) {
		PyErr_SetString(SATError, "Caught keyboard interrupt");
		return NULL;
	}

	Minisat22::vec<Minisat22::Lit> p;
	bool res = s->prop_check(a, p, save_phases);

	PyObject *propagated = PyList_New(p.size());
	for (int i = 0; i < p.size(); ++i) {
		int l = Minisat22::var(p[i]) * (Minisat22::sign(p[i]) ? -1 : 1);
		PyObject *lit = pyint_from_cint(l);
		PyList_SetItem(propagated, i, lit);
	}

	PyObject *ret = Py_BuildValue("nO", (Py_ssize_t)res, propagated);
	Py_DECREF(propagated);

	return ret;
}

//
//=============================================================================
static PyObject *py_minisat22_setphases(PyObject *self, PyObject *args)
{
	signal(SIGINT, sigint_handler);

	PyObject *s_obj;
	PyObject *p_obj;  // polarities given as a list of integer literals

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &p_obj))
		return NULL;

	// get pointer to solver
	Minisat22::Solver *s = (Minisat22::Solver *)pyobj_to_void(s_obj);

	int size = (int)PyList_Size(p_obj);
	vector<int> p(size);

	int max_var = -1;
	for (int i = 0; i < size; ++i) {
		PyObject *l_obj = PyList_GetItem(p_obj, i);
		p[i] = pyint_to_cint(l_obj);

		if (abs(p[i]) > max_var)
			max_var = abs(p[i]);
	}

	if (max_var > 0)
		minisat22_declare_vars(s, max_var);

	for (int i = 0; i < size; ++i)
		s->setPolarity(abs(p[i]), p[i] < 0);

	PyObject *ret = Py_BuildValue("");
	return ret;
}

//
//=============================================================================
static PyObject *py_minisat22_cbudget(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	int64_t budget;

	if (!PyArg_ParseTuple(args, "Ol", &s_obj, &budget))
		return NULL;

	// get pointer to solver
	Minisat22::Solver *s = (Minisat22::Solver *)pyobj_to_void(s_obj);

	if (budget != 0 && budget != -1)  // it is 0 by default
		s->setConfBudget(budget);
	else
		s->budgetOff();

	PyObject *ret = Py_BuildValue("");
	return ret;
}

//
//=============================================================================
static PyObject *py_minisat22_pbudget(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	int64_t budget;

	if (!PyArg_ParseTuple(args, "Ol", &s_obj, &budget))
		return NULL;

	// get pointer to solver
	Minisat22::Solver *s = (Minisat22::Solver *)pyobj_to_void(s_obj);

	if (budget != 0 && budget != -1)  // it is 0 by default
		s->setPropBudget(budget);
	else
		s->budgetOff();

	PyObject *ret = Py_BuildValue("");
	return ret;
}

//
//=============================================================================
static PyObject *py_minisat22_core(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	Minisat22::Solver *s = (Minisat22::Solver *)pyobj_to_void(s_obj);

	Minisat22::vec<Minisat22::Lit> *c = &(s->conflict);  // minisat's conflict

	PyObject *core = PyList_New(c->size());
	for (int i = 0; i < c->size(); ++i) {
		int l = Minisat22::var((*c)[i]) * (Minisat22::sign((*c)[i]) ? 1 : -1);
		PyObject *lit = pyint_from_cint(l);
		PyList_SetItem(core, i, lit);
	}

	PyObject *ret = Py_None;

	if (c->size())
		ret = Py_BuildValue("O", core);

	Py_DECREF(core);
	return ret;
}

//
//=============================================================================
static PyObject *py_minisat22_model(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	Minisat22::Solver *s = (Minisat22::Solver *)pyobj_to_void(s_obj);

	// minisat's model
	Minisat22::vec<Minisat22::lbool> *m = &(s->model);

	if (m->size()) {
		// l_True fails to work
		Minisat22::lbool True = Minisat22::lbool((uint8_t)0);

		PyObject *model = PyList_New(m->size());
		for (int i = 0; i < m->size(); ++i) {
			int l = (i + 1) * ((*m)[i] == True ? 1 : -1);
			PyObject *lit = pyint_from_cint(l);
			PyList_SetItem(model, i, lit);
		}

		PyObject *ret = Py_BuildValue("O", model);
		Py_DECREF(model);
		return ret;
	}

	Py_RETURN_NONE;
}

//
//=============================================================================
static PyObject *py_minisat22_nof_vars(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	Minisat22::Solver *s = (Minisat22::Solver *)pyobj_to_void(s_obj);

	int nof_vars = s->nVars() - 1;  // 0 is a dummy variable

	PyObject *ret = Py_BuildValue("n", (Py_ssize_t)nof_vars);
	return ret;
}

//
//=============================================================================
static PyObject *py_minisat22_nof_cls(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	Minisat22::Solver *s = (Minisat22::Solver *)pyobj_to_void(s_obj);

	int nof_cls = s->nClauses();

	PyObject *ret = Py_BuildValue("n", (Py_ssize_t)nof_cls);
	return ret;
}

//
//=============================================================================
static PyObject *py_minisat22_del(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	Minisat22::Solver *s = (Minisat22::Solver *)pyobj_to_void(s_obj);

	delete s;

	PyObject *ret = Py_BuildValue("");
	return ret;
}

#endif  // WITH_MINISAT22







































// API for MiniSat from github
//=============================================================================
#ifdef WITH_MINISATGH
static PyObject *py_minisatgh_new(PyObject *self, PyObject *args)
{
	MinisatGH::Solver *s = new MinisatGH::Solver();

	if (s == NULL) {
		PyErr_SetString(PyExc_RuntimeError,
				"Cannot create a new solver.");
		return NULL;
	}

	return void_to_pyobj((void *)s);
}

// auxiliary function for declaring new variables
//=============================================================================
static inline void minisatgh_declare_vars(MinisatGH::Solver *s, const int max_id)
{
	while (s->nVars() < max_id + 1)
		s->newVar();
}

//
//=============================================================================
static PyObject *py_minisatgh_add_cl(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	PyObject *c_obj;

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &c_obj))
		return NULL;

	// get pointer to solver
	MinisatGH::Solver *s = (MinisatGH::Solver *)pyobj_to_void(s_obj);
	MinisatGH::vec<MinisatGH::Lit> cl;
	int max_var = -1;

	// clause iterator
	PyObject *i_obj = PyObject_GetIter(c_obj);
	if (i_obj == NULL) {
		PyErr_SetString(PyExc_RuntimeError,
				"Clause does not seem to be an iterable object.");
		return NULL;
	}

	PyObject *l_obj;
	while ((l_obj = PyIter_Next(i_obj)) != NULL) {
		if (!pyint_check(l_obj)) {
			Py_DECREF(l_obj);
			Py_DECREF(i_obj);
			PyErr_SetString(PyExc_TypeError, "integer expected");
			return NULL;
		}

		int l = pyint_to_cint(l_obj);
		Py_DECREF(l_obj);

		if (l == 0) {
			Py_DECREF(i_obj);
			PyErr_SetString(PyExc_ValueError, "non-zero integer expected");
			return NULL;
		}

		cl.push((l > 0) ? MinisatGH::mkLit(l, false) : MinisatGH::mkLit(-l, true));

		if (abs(l) > max_var)
			max_var = abs(l);
	}

	Py_DECREF(i_obj);

	if (max_var > 0)
		minisatgh_declare_vars(s, max_var);

	bool res = s->addClause(cl);

	PyObject *ret = PyBool_FromLong((long)res);
	return ret;
}

//
//=============================================================================
static PyObject *py_minisatgh_solve(PyObject *self, PyObject *args)
{
	signal(SIGINT, sigint_handler);

	PyObject *s_obj;
	PyObject *a_obj;  // assumptions

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &a_obj))
		return NULL;

	// get pointer to solver
	MinisatGH::Solver *s = (MinisatGH::Solver *)pyobj_to_void(s_obj);

	int size = (int)PyList_Size(a_obj);
	MinisatGH::vec<MinisatGH::Lit> a((int)size);

	int max_var = -1;
	for (int i = 0; i < size; ++i) {
		PyObject *l_obj = PyList_GetItem(a_obj, i);
		int l = pyint_to_cint(l_obj);
		a[i] = (l > 0) ? MinisatGH::mkLit(l, false) : MinisatGH::mkLit(-l, true);

		if (abs(l) > max_var)
			max_var = abs(l);
	}

	if (max_var > 0)
		minisatgh_declare_vars(s, max_var);

	if (setjmp(env) != 0) {
		PyErr_SetString(SATError, "Caught keyboard interrupt");
		return NULL;
	}

	bool res = s->solve(a);

	PyObject *ret = PyBool_FromLong((long)res);
	return ret;
}

//
//=============================================================================
static PyObject *py_minisatgh_solve_lim(PyObject *self, PyObject *args)
{
	signal(SIGINT, sigint_handler);

	PyObject *s_obj;
	PyObject *a_obj;  // assumptions

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &a_obj))
		return NULL;

	// get pointer to solver
	MinisatGH::Solver *s = (MinisatGH::Solver *)pyobj_to_void(s_obj);

	int size = (int)PyList_Size(a_obj);
	MinisatGH::vec<MinisatGH::Lit> a((int)size);

	int max_var = -1;
	for (int i = 0; i < size; ++i) {
		PyObject *l_obj = PyList_GetItem(a_obj, i);
		int l = pyint_to_cint(l_obj);
		a[i] = (l > 0) ? MinisatGH::mkLit(l, false) : MinisatGH::mkLit(-l, true);

		if (abs(l) > max_var)
			max_var = abs(l);
	}

	if (max_var > 0)
		minisatgh_declare_vars(s, max_var);

	if (setjmp(env) != 0) {
		PyErr_SetString(SATError, "Caught keyboard interrupt");
		return NULL;
	}

	MinisatGH::lbool res = s->solveLimited(a);

	PyObject *ret;
	if (res != MinisatGH::lbool((uint8_t)2))  // l_Undef
		ret = PyBool_FromLong((long)!(MinisatGH::toInt(res)));
	else
		ret = Py_BuildValue("");  // return Python's None if l_Undef

	return ret;
}

//
//=============================================================================
static PyObject *py_minisatgh_propagate(PyObject *self, PyObject *args)
{
	signal(SIGINT, sigint_handler);

	PyObject *s_obj;
	PyObject *a_obj;  // assumptions
	int save_phases;

	if (!PyArg_ParseTuple(args, "OOi", &s_obj, &a_obj, &save_phases))
		return NULL;

	// get pointer to solver
	MinisatGH::Solver *s = (MinisatGH::Solver *)pyobj_to_void(s_obj);

	int size = (int)PyList_Size(a_obj);
	MinisatGH::vec<MinisatGH::Lit> a((int)size);

	int max_var = -1;
	for (int i = 0; i < size; ++i) {
		PyObject *l_obj = PyList_GetItem(a_obj, i);
		int l = pyint_to_cint(l_obj);
		a[i] = (l > 0) ? MinisatGH::mkLit(l, false) : MinisatGH::mkLit(-l, true);

		if (abs(l) > max_var)
			max_var = abs(l);
	}

	if (max_var > 0)
		minisatgh_declare_vars(s, max_var);

	if (setjmp(env) != 0) {
		PyErr_SetString(SATError, "Caught keyboard interrupt");
		return NULL;
	}

	MinisatGH::vec<MinisatGH::Lit> p;
	bool res = s->prop_check(a, p, save_phases);

	PyObject *propagated = PyList_New(p.size());
	for (int i = 0; i < p.size(); ++i) {
		int l = MinisatGH::var(p[i]) * (MinisatGH::sign(p[i]) ? -1 : 1);
		PyObject *lit = pyint_from_cint(l);
		PyList_SetItem(propagated, i, lit);
	}

	PyObject *ret = Py_BuildValue("nO", (Py_ssize_t)res, propagated);
	Py_DECREF(propagated);

	return ret;
}

//
//=============================================================================
static PyObject *py_minisatgh_setphases(PyObject *self, PyObject *args)
{
	signal(SIGINT, sigint_handler);

	PyObject *s_obj;
	PyObject *p_obj;  // polarities given as a list of integer literals

	if (!PyArg_ParseTuple(args, "OO", &s_obj, &p_obj))
		return NULL;

	// get pointer to solver
	MinisatGH::Solver *s = (MinisatGH::Solver *)pyobj_to_void(s_obj);

	int size = (int)PyList_Size(p_obj);
	vector<int> p(size);

	int max_var = -1;
	for (int i = 0; i < size; ++i) {
		PyObject *l_obj = PyList_GetItem(p_obj, i);
		p[i] = pyint_to_cint(l_obj);

		if (abs(p[i]) > max_var)
			max_var = abs(p[i]);
	}

	if (max_var > 0)
		minisatgh_declare_vars(s, max_var);

	for (int i = 0; i < size; ++i)
		s->setPolarity(abs(p[i]), MinisatGH::lbool(p[i] < 0));

	PyObject *ret = Py_BuildValue("");
	return ret;
}

//
//=============================================================================
static PyObject *py_minisatgh_cbudget(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	int64_t budget;

	if (!PyArg_ParseTuple(args, "Ol", &s_obj, &budget))
		return NULL;

	// get pointer to solver
	MinisatGH::Solver *s = (MinisatGH::Solver *)pyobj_to_void(s_obj);

	if (budget != 0 && budget != -1)  // it is 0 by default
		s->setConfBudget(budget);
	else
		s->budgetOff();

	PyObject *ret = Py_BuildValue("");
	return ret;
}

//
//=============================================================================
static PyObject *py_minisatgh_pbudget(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	int64_t budget;

	if (!PyArg_ParseTuple(args, "Ol", &s_obj, &budget))
		return NULL;

	// get pointer to solver
	MinisatGH::Solver *s = (MinisatGH::Solver *)pyobj_to_void(s_obj);

	if (budget != 0 && budget != -1)  // it is 0 by default
		s->setPropBudget(budget);
	else
		s->budgetOff();

	PyObject *ret = Py_BuildValue("");
	return ret;
}

//
//=============================================================================
static PyObject *py_minisatgh_core(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	MinisatGH::Solver *s = (MinisatGH::Solver *)pyobj_to_void(s_obj);

	MinisatGH::LSet *c = &(s->conflict);  // minisat's conflict

	PyObject *core = PyList_New(c->size());
	for (int i = 0; i < c->size(); ++i) {
		int l = MinisatGH::var((*c)[i]) * (MinisatGH::sign((*c)[i]) ? 1 : -1);
		PyObject *lit = pyint_from_cint(l);
		PyList_SetItem(core, i, lit);
	}

	PyObject *ret = Py_None;

	if (c->size())
		ret = Py_BuildValue("O", core);

	Py_DECREF(core);
	return ret;
}

//
//=============================================================================
static PyObject *py_minisatgh_model(PyObject *self, PyObject *args)
{
	PyObject *s_obj;
	PyObject *ret = Py_None;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	MinisatGH::Solver *s = (MinisatGH::Solver *)pyobj_to_void(s_obj);

	// minisat's model
	MinisatGH::vec<MinisatGH::lbool> *m = &(s->model);

	if (m->size()) {
		// l_True fails to work
		MinisatGH::lbool True = MinisatGH::lbool((uint8_t)0);

		PyObject *model = PyList_New(m->size() - 1);
		for (int i = 1; i < m->size(); ++i) {
			int l = i * ((*m)[i] == True ? 1 : -1);
			PyObject *lit = pyint_from_cint(l);
			PyList_SetItem(model, i - 1, lit);
		}

		ret = Py_BuildValue("O", model);
		Py_DECREF(model);
	}

	return ret;
}

//
//=============================================================================
static PyObject *py_minisatgh_nof_vars(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	MinisatGH::Solver *s = (MinisatGH::Solver *)pyobj_to_void(s_obj);

	int nof_vars = s->nVars() - 1;  // 0 is a dummy variable

	PyObject *ret = Py_BuildValue("n", (Py_ssize_t)nof_vars);
	return ret;
}

//
//=============================================================================
static PyObject *py_minisatgh_nof_cls(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	MinisatGH::Solver *s = (MinisatGH::Solver *)pyobj_to_void(s_obj);

	int nof_cls = s->nClauses();

	PyObject *ret = Py_BuildValue("n", (Py_ssize_t)nof_cls);
	return ret;
}

//
//=============================================================================
static PyObject *py_minisatgh_del(PyObject *self, PyObject *args)
{
	PyObject *s_obj;

	if (!PyArg_ParseTuple(args, "O", &s_obj))
		return NULL;

	// get pointer to solver
	MinisatGH::Solver *s = (MinisatGH::Solver *)pyobj_to_void(s_obj);

	delete s;

	PyObject *ret = Py_BuildValue("");
	return ret;
}
#endif  // WITH_MINISATGH

}  // extern "C"
