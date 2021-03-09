#ifndef Minisat22_Oracle_h
#define Minisat22_Oracle_h

// #define PY_ARRAY_UNIQUE_SYMBOL PYSAT_ARRAY_API
// #define NO_IMPORT_ARRAY

#ifdef DEBUGMODE
#include <stdio.h>
#endif

#include <Python.h>
#include <numpy/arrayobject.h>

#include <assert.h>

#include "minisat22/core/Solver.h"
#include "minisat22/core/SolverTypes.h"
#include "minisat22/mtl/Sort.h"


namespace Minisat22 {

class Oracle
{
private:
    Solver   *solver   = NULL;
    PyObject *callback = NULL;
    ClauseAllocator &ca;

public:
    Oracle(Solver *solver, PyObject *callback) :
        solver(solver),
        callback(callback),
        ca(solver->ca)
    {}

    virtual ~Oracle() {
        Py_XDECREF(callback);
    }

    void      setSolver (Solver   *psolver) { solver = psolver; }
    Solver*   getSolver () { return solver; }

    // take the field to sort by in the constructor
    void      setCallback (PyObject *pycallback) { callback = pycallback; }
    PyObject *getCallback () { return callback; }

    void      reduceDBDelegate ()
    {
        /* Ugly hack! The import_array() cannot be shared between translation units
        * so even though we can call import_array() from pysolvers.cc, it would not be
        * visible here because this function is being called from another translation unit
        * so we get a segfault.
        *
        * Some workarounds have been proposed but none seems to work. Involving NO_IMPORT_ARRAY
        *
        */
        if (!PyArray_API) { // Make sure to call import_array only once
            _import_array();
        }

#ifdef DEBUGMODE
        FILE * pFile;
        pFile = fopen ("_minisat.tmp","w");
#endif

        vec<CRef> &learnts  = solver->learnts;
        int nLearnts_cl  = solver->nLearnts();
        int nLearnts_lit = solver->learnts_literals;

        // 1. Update the reduction frequency
        if (solver->gc_freq == GCFreq::GLUCOSE)
            solver->handleGCFreq();
        else if (solver->gc_freq == GCFreq::UTILITY) {
            sort(learnts, utility_lt(ca));

            // NOTE: the values are taken straight from the glucose, use different numbers/logic here...
            if (ca[learnts[learnts.size() / 2]].utility() <= 3) solver->reduce_base += 1000;
            if (ca[learnts.last()].utility() > 5) solver->reduce_base -= 1000;
        } // else fixed frequency

        // 2. Building the array containing the Clause labels
        int cl_lable_size = 6; // num_used, size, lbd, activity, locked
        npy_intp cl_dims[] = { static_cast<npy_intp>(nLearnts_cl),
                               static_cast<npy_intp>(cl_lable_size)};
        PyArrayObject *cl_label_arr =
                            (PyArrayObject*) PyArray_ZEROS(2, cl_dims, NPY_DOUBLE, 0);

        // 3. Building the arrays for the numpy sparse matrix
        // representation of the adjacency matrix
        npy_intp adj_dims[] = { static_cast<npy_intp>(nLearnts_lit)};
        PyArrayObject *rows_arr = (PyArrayObject*) PyArray_ZEROS(1, adj_dims, NPY_INT, 0);
        PyArrayObject *cols_arr = (PyArrayObject*) PyArray_ZEROS(1, adj_dims, NPY_INT, 0);
        PyArrayObject *data_arr = (PyArrayObject*) PyArray_ZEROS(1, adj_dims, NPY_INT, 0);
        if (cl_label_arr == NULL ||
            rows_arr     == NULL ||
            cols_arr     == NULL ||
            data_arr     == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Couldn't build numpy arrays.");

            // TODO: Check if we can fail more gracefully instead of exiting.
            exit(1);
        }

        // 4. Populating the arrays
        int i, j, lits_count;
        for (i = j = lits_count = 0; i < nLearnts_cl; i++){
            Clause& c = ca[learnts[i]];

#ifdef DEBUGMODE
            printClause(pFile, c);
#endif

            double *v = (double*)PyArray_GETPTR2(cl_label_arr, i, j++);
            *v = i; // index
            v = (double*)PyArray_GETPTR2(cl_label_arr, i, j++);
            *v = c.num_used(); //num_used
            v = (double*)PyArray_GETPTR2(cl_label_arr, i, j++);
            *v = c.size(); //size
            v = (double*)PyArray_GETPTR2(cl_label_arr, i, j++);
            *v = c.lbd(); //lbd
            v = (double*)PyArray_GETPTR2(cl_label_arr, i, j++);
            *v = c.activity(); //activity
            v = (double*)PyArray_GETPTR2(cl_label_arr, i, j++);
            *v = solver->locked(c); //locked

            j = 0;

            for (int k = 0; k < c.size(); k++) {
                assert(lits_count<nLearnts_lit);
                int *rr = (int*)PyArray_GETPTR1(rows_arr, lits_count);
                int *cc = (int*)PyArray_GETPTR1(cols_arr, lits_count);
                int *dd = (int*)PyArray_GETPTR1(data_arr, lits_count);
                *rr = i;
                // We keep the variable convention of minisat in that
                // the variable indices start from 0 and not 1
                *cc = var (c[k]);
                *dd = sign(c[k])?-1:1;

                lits_count++;
            }
        }

#ifdef DEBUGMODE
        fclose (pFile);
#endif

        // 5. Calling the python callback
        PyObject *arglist = Py_BuildValue("(OOOO)", cl_label_arr, rows_arr, cols_arr, data_arr);
        // printf("Calling python callback...\n");
        PyObject *result = PyEval_CallObject(callback, arglist);
        PyArrayObject *res_array = NULL;
        if (result)
            res_array = (PyArrayObject*) PyArray_FROM_OTF(result, NPY_BOOL, NPY_IN_ARRAY);

        // 6. Check to see if terminate() was called in the callback...
        if (!solver->withinBudget()){
            Py_XDECREF(cl_label_arr);
            Py_XDECREF(rows_arr);
            Py_XDECREF(cols_arr);
            Py_XDECREF(data_arr);
            Py_XDECREF(arglist);
            Py_XDECREF(result);
            Py_XDECREF(res_array);

            return;
        }

        // 7. Check for erros in the callback
        if (result == NULL || res_array == NULL) {
            /* cannot return NULL... */
            printf("Error in callback...\n");

            // TODO: Check if we can fail more gracefully instead of exiting.
            exit(1);
        }

        // 8. Call the one of the clause reduction procedure
        if (res_array->dimensions[0] == 0) { // LBD-based
            if (solver->gc_freq == GCFreq::UTILITY) {
                // The callback must return utilities...
                printf("Error: The GC callback cannot return an empty array while gc_freq is set to 'UTILITY'.\n");

                // TODO: Check if we can fail more gracefully instead of exiting.
                exit(1);
            }

            // Sorting and frequency adjustment is done before so just call solver's reduceDB().
            solver->reduceDBDefault();
        } else                               // RL-based
            reduceDB(res_array);

        // 9. Clean up
        Py_XDECREF(cl_label_arr);
        Py_XDECREF(rows_arr);
        Py_XDECREF(cols_arr);
        Py_XDECREF(data_arr);
        Py_XDECREF(arglist);
        Py_XDECREF(result);
        Py_XDECREF(res_array);
    }

    // TODO: Move this out of Oracle to another friend class of Solver
    // This function is independent of clause reduction, so it should
    // be callable even if a gcOracle is not set on the Solver.
    // TODO: utilize the learnts_info flag to choose whether to send the info for
    // learnt or input clauses
    PyObject *clause2SPArray(bool learnts_info)
    {
        if (!PyArray_API) { // Make sure to call import_array only once
            _import_array();
        }

        vec<CRef> &cl_list = learnts_info ? solver->learnts : solver->clauses;
        int n_cl           = learnts_info ? solver->nLearnts() : solver->nClauses();
        int n_lits         = learnts_info ? solver->learnts_literals : solver->nLits();

        npy_intp adj_dims[] = { static_cast<npy_intp>(n_lits)};

        PyArrayObject *rows_arr = (PyArrayObject*) PyArray_ZEROS(1, adj_dims, NPY_INT, 0);
        PyArrayObject *cols_arr = (PyArrayObject*) PyArray_ZEROS(1, adj_dims, NPY_INT, 0);
        PyArrayObject *data_arr = (PyArrayObject*) PyArray_ZEROS(1, adj_dims, NPY_INT, 0);
        if (rows_arr    == NULL ||
            cols_arr    == NULL ||
            data_arr    == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Couldn't build numpy arrays.");

            /* cannot return NULL... */
            // TODO: Check if we can fail more gracefully instead of exiting.
            // Py_XDECREF(rows_arr);
            // Py_XDECREF(cols_arr);
            // Py_XDECREF(data_arr);

            exit(1);
        }

        int i, lits_count;
        for (i = lits_count = 0; i < n_cl; i++){
            Clause &c = ca[cl_list[i]];
            for (int j = 0; j < c.size(); j++) {
                assert(lits_count<n_lits);
                int *rr = (int*)PyArray_GETPTR1(rows_arr, lits_count);
                int *cc = (int*)PyArray_GETPTR1(cols_arr, lits_count);
                int *dd = (int*)PyArray_GETPTR1(data_arr, lits_count);
                *rr = i;
                // We keep the variable convention of minisat in that
                // the variable indices start from 0 and not 1
                *cc = var (c[j]);
                *dd = sign(c[j])?-1:1;

                lits_count++;
            }
        }

        PyObject *ret = Py_BuildValue("(OOO)", rows_arr, cols_arr, data_arr);

        // Clean up
        Py_XDECREF(rows_arr);
        Py_XDECREF(cols_arr);
        Py_XDECREF(data_arr);

        return ret;
    }

    // TODO: Move this out of Oracle to another friend class of Solver
    // This function is independent of clause reduction, so it should
    // be callable even if a gcOracle is not set on the Solver.
    PyObject *getVarLabels()
    {
        if (!PyArray_API) { // Make sure to call import_array only once
            _import_array();
        }

        int nVars  = solver->nVars();

        // 1. Building the array containing the Variable labels
        int var_lable_size = 6;
        npy_intp cl_dims[] = { static_cast<npy_intp>(nVars),
                               static_cast<npy_intp>(var_lable_size)};
        PyArrayObject *var_label_arr =
                            (PyArrayObject*) PyArray_ZEROS(2, cl_dims, NPY_DOUBLE, 0);
        if (var_label_arr == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Couldn't build numpy arrays.");

            // TODO: Check if we can fail more gracefully instead of exiting.
            exit(1);
        }

        // 2. Populating the array
        // Start from index 1 as the variables start from 1
        // (In Minisat var 0 is a dummy)
        int i, j;
        for (i = 1, j = 0; i < nVars; i++){
            double *v = (double*)PyArray_GETPTR2(var_label_arr, i, j++);
            *v = isWatch(i); // is there a watch literal form this var?
            v = (double*)PyArray_GETPTR2(var_label_arr, i, j++);
            *v = toInt(solver->value(i)); // current assignement (lbool: T, F or Undef)
            v = (double*)PyArray_GETPTR2(var_label_arr, i, j++);
            *v = solver->level(i); // decision level at which the variable was assigned
            v = (double*)PyArray_GETPTR2(var_label_arr, i, j++);
            *v = solver->activity[i]; // activity (avergae for a clasue) TODO
            v = (double*)PyArray_GETPTR2(var_label_arr, i, j++);
            *v = solver->polarity[i]; // polarity
            v = (double*)PyArray_GETPTR2(var_label_arr, i, j++);
            *v = solver->value(i) != l_Undef && solver->level(i) == 0; // is var forced?

            j = 0;
        }

        // 3. Build the label array
        PyObject *ret = Py_BuildValue("O", var_label_arr);

        // 4. Clean up
        Py_XDECREF(var_label_arr);

        return ret;
    }

    // TODO: Move this out of Oracle to another friend class of Solver
    // This function is independent of clause reduction, so it should
    // be callable even if a gcOracle is not set on the Solver.
    // TODO: utilize the learnts_info flag to choose wether to send the info for
    // learnt or input clauses
    PyObject *getClaLabels(bool learnts_info)
    {
        if (!PyArray_API) { // Make sure to call import_array only once
            _import_array();
        }

        int nClauses = solver->nClauses();

        // 1. Building the array containing the Variable labels
        int cl_lable_size = 6; // num_used, size, lbd, activity, locked
        npy_intp cl_dims[] = { static_cast<npy_intp>(nClauses),
                               static_cast<npy_intp>(cl_lable_size)};
        PyArrayObject *cl_label_arr =
                            (PyArrayObject*) PyArray_ZEROS(2, cl_dims, NPY_DOUBLE, 0);
        if (cl_label_arr == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Couldn't build numpy arrays.");

            // TODO: Check if we can fail more gracefully instead of exiting.
            exit(1);
        }

        // 2. Populating the array
        int i, j;
        for (i = j = 0; i < nClauses; i++){
            Clause& c = ca[solver->clauses[i]];
            double *v = (double*)PyArray_GETPTR2(cl_label_arr, i, j++);
            *v = i; // index
            v = (double*)PyArray_GETPTR2(cl_label_arr, i, j++);
            *v = c.num_used(); //num_used (make it temporal)
            v = (double*)PyArray_GETPTR2(cl_label_arr, i, j++);
            *v = c.size(); //size
            v = (double*)PyArray_GETPTR2(cl_label_arr, i, j++);
            *v = c.lbd(); //lbd
            v = (double*)PyArray_GETPTR2(cl_label_arr, i, j++);
            *v = c.activity(); //activity
            v = (double*)PyArray_GETPTR2(cl_label_arr, i, j++);
            *v = solver->locked(c); //locked

            j = 0;
        }

        // 3. Build the label array
        PyObject *ret = Py_BuildValue("O", cl_label_arr);

        // 4. Clean up
        Py_XDECREF(cl_label_arr);

        return ret;
    }

    // TODO: Move this out of Oracle to another friend class of Solver
    // This function is independent of clause reduction, so it should
    // be callable even if a gcOracle is not set on the Solver.
    /* Returns the global solver state */
    PyObject *getGSS()
    {
        if (!PyArray_API) { // Make sure to call import_array only once
            _import_array();
        }

        // 1. Building the array containing the solver state features
        int state_size = 3;
        npy_intp dims[] = { static_cast<npy_intp>(state_size) };
        PyArrayObject *solver_state =
                            (PyArrayObject*) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
        if (solver_state == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Couldn't build numpy arrays.");

            // TODO: Check if we can fail more gracefully instead of exiting.
            exit(1);
        }

        int    i = 0, j = 0;
        int    nWindows = solver->t_multiple * solver->t_fraction;
        int    lbd_dim  = solver->lbd_range;
        double nVars_d = solver->nVars();
        double nClause_d = solver->nClauses();
        double nLearnts_d = solver->nLearnts();

        // 2. Regular Features
        // # of learnt clauses (normalized)
        double *v = (double*)PyArray_GETPTR1(solver_state, i++);
        *v = nLearnts_d / nClause_d;

        // # of conflicts (normalized)
        v = (double*)PyArray_GETPTR1(solver_state, i++);
        *v = solver->conflicts / nVars_d;

        // Global Avg LBD
        v = (double*)PyArray_GETPTR1(solver_state, i++);
        *v = solver->global_lbd_sum / nLearnts_d;


        // 3. Temporal Features
        *dims = { static_cast<npy_intp>(nWindows) };
        // Avg trail size in a window
        PyArrayObject *trail_sz = (PyArrayObject*) PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, solver->trail_sz_agg.data());
        // Avg decision level in a window
        PyArrayObject *decl_cnt = (PyArrayObject*) PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, solver->decl_cnt_agg.data());
        // Avg vars in decision level in a window
        PyArrayObject *ts_dlc   = (PyArrayObject*) PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, solver->ts_dlc_agg.data());

        npy_intp lbd_dims[] = { static_cast<npy_intp>(lbd_dim),
                                static_cast<npy_intp>(nWindows)};
        // Percentage of LBDs of a category in total per window
        PyArrayObject *lbd_queue = (PyArrayObject*) PyArray_ZEROS(2, lbd_dims, NPY_DOUBLE, 0);
        for (i = 0; i < lbd_dim; i++) {
            std::deque<double> q = solver->lbd_queue_agg[i].q;
            typename std::deque<double>::iterator it = q.begin();

            while (it != q.end()){
                v = (double*)PyArray_GETPTR2(lbd_queue, i, j++);
                *v = *it++;
            }

            j = 0;
        }

        // setting this flags so that the memory is freed when the object is out of scope
        PyArray_ENABLEFLAGS(trail_sz,  NPY_ARRAY_OWNDATA);
        PyArray_ENABLEFLAGS(decl_cnt,  NPY_ARRAY_OWNDATA);
        PyArray_ENABLEFLAGS(ts_dlc,    NPY_ARRAY_OWNDATA);

        if (trail_sz  == NULL ||
            decl_cnt  == NULL ||
            ts_dlc    == NULL ||
            lbd_queue == NULL) { // TODO: Check all the other arrays here
            PyErr_SetString(PyExc_RuntimeError, "Couldn't build numpy arrays.");

            // TODO: Check if we can fail more gracefully instead of exiting.
            exit(1);
        }

        // 4. Build the return value
        PyObject *ret = Py_BuildValue("(OOOOO)",
            solver_state, trail_sz, decl_cnt, ts_dlc, lbd_queue);

        // 5. Clean up
        Py_XDECREF(solver_state);
        Py_XDECREF(trail_sz );
        Py_XDECREF(decl_cnt );
        Py_XDECREF(ts_dlc   );
        Py_XDECREF(lbd_queue);

        return ret;
    }


private:
    bool isWatch(Var v) {
        bool isWatch = false;
        if (solver->watches[mkLit(v, false)].size() > 0 ||
                solver->watches[mkLit(v, true)].size() > 0)
            isWatch = true;

        return isWatch;
    }

    void reduceDB(PyArrayObject *res_array)
    {
        int     i, j;
        vec<CRef> &learnts  = solver->learnts;

        assert(res_array->dimensions[0] == learnts.size());
        for (i = j = 0; i < learnts.size(); i++){
            bool *v = (bool*) PyArray_GETPTR1(res_array, i);

            if (!*v){
                // RL should not ask to delete a locked clause...
                assert(!solver->locked(ca[learnts[i]]));

                solver->removeClause(learnts[i]);
            }
            else
                learnts[j++] = learnts[i];
        }

        learnts.shrink(i - j);
        solver->checkGarbage();
    }


    struct utility_lt {
        ClauseAllocator& ca;
        utility_lt(ClauseAllocator& ca_) : ca(ca_) {}
        bool operator () (CRef x, CRef y) {
            if (ca[x].size() == 2) return 0;
            if (ca[y].size() == 2) return 1;

            if (ca[x].utility() > ca[y].utility()) return 1;
            if (ca[x].utility() < ca[y].utility()) return 0;

            return ca[x].activity() < ca[y].activity();
        }
    };

    // The following is the code snippet for the utility-based reduction.
    // The Rl side is expected to send back real-valued "utility" for clauses.

    // void reduceDB()
    // {
    //     int     i, j;
    //     vec<CRef> &learnts  = solver->learnts;

    //     int limit = learnts.size() / 2;
    //     for (i = j = 0; i < learnts.size(); i++){
    //         Clause& c = ca[learnts[i]];
    //         // TODO: Maybe check for c.lbd() > 2 ?
    //         if (c.size() > 2 && c.utility() > 2 && c.removable() && !solver->locked(c) && i < limit)
    //             solver->removeClause(learnts[i]);
    //         else{
    //             if (!c.removable()) limit++;
    //             c.removable(true);
    //             learnts[j++] = learnts[i]; }
    //     }

    //     learnts.shrink(i - j);
    //     solver->checkGarbage();
    // }

#ifdef DEBUGMODE
    void printClause(FILE *pFile, const Clause& c) const {
        fprintf(pFile, "%d,%d,%d,%.3f", c.num_used(), c.size(), c.lbd(), c.activity());//, c.num_used());

        fprintf(pFile, "[");
        vec<Lit> lits;
        for (int i=0; i < c.size(); i++)
            lits.push(c[i]);
        sort(lits);
        fprintf(pFile, "%s%d", sign(lits[0]) ? "-" : "", var(lits[0])+1);
        for (int i=1; i < lits.size(); i++)
            fprintf(pFile, ",%s%d", sign(lits[i]) ? "-" : "", var(lits[i])+1);
        fprintf(pFile, "]\n");
    }
#endif
};

}
#endif
