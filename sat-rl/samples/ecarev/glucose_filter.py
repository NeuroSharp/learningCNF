import os
import logging

"""
Filters the SAT instance according to a set of constrainst:
    steps_min
    time_min
    time_max

sample usage:
>>> # import GlucoseFilter
>>> file_path = # path to the cnf file
>>> time_max = 10
>>> filter = GlucoseFilter()
>>>
>>> glucose = Glucose3(gc_freq="fixed", use_timer=True, reduce_base=reduce_base)
>>> glucose.append_formula(file_path)
>>> glucose.time_budget(time_max)
>>> res = glucose.solve_limited()
>>> filter.filter(sharpSAT, cnf_id)
"""
class GlucoseFilter:
    def __init__(self, steps_min = 4, steps_max = 100, op_cnt_max = 1e10, time_max = 10):
        self.steps_min = steps_min
        self.steps_max = steps_max
        self.op_cnt_max = op_cnt_max
        self.time_max = time_max

        self.log = logging.getLogger("glucose")

    def filter(self, glucose, cnf_id=None):
        message = None
        res = False
        if (self.op_cnt_max < glucose.reward()):
            message = f"{cnf_id}: Too hard: op_cnt > {self.op_cnt_max}!"
            res = True
        if (glucose.nof_gc() < self.steps_min):
            message = f"{cnf_id}: Degenerate: nof_gc < {self.steps_min}!"
            res = True
        if (self.steps_max < glucose.nof_gc()):
            message = f"{cnf_id}: Too many steps: nof_gc > {self.steps_max}!"
            res = True
        if (self.time_max <= glucose.time()):
            message = f"{cnf_id}: Too hard: Time > {self.time_max}!"
            res = True

        if (res):
            message += f" (op_cnt/step/time/pid: {glucose.reward()}/{glucose.nof_gc()}/{glucose.time():.2f}/{os.getpid()})"
            self.log.info(message)

        return res
