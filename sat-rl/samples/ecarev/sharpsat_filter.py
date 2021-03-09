import os
import logging

"""
Filters the shaprSAT instance according to a set of constrainst:
    steps_min
    time_min
    time_max

sample usage:
>>> # import SharpSATFilter
>>> file_path = # path to the cnf file
>>> time_max = 2
>>> filter = SharpSATFilter()
>>>
>>> sharpSAT = SharpSAT(time_budget = time_max, use_timer= True).solve(file_path)
>>> filter.filter(sharpSAT, cnf_id)
"""
class SharpSATFilter:
    def __init__(self, steps_min = 30, time_min = 0.15, time_max = 2):
        self.steps_min = steps_min
        self.time_min = time_min
        self.time_max = time_max

        self.log = logging.getLogger("sharpSAT")

    def filter(self, sharpSAT, cnf_id = "CNF_ID"):
        message = ""
        res = False
        if (sharpSAT.reward() < self.steps_min):
            message = f"{cnf_id}: Too easy! Steps < {self.steps_min}"
            res = True
        if (sharpSAT.time() < self.time_min):
            message = f"{cnf_id}: Too easy! Time < {self.time_min}s"
            res = True
        if (self.time_max <= sharpSAT.time()):
            message = f"{cnf_id}: Too hard! Time > {self.time_max}s"
            res = True

        if (res):
            message += f" (step/time/pid: {sharpSAT.reward()}/{sharpSAT.time():.2f}/{os.getpid()})"
            self.log.info(message)

        return res
