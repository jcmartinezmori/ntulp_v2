import gurobipy as gp
import pandas as pd
import pickle
from src.config import *
import src.solver as solver


convergence_names = ['0', '1', '2']
objectives = ['maximin', 'utilitarian']

for convergence_name in convergence_names:
    for objective in objectives:

        with open(
                '{0}/results/instances/instance_{1}_convergence-{2}.pkl'.format(
                    RELPATH, FILENAME, convergence_name
                ), 'rb'
        ) as file:
            instance = pickle.load(file)

        timeLimit = 300
        epsLimit = 1
        iterLimit = 20
        modelname = 'convergence-{0}-{1}-{2}-{3}'.format(convergence_name, objective, timeLimit, epsLimit)

        solver.main(
            instance, modelname,
            objective=objective,
            timeLimit=timeLimit,
            epsLimit=epsLimit,
            iterLimit=iterLimit
        )


