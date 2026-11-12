import gurobipy as gp
import pandas as pd
import pickle
from src.config import *
import src.solver as solver


objectives = ['utilitarian', 'maximin']

for objective in objectives:

    with open(
            '{0}/results/instances/instance_{1}_population.pkl'.format(
                RELPATH, FILENAME
            ), 'rb'
    ) as file:
        instance = pickle.load(file)

    timeLimit = 300
    epsLimit = 1
    iterLimit = 10
    modelname = 'population-{0}-{1}-{2}'.format(objective, timeLimit, epsLimit)

    solver.main(
        instance, modelname,
        objective=objective,
        timeLimit=timeLimit,
        epsLimit=epsLimit,
        iterLimit=iterLimit
    )
