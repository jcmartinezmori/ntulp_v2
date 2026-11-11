import gurobipy as gp
import pickle
from src.config import *

n = 'community-area'

with open('{0}/results/instances/instance_{1}_{2}.pkl'.format(RELPATH, FILENAME, n), 'rb') as file:
    instance = pickle.load(file)

N, J, K, A, B, V = instance

m = gp.Model()

m._x = m.addVars(J, lb=0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name='x')
m._u = m.addVars(N, lb=0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name='u')
m._logu = m.addVars(N, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name='logu')

for i in N:
    m.addConstr(m._u[i] == gp.quicksum(V[i][j] * m._x[j] for j in J))
    m.addGenConstrLog(m._u[i], m._logu[i])

m.addConstr(gp.quicksum(A[0][j] *m._x[j] for j in J) <= sum(B[i][0] for i in N))

m.setObjective(gp.quicksum(m._logu.values()), gp.GRB.MAXIMIZE)

m.optimize()




