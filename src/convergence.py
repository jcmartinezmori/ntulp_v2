import gurobipy as gp
import pandas as pd
import pickle
from src.config import *
import src.solver as solver


# trips_df = pd.read_csv('{0}/data/chicago_data.csv'.format(RELPATH))

# areas = [
#     8,
#     32,
#     33,
#     35,
#     24,
#     28,
#     31,
#     59,
#     60,
#     34
# ]

areas = [
    8,
    32,
    33,
    24,
    28,
    31,
    23,
    27,
    29,
]

n = 'community-area'
with open('{0}/results/instances/instance_{1}_{2}.pkl'.format(RELPATH, FILENAME, n), 'rb') as file:
    instance = pickle.load(file)

objective = 'utilitarian'
timeLimit = 300
epsLimit = 1
iterLimit = 100
modelname = '{0}-{1}-{2}-{3}'.format(n, objective, timeLimit, epsLimit)

N, J, K, A, B, V = instance

new_N = [i - 1 for i in sorted(areas)]
new_J = [j for j in J if any(V[i][j] for i in new_N)]
new_K = K
new_A = [[A[0][j] for j in new_J]]
new_B = [B[i] for i in new_N]
new_V = [[V[i][j] for j in new_J] for i in new_N]

N_map = {
    i: idx for idx, i in enumerate(new_N)
}
J_map = {
    j: idx for idx, j in enumerate(new_J)
}

new_N = [N_map[i] for i in new_N]
new_J = [J_map[j] for j in new_J]

N, J, K, A, B, V = new_N, new_J, new_K, new_A, new_B, new_V

instance = N, J, K, A, B, V

solver.main(
    instance, modelname,
    objective=objective,
    timeLimit=timeLimit,
    epsLimit=epsLimit,
    iterLimit=iterLimit
)

