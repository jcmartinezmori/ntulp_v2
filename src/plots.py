import pandas as pd
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.config import *


def plot_convergence(ns, objectives, timeLimit, epsLimit, iterLimit, showlegend=False):

    data = []

    for n in ns:

        with open('{0}/results/instances/instance_{1}_{2}.pkl'.format(RELPATH, FILENAME, n), 'rb') as file:
            instance = pickle.load(file)
        N, J, K, A, B, V = instance

        for objective in objectives:
            modelname = '{0}-{1}-{2}-{3}'.format(n, objective, timeLimit, epsLimit)
            for iterCount in range(iterLimit + 1):
                try:
                    with open('{0}/results/solutions/{1}_{2}_{3}.pkl'.format(
                            RELPATH, FILENAME, modelname, iterCount
                    ), 'rb') as file:
                        _, u_N, totTime, cutCount, eps, _, kappa, epsMIPGap = pickle.load(file)
                        totTime /= (60 * 60)
                        epsMIPGap *= 100
                        utilObj = sum(B[i][0] * u_N[i] for i in N) / sum(B[i][0] for i in N)
                        maximinObj = min(u_N[i] for i in N)
                        data.append(
                            (
                                n, objective, timeLimit, iterCount,
                                totTime, cutCount, eps, kappa, epsMIPGap,
                                utilObj, maximinObj
                            )
                        )
                except FileNotFoundError:
                    pass

    df = pd.DataFrame(
        data,
        columns=[
            'n', 'objective', 'timelimit', 'iterCount',
            'totTime', 'cutCount', 'eps', 'kappa', 'epsMIPGap',
            'utilObj', 'maximinObj'
        ]
    )

    color_map = {n: HEXCOLOR for n, HEXCOLOR in zip(ns, HEXCOLORS)}
    marker_map = {n: MARKER for n, MARKER in zip(ns, MARKERS)}
    subplot_map = {objective: idx + 1 for idx, objective in enumerate(objectives)}

    plots = (
        ('totTime', r'$\large \textrm{Elapsed time [hr.]}$'),
        ('cutCount', r'$\large \textrm{Number of Cuts}$'),
        ('eps', r'$\large \textrm{Multiplicative Least Objection}$'),
        ('kappa', r'$\large \textrm{Basis Condition Number}$'),
        ('epsMIPGap', r'$\large \textrm{Optimality Gap % of Multiplicative Least Objection}$'),
        ('utilObj', r'$\large \textrm{Utilitarian Social Welfare}$'),
        ('maximinObj', r'$\large \textrm{Maximin Social Welfare}$')
    )
    subplot_titles = {
        'utilitarian': r'$\Large \textrm{Utilitarian Service Goal}$',
        'maximin': r'$\Large \textrm{Maximin Service Goal}$',
        'nash': r'$\Large \textrm{Nash Service Goal}$'
    }

    for col, title in plots:
        fig = make_subplots(
            rows=1, cols=len(objectives), shared_xaxes=True, shared_yaxes=True,
            subplot_titles=[subplot_titles[objective] for objective in objectives]
        )
        for (objective, n), group_df in df.groupby(['objective', 'n']):
            fig.add_trace(
                go.Scatter(
                    x=group_df['iterCount'],
                    y=group_df[col],
                    mode='lines+markers',
                    name=r'$n: {0}$'.format(n),
                    showlegend=True if subplot_map[objective] == 1 and showlegend else False,
                    line={'color': color_map[n], 'dash': 'solid'},
                    marker={'color': color_map[n], 'symbol': marker_map[n], 'size': 5}
                ),
                row=1,
                col=subplot_map[objective]
            )
            if col in {'eps'}:
                fig.add_trace(
                    go.Scatter(
                        x=group_df['iterCount'],
                        y=[1 for _ in group_df['iterCount']],
                        line={'color': HEXREDDISHPURPUPLE, 'dash':'dot'},
                        showlegend=False
                    ),
                    row=1,
                    col=subplot_map[objective]
                )
        if col in {'kappa'}:
            fig.update_yaxes(type='log')
        fig.update_yaxes(
            row=1, col=1,
            title_text=title, title_font={'size': 26}
        )
        for idx, _ in enumerate(objectives):
            fig.update_xaxes(
                row=1, col=idx + 1,
                range=[0, iterLimit], title_text=r'$\large \textrm{Number of Iterations}$', title_font={'size': 24}
            )
        for annotation in fig['layout']['annotations']:
            annotation['font'] = {'size': 24}
            annotation['y'] = 1.0125

        # fig.show()
        fig.write_image('./results/figures/{0}.png'.format(col), width=1200, height=600)


def plot_utilities(title, keys):

    data = []

    for n, objective, timeLimit, epsLimit, iterCount, subtitle in keys:

        V_riders = pd.read_csv('{0}/results/instances/V_riders.csv'.format(RELPATH, FILENAME))

        modelname = '{0}-{1}-{2}-{3}'.format(n, objective, timeLimit, epsLimit)
        try:
            with open('{0}/results/solutions/{1}_{2}_{3}.pkl'.format(
                    RELPATH, FILENAME, modelname, iterCount
            ), 'rb') as file:
                x_N, _, _, _, _, _, _, _ = pickle.load(file)

                x_N = pd.Series(x_N)
                x_N.index = x_N.index.astype(int) - 1

                u_riders = sorted(V_riders.to_numpy() @ x_N.to_numpy())
                for r, u_r in enumerate(u_riders):
                    data.append(
                        (
                            n, objective, timeLimit, epsLimit, iterCount, subtitle, r + 1, u_r
                        )
                    )
        except FileNotFoundError:
            pass

    df = pd.DataFrame(
        data,
        columns=['n', 'objective', 'timeLimit', 'epsLimit', 'iterCount', 'title', 'r', 'u_r']
    )

    color_map = {key: HEXCOLOR for key, HEXCOLOR in zip(keys, HEXCOLORS)}
    marker_map = {key: MARKER for key, MARKER in zip(keys, MARKERS)}

    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=(title,)
    )

    for key, group_df in df.groupby(['n', 'objective', 'timeLimit', 'epsLimit', 'iterCount', 'title']):
        _, objective, _, _, _, subtitle = key
        fig.add_trace(
            go.Scatter(
                x=group_df['r'],
                y=group_df['u_r'],
                mode='lines+markers',
                name=subtitle,
                line={'color': color_map[key], 'dash': 'solid'},
                marker={'color': color_map[key], 'symbol': marker_map[key], 'size': 5}
            ),
            row=1, col=1
        )

    fig.update_yaxes(
        row=1, col=1,
        title_text=r'$\large \textrm{Rider Utility}$', title_font={'size': 24},
        range=[0, 50000]
    )
    fig.update_xaxes(
        row=1, col=1,
        title_text=r'$\large \textrm{Riders (sorted by Rider Utility)}$', title_font={'size': 24}
    )
    for annotation in fig['layout']['annotations']:
        annotation['font'] = {'size': 24}
        annotation['y'] = 1.0125
    fig.update_layout(
        legend={
            'orientation': 'h', 'yanchor': 'top', 'y': 0.925, 'xanchor': 'left', 'x': 0.075,
            'font': {'size': 22}, 'entrywidth': 150,
        }
    )

    # fig.show()
    fig.write_image('./results/figures/utils-{0}.png'.format(objective), width=800, height=600, scale=4)


if __name__ == '__main__':

    ns = ['community-area']
    objectives = ['maximin', 'utilitarian', 'nash']
    # objectives = ['nash']
    timeLimit = 300
    epsLimit = 1
    iterLimit = 100

    plot_convergence(ns, objectives, timeLimit, epsLimit, iterLimit)

    # title = r'$\Large \textrm{Maximin Service Goal}$'
    # keys = [
    #     ('community-area', 'maximin', timeLimit, epsLimit, iterLimit, r'$\textrm{With cooperation}$'),
    #     ('community-area', 'maximin', timeLimit, epsLimit, -1, r'$\textrm{Without cooperation}$'),
    #     ('community-area', 'nash', timeLimit, epsLimit, -1, r'$\textrm{Nash Social Welfare}$'),
    # ]
    # plot_utilities(title, keys)
    # #
    # title = r'$\Large \textrm{Utilitarian Service Goal}$'
    # keys = [
    #     ('community-area', 'utilitarian', timeLimit, epsLimit, iterLimit, r'$\textrm{With cooperation}$'),
    #     ('community-area', 'utilitarian', timeLimit, epsLimit, -1, r'$\textrm{Without cooperation}$'),
    #     ('community-area', 'nash', timeLimit, epsLimit, -1, r'$\textrm{Nash Social Welfare}$'),
    # ]
    # plot_utilities(title, keys)

    # title = r'$\Large \textrm{Nash Bargaining Service Goal}$'
    # keys = [
    #     ('community-area', 'nash', timeLimit, epsLimit, -1, r'$\textrm{Nash}$'),
    # ]
    # plot_utilities(title, keys)

