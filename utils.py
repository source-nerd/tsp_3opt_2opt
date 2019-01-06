"""
Author: Sonu Prasad
Email: sonu.prasad@mycit.ie
file: utils.py
"""
import logging
import plotly as py
import plotly.graph_objs as go
from plotly import tools


def print_multi_line_graph(data, file_name, graph_title):
    # Create and style traces
    trace0 = go.Scatter(
        x=data,
        y=list(map(lambda x: x/len(data), range(len(data)))),
        line=dict(
            color='rgb(251, 212, 138)',
            width=4)
    )

    new_data = [trace0]

    # Edit the layout
    layout = dict(title=graph_title,
                  xaxis=dict(title='Run Time'),
                  yaxis=dict(title='Iteration COunt'),
                  )

    fig = dict(data=new_data, layout=layout)
    py.offline.plot(fig, filename=file_name)


def initialize_loggers(file_name):
    """
    Initialize Logging
    :return:
    - log_array: Which consists of multiple objects of Logging
    - logging- Besic logging which can be used at the root level
    """
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='{}.log'.format(file_name),
                        filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-Initializing*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
    logging.info('Setting up Loggers')

    return logging
