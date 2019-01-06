"""
Author: Sonu Prasad
Email: sonu.prasad@mycit.ie
file: cities_graph.py
"""
import plotly as py
import numpy as np


def plot_graph(cities, length_of_route, file_name):
    """
    Generates TSP Graph using Plotly
    :param cities: list of cities
    :param length_of_route: Total cost of cities
    :param file_name: For saving the graph
    :return: void
    """
    cities_arr = np.array(cities)
    cities_arr = np.vstack([cities_arr, cities_arr[0]])

    trace0 = {
      "x": cities_arr[:, 0],
      "y": cities_arr[:, 1],
      "name": 'col2',
      "type": 'scatter',
      "mode": 'lines+markers'
    }
    data = [trace0]
    layout = {
      "autosize": True,
      "font": {"color": "rgb(61, 133, 198)"},
      "showlegend": True,
      "title": "Length of the route is {}".format(length_of_route),
      "titlefont": {"color": "rgb(153, 0, 255)"},
      "xaxis": {
        "autorange": True,
        "title": "",
        "type": "linear"
      },
      "yaxis": {
        "autorange": True,
        "title": "",
        "type": "linear"
      }
    }
    fig = dict(data=data, layout=layout)
    py.offline.plot(fig, filename=file_name)



