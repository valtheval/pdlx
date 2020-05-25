import matplotlib.pyplot as plt
from bokeh.plotting import figure, show
import numpy as np

def make_histplot(measures1, measures2=None, title="", xaxis_label="", yaxis_label="",legend_label1="",
                  legend_label2="", bins=50, density=True):
    hist1, edges1 = np.histogram(measures1, density=density, bins=bins)

    p = figure(title=title, background_fill_color="#fafafa", plot_width=900, plot_height=400)
    p.quad(top=hist1, bottom=0, left=edges1[:-1], right=edges1[1:], fill_color="#4682B4", line_color="white",
           legend_label=legend_label1)

    if measures2 is not None:
        hist2, edges2 = np.histogram(measures2, density=density, bins=bins)
        p.quad(top=hist2, bottom=0, left=edges2[:-1], right=edges2[1:], fill_color="#FF8C00", line_color="white",
               legend_label=legend_label2)

    p.xaxis.axis_label = xaxis_label
    p.yaxis.axis_label = yaxis_label
    return p