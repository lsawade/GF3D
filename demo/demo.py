# %%
# LOAD THE KDTREE FIRST!!!!!!!!!!!!!
# If other packages are loaded, the wrong libstdc++ is picked up and
# doesn't contain the right GLIBCXX version

import numpy as np
import matplotlib.pyplot as plt
from lwsspy.plot import plot_label
from lwsspy.GF.source import CMTSOLUTION
from lwsspy.GF.seismograms import SGTManager
from matplotlib.colors import Normalize, LogNorm
from lwsspy.plot import axes_from_axes, get_aspect
