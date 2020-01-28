"""
Template for loading a data set and creating a plot
"""
import matplotlib.pyplot as plt
import os

from src.data_formats import solution_loader            # Load the solution loader module to load a solution
from src.utils import ODEIndex                          # You probably want the ODE index to plot your data

os.chdir("..")                                          # For running files in the plot directory.
file = "solved_odes/TidalStability_2020-01-28--12-01"   # Relative path and file name

# ------------------------------------------------------- DATA ------------------------------------------------------- #
fig, ax = plt.subplots(1, 1)
soln, init_con, config_params, internal_data = solution_loader(file_name=file)

# ------------------------------------------------------- PLOT ------------------------------------------------------- #
ax.plot(soln.times, soln.solution[:, ODEIndex.a1], label='', color='green')
ax.plot(soln.times, soln.solution[:, ODEIndex.a2], label='', color='blue')
ax.plot(soln.times, soln.solution[:, ODEIndex.a3], label='', color='red')

fig.show()
