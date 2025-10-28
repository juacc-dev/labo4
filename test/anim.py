from matplotlib import pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# initializing a figure in
# which the graph will be plotted
# fig = plt.figure()

# marking the x-axis and y-axis

fig, ax = plt.subplots()

# initializing a line variable
line, = ax.plot([], [], lw=3)


def init():
    line.set_data([], [])
    return line,


x = np.linspace(-10 * np.pi, 10 * np.pi, 1000)


def animate(n):
    a_n = 0
    b_n = (-1) ** n
    animate.y += a_n * np.cos(n * x) + b_n * np.sin(n * x)
    line.set_data(x, animate.y)

    return line,


animate.y = 0

anim = FuncAnimation(fig, animate, init_func=init,
                     frames=10, interval=20, blit=True)

plt.show()
