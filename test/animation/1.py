import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
ln, = ax.plot([], [], '.')

x = np.linspace(0, 2 * np.pi, 200)
y = np.sin(x)


def init():
    ax.set_xlim(-0.1, 2.1 * np.pi)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True)
    return ln,


def animate(i):
    ln.set_data(x[:i], y[:i])
    return ln,


ani = FuncAnimation(
    fig,
    animate,
    frames=range(1, x.size + 1),
    init_func=init,
    interval=16,
    blit=True,
    # repeat=False
)
plt.show()
