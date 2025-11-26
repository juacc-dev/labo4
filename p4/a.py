import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 1000)
y = np.unwrap(x)


plt.plot(x, y)
plt.grid()
plt.show()
