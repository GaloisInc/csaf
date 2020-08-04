import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(-1, 1, 100)
x = t**2

plt.plot(t, x)
plt.title("The Parabola")
plt.show()
