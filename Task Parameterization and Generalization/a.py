import matplotlib.pyplot as plt
import numpy as np

# Simple data to display in various forms
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

f, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Simple plot')

wait = input("PRESS ENTER TO CONTINUE.")

    