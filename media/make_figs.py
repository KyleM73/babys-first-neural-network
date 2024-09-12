import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

n = 1000
x = np.linspace(-2, 2.2, n, True)
y = x**4 - 3*x**2 - x + 2

fcolor = "grey"

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_ylabel("Loss", size=20, color=fcolor)
ax.set_yticks([])
ax.set_xlabel(r"Model Parameter $w$", size=20, color=fcolor)
ax.set_xticks([])
ax.set_frame_on(False)

"""ax.annotate("",
            xy=(-0.55, 1.77), xycoords="data",
            xytext=(202, 80), textcoords="offset points",
            arrowprops=dict(fc="white", ec="gray", ls="--", lw=1),
            horizontalalignment="center", verticalalignment="bottom")"""
ax.annotate("",
            xy=(1.89, 2), xycoords="data",
            xytext=(13.6, 80), textcoords="offset points",
            arrowprops=dict(fc="red", ec="red", shrink=0.05),
            horizontalalignment="center", verticalalignment="bottom")
ax.annotate(r"$-\nabla L_w$", color=fcolor,
            xy=(1.89, 2), xycoords="data",
            xytext=(-55, 45), textcoords="offset points",
            size=20)
plt.savefig("media/loss.png", transparent=True, bbox_inches="tight")
# plt.show()
