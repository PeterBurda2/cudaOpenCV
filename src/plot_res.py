import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("results_csv/resultsCPU.csv")
dt = 1.1 #sec
t = data["iter"].to_numpy() * dt
x = data["x [px]"].to_numpy()
y = data["y [px]"].to_numpy()

plt.plot(t, x, "bo",label = f"Pos x")
plt.plot(t, y, "ro", label = f"Pos y")
plt.legend()
plt.show()