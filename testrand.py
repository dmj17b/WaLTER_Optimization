import numpy as np

rng = np.random.default_rng(seed=46)

for i in range(10):
  print(rng.uniform(-1,1))

xi_max = 2
yi_max = 2

ledge_height_max = 0.4

max_knee_vel = 0.05
max_wheel_vel = 0.1


def randomize_test_scene():
	xi = rng.uniform(-1,1)*xi_max
	yi = rng.uniform(-1,1)*yi_max

	ledge_height = rng.uniform(0,ledge_height_max)