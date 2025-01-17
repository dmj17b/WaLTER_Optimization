import os
import mujoco
import jax
import jax.numpy as jnp
from pathlib import Path
import AutoSim
import yaml

# Update jax version to 64-bit
jax.config.update("jax_enable_x64", True)

# Define the model:

# Create a new simulation model with AutoSim
model_config_path = 'model_config.yaml'
motor_config_path = 'motor_config.yaml'
motor_config = yaml.safe_load(Path(motor_config_path).read_text())

num_tests = 50
rng_seed = 420
i = 1

success_dist = 3
torso_head_angle_tolerance = 25

num_successes = 0
num_failures = 0

walter = AutoSim.GenerateModel(model_config_path=model_config_path, motor_config_path=motor_config_path)
walter.gen_scene()
rng = jnp.random.default_rng(seed=i+rng_seed)

# Compile the model:
m = walter.spec.compile()
d = mujoco.MjData(m)

# Randomize initial robot pose:
d = walter.randomize_pose(rng,m,d)



# Main function:
def main(argv=None):
    pass



if __name__ == '__main__':
    main()