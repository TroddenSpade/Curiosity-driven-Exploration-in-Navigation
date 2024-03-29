from filecmp import cmp
from cv2 import waitKey
import matplotlib.pyplot as plt

from envs.DroneEnv import DroneEnv

env = DroneEnv('./envs/1.csv')

state = env.reset()


for k in range(16):
    print("forward")
    state, r, d, i = env.step([1, 0.0])
    print(r, d, i)
    # plt.imsave(f'{k}.png', state)

for k in range(2):
    print("right")
    state, r, d, _ = env.step([1, 1])
    print(r, d)
    plt.imsave(f'{k}.png', state)

for _ in range(12):
    print("forward")
    state, r, d, _ = env.step([1, 0.0])
    print(r, d)

for _ in range(2):
    print("right")
    state, r, d, _ = env.step([1, 1])
    print(r, d)

for _ in range(3):
    print("forward")
    state, r, d, _ = env.step([1, 0.0])
    print(r, d)

for _ in range(2):
    print("right")
    state, r, d, _ = env.step([1, 1])
    print(r, d)

for _ in range(7):
    print("forward")
    state, r, d, _ = env.step([1, 0.0])
    print(r, d)

for _ in range(2):
    print("left")
    state, r, d, _ = env.step([1, -1])
    print(r, d)

for _ in range(3):
    print("forward")
    state, r, d, _ = env.step([1, 0.0])
    print(r, d)

for _ in range(2):
    print("left")
    state, r, d, _ = env.step([1, -1])
    print(r, d)

for _ in range(6):
    print("forward")
    state, r, d, _ = env.step([1, 0.0])
    print(r, d)

for _ in range(2):
    print("right")
    state, r, d, i = env.step([1, 1])
    print(r, d, i)

for _ in range(2):
    print("forward")
    state, r, d, i = env.step([1, 0.0])
    print(r, d, i)

for _ in range(2):
    print("right")
    state, r, d, i = env.step([1, 1])
    print(r, d, i)

for _ in range(3):
    print("forward")
    state, r, d, i = env.step([1, 0.0])
    print(r, d, i)
exit()