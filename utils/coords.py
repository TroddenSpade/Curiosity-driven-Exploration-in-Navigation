import airsim
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.reset()

client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()

coords = []

pos = [0, 0, -1]
for i in range(15):
    pos[0] += 1
    client.moveToPositionAsync(pos[0], pos[1], pos[2], 1).join()

    state = client.getMultirotorState()
    # print(state)
    coords.append([state.kinematics_estimated.position.x_val, 
                    state.kinematics_estimated.position.y_val])
    print(state.kinematics_estimated.position)

client.rotateByYawRateAsync(90, 1).join()
time.sleep(0.5)

for i in range(10):
    pos[1] += 1
    client.moveToPositionAsync(pos[0], pos[1], pos[2], 1).join()

    state = client.getMultirotorState()
    coords.append([state.kinematics_estimated.position.x_val, 
                    state.kinematics_estimated.position.y_val])

    print(state.kinematics_estimated.position)

client.rotateByYawRateAsync(90, 1).join()
time.sleep(0.5)

for i in range(5):
    pos[0] -= 1
    client.moveToPositionAsync(pos[0], pos[1], pos[2], 1).join()

    state = client.getMultirotorState()
    coords.append([state.kinematics_estimated.position.x_val, 
                    state.kinematics_estimated.position.y_val])

    print(state.kinematics_estimated.position)

client.rotateByYawRateAsync(-90, 1).join()
time.sleep(0.5)

for i in range(5):
    pos[1] += 1
    client.moveToPositionAsync(pos[0], pos[1], pos[2], 1).join()

    state = client.getMultirotorState()
    coords.append([state.kinematics_estimated.position.x_val, 
                    state.kinematics_estimated.position.y_val])

    print(state.kinematics_estimated.position)

client.rotateByYawRateAsync(90, 1).join()
time.sleep(0.5)

for i in range(6):
    pos[0] -= 1
    client.moveToPositionAsync(pos[0], pos[1], pos[2], 1).join()

    state = client.getMultirotorState()
    coords.append([state.kinematics_estimated.position.x_val, 
                    state.kinematics_estimated.position.y_val])

    print(state.kinematics_estimated.position)

client.rotateByYawRateAsync(90, 1).join()
time.sleep(0.5)

for i in range(5):
    pos[1] -= 1
    client.moveToPositionAsync(pos[0], pos[1], pos[2], 1).join()

    state = client.getMultirotorState()
    coords.append([state.kinematics_estimated.position.x_val, 
                    state.kinematics_estimated.position.y_val])

    print(state.kinematics_estimated.position)

client.rotateByYawRateAsync(-90, 1).join()
time.sleep(0.5)

for i in range(4):
    pos[0] -= 1
    client.moveToPositionAsync(pos[0], pos[1], pos[2], 1).join()

    state = client.getMultirotorState()
    coords.append([state.kinematics_estimated.position.x_val, 
                    state.kinematics_estimated.position.y_val])

    print(state.kinematics_estimated.position)

coords = np.array(coords)
np.save("./envs/coords1.npy", coords)