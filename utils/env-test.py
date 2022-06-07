import airsim
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time


# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Async methods returns Future. Call join() to wait for task to complete.
client.takeoffAsync().join()
# client.moveByVelocityAsync(0, 0, -1.5, 1).join()

fig, axs = plt.subplots(ncols=2, nrows=13, figsize=(5, 25))

for i in range(13):
    client.moveByVelocityBodyFrameAsync(2,0,0,0.1).join()
    responses = client.simGetImages([
    airsim.ImageRequest(1, airsim.ImageType.DepthVis, True, False),])
    print('Retrieved images: %d', len(responses))

    depth = np.array(responses[0].image_data_float, dtype=np.float32)
    depth = depth.reshape(responses[0].height, responses[0].width)
    print(depth.shape)
    axs[i,0].imshow(depth, cmap='gray')


state = client.getMultirotorState()
print(state)
# for i in range(9):
client.rotateByYawRateAsync(90, 0.1).join()
client.rotateByYawRateAsync(90, 0.1).join()
responses = client.simGetImages([
airsim.ImageRequest(1, airsim.ImageType.DepthVis, True, False),])
print('Retrieved images: %d', len(responses))

depth = np.array(responses[0].image_data_float, dtype=np.float32)
depth = depth.reshape(responses[0].height, responses[0].width)

axs[0,1].imshow(depth, cmap='gray')

for i in range(10):
    client.moveByVelocityBodyFrameAsync(2,0,0,0.5).join()
    responses = client.simGetImages([
    airsim.ImageRequest(1, airsim.ImageType.DepthVis, True, False),])
    print('Retrieved images: %d', len(responses))

    depth = np.array(responses[0].image_data_float, dtype=np.float32)
    depth = depth.reshape(responses[0].height, responses[0].width)

    axs[i+1,1].imshow(depth, cmap='gray')
plt.show()

# client.moveToPositionAsync(5, 0, 0, 1).join()
# client.moveByAngleZAsync(1, 0, 0, 0, 1).join()

# for i in range(5):
#     client.moveByVelocityZBodyFrameAsync(1, 0, 0, 1).join()


# take images
responses = client.simGetImages([
    airsim.ImageRequest(1, airsim.ImageType.DisparityNormalized, True, False),])
print('Retrieved images: %d', len(responses))

depth = np.array(responses[0].image_data_float, dtype=np.float32)
depth = depth.reshape(responses[0].height, responses[0].width)


client.reset()
