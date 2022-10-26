import gym
import cv2

IMAGE_SHAPE = (50, 80)
class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, shape=IMAGE_SHAPE):
        super().__init__(env)
        self.image_shape = shape
        self.image_shape_reverse = shape[::-1]

        # Create new observation space with the new shape
        new_shape = (shape[0], shape[1])
        self.observation_space = gym.spaces.Box(0, 255, shape=new_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = observation["rgb"][:200, :]
        observation = cv2.resize(observation, self.image_shape_reverse)
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return observation