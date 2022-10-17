import sys
import cherry as ch
sys.path.append("../")

from envs.DroneEnv import DroneEnv
from wrapper import FrameStack
from wrapper import ResizeObservation
from wrapper import GreyScaleObservation
from wrapper import LazyFramesToArray

def DroneEnvModified():
    env = DroneEnv()
    env = GreyScaleObservation(env)
    env = ResizeObservation(env, 84)
    env = FrameStack(env, 4)
    env = LazyFramesToArray(env)
    env = ch.envs.Torch(env)
    return env


if __name__ == "__main__":
    env = DroneEnvModified()
    state = env.reset()
    print(state.shape)
