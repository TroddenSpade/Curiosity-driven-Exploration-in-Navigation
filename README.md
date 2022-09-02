# Curiosity-driven Exploration in Drone Navigation

## Requirements
* Pytorch
* Unreal Engine 4
* [Airsim Drone](https://microsoft.github.io/AirSim/)
* Gym
* [Cherry-RL](http://cherry-rl.net)

## Environments
This repository includes the following environments, each of them is composed of `Sparse` and `Dense` reward modes.
- CartPole-v1
- Montezuma Revenge
- UE4 Airsim Maze Environment 
  - This environment was derived form [`frasergeorgeking/UE4_BP_MazeGen_MIT`](https://github.com/frasergeorgeking/UE4_BP_MazeGen_MIT) which is a free and open source maze generator with various themes for Unreal Engine. The `Airsim Drone` package was attached to it and it is available for modification and download in [`TroddenSpade/UE4-Airsim-Maze-Environment`](https://github.com/TroddenSpade/UE4-Airsim-Maze-Environment)

| First-person | Third-person | Top-down |
| :---: | :---: | :---: |
| <img src="https://github.com/TroddenSpade/Curiosity-driven-Exploration-in-Drone-Navigation/blob/main/assets/front.gif?raw=true" width="300px"> | <img src="https://github.com/TroddenSpade/Curiosity-driven-Exploration-in-Drone-Navigation/blob/main/assets/thirdperson.gif?raw=true" width="300px"> | <img src="https://github.com/TroddenSpade/Curiosity-driven-Exploration-in-Drone-Navigation/blob/main/assets/top.gif?raw=true" width="300px"> |

## Contents
  - [x] Advantage Actor Critic (A2C)
  - [x] Proximal Policy Optimization (PPO)
  - [x] Intrinsic Curiosity Module (ICM)
  - [x] Random Network Distillation (RND)
  - [ ] Universal Value Function Approximators (UVFA)
  - [ ] Never Give Up (NGU)
  
## Experiments

### CartPole-v1
<img src="https://github.com/TroddenSpade/Curiosity-driven-Exploration-in-Drone-Navigation/blob/main/assets/cartpole_plot.png?raw=true" width="800px">


## References

1. Pathak, D., Agrawal, P., Efros, A. A. & Darrell, T. Curiosity-driven Exploration by Self-supervised Prediction. 34th Int. Conf. Mach. Learn. ICML 2017 6, 4261–4270 (2017).
2. Burda, Y., Edwards, H., Storkey, A. & Klimov, O. Exploration by Random Network Distillation. 7th Int. Conf. Learn. Represent. ICLR 2019 1–17 (2018).
3. Burda, Y., Storkey, A., Darrell, T. & Efros, A. A. Large-scale study of curiosity-driven learning. 7th Int. Conf. Learn. Represent. ICLR 2019 (2019).
4. Badia, A. P. et al. Never Give Up: Learning Directed Exploration Strategies. 1–28 (2020).
