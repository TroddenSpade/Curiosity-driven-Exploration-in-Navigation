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
- [VizDoom](https://github.com/mwydmuch/ViZDoom)
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

### Montezuma Revenge
The goal is to acquire Montezuma’s treasure by making a way through a maze of chambers within the emperor’s fortress. Player must avoid deadly creatures while collecting valuables and tools which can help them escape with the treasure.

### VizDoom
VizDoom is a Doom-based AI Research Platform for Reinforcement Learning from Raw Visual Information.

#### Basic
This map is a rectangle with walls, ceiling and floor. Player is spawned along the longer wall, in the center. A circular monster is spawned randomly somewhere along the opposite wall. Player can only go left/right and shoot. One hit is enough to kill the monster and the episode finishes when the monster is killed or on timeout.

https://user-images.githubusercontent.com/33734646/195578564-e24e123f-b71f-41a2-92f9-f86bb035a248.mp4


#### Defend the Center
This map is a large circular environment. Player is spawned in the exact center. 5 melee-only, monsters are spawned along the wall. Monsters are killed after a single shot. After dying each monster is respawned after some time. Episode ends when the player dies.

#### Deadly Corridor
This map is a corridor with shooting monsters on both sides (6 monsters in total). A green vest is placed at the oposite end of the corridor. Reward is proportional (negative or positive) to change of the distance between the player and the vest. If player ignores monsters on the sides and runs straight for the vest he will be killed somewhere along the way.


## References

1. Pathak, D., Agrawal, P., Efros, A. A. & Darrell, T. Curiosity-driven Exploration by Self-supervised Prediction. 34th Int. Conf. Mach. Learn. ICML 2017 6, 4261–4270 (2017).
2. Burda, Y., Edwards, H., Storkey, A. & Klimov, O. Exploration by Random Network Distillation. 7th Int. Conf. Learn. Represent. ICLR 2019 1–17 (2018).
3. Burda, Y., Storkey, A., Darrell, T. & Efros, A. A. Large-scale study of curiosity-driven learning. 7th Int. Conf. Learn. Represent. ICLR 2019 (2019).
4. Badia, A. P. et al. Never Give Up: Learning Directed Exploration Strategies. 1–28 (2020).
