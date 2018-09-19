# Model-Ensemble Trust-Region Policy Optimization (ME-TRPO)
[Paper](https://arxiv.org/abs/1802.10592)

ME-TRPO is a deep model-based reinforcement learning algorithm that uses neural networks to model both the dynamics and the policy. The dynamics model maintains uncertainty due to limited data through an ensemble of models. The algorithm alternates among adding transitions to a replay buffer, optimizing the dynamics models given the buffer, and optimizing the policy given the dynamics models in [Dyna's style](https://dl.acm.org/citation.cfm?id=122377). This algorithm significantly helps alleviating the *model bias* problem in model-based RL, when the policy exploits the error in the dynamics model. In many Mujoco domains, we show that it can achieve the same final performance as model-free approaches. Here we assume that the reward function can be specified.

## Set-up
1) Install [rllab](https://github.com/rll/rllab) and [conda](https://conda.io/docs/user-guide/install/index.html).
2) Create a python environment and install dependencies `conda env create -f tf14.yml`.
3) Put this folder inside `rllab/sandbox/thanard/me-trpo` folder.
4) run `python run_model_based_rl.py trpo -env half-cheetah`.
