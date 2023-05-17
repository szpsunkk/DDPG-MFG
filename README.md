<h1 align="center">DDPG-MFG</h1>
<div align="center">
  <a href="https://github.com/szpsunkk/DDPG-MFG/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/License-BSD%203--Clause-red.svg" alt="License">
  </a>
   <a>
    <img src="https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg" alt="Python 3.7, 3.8">
  </a>
   <a>
    <img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fimperial-qore%2FPreGAN&count_bg=%23FFC401&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false" alt="Hits">
  </a>
  </div>
The code for 'Vehicular Digital Twin Context Offloading for Collaborative Autonomous Driving: A MFG Empowered DRL Approach'

## Quick Test
Clone the repo:
```
git clone https://github.com/szpsunkk/DDPG-MFG.git
cd DDPG-MFG
```

Create `Conda` environment:
```
conda create -n DDPG-MFG python=3.8
conda activate DDPG-MFG
```

Install dependencies of DRL:
```
sudo apt update & sudo apt upgrade
pip install torch  torchvision datetime numpy random matplotlib seaborn
```

run the main code:
```
python task0.py
```
## Baseline algorithms

|Baseline algoritms|details|
| --- | --- |
|AC|Actor-Critic algorithms|
|DDPG|DRL algorithm|
|TD3|DRL algorithm|


