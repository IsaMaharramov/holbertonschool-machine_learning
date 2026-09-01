# Deep Q-learning (Atari Breakout)

This project contains two Python scripts demonstrating the implementation of a Deep Q-Network (DQN) applied to the Atari game Breakout, using `keras-rl2` and `gymnasium`.

## Tasks

### 0. Breakout
* `train.py`: Initializes an `ALE/Breakout-v5` environment, bridges the Gymnasium API with `keras-rl2` using a custom wrapper, and trains a `DQNAgent` employing `SequentialMemory` and `EpsGreedyQPolicy`. It saves the weights to `policy.h5`.
* `play.py`: Loads the trained `policy.h5` and visualizes the model playing Breakout utilizing the `GreedyQPolicy`.

## Installation & Dependencies
Ensure you run this on **Ubuntu 20.04 LTS** using **python3.9**.
```bash
pip install --user gymnasium[atari]==0.29.1
pip install --user tensorflow==2.15.0
pip install --user keras==2.15.0
pip install --user numpy==1.25.2
pip install --user Pillow==10.3.0
pip install --user h5py==3.11.0
pip install autorom[accept-rom-license]
pip install --user keras-rl2==1.0.4
```