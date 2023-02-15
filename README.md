# Neural Control and Online Learning for Robust Locomotion of Legged Robots

This contains the CPG-RBFN and PIBB learning as well as the framework for working using IsaacGym.

### Content:
- [System Requirements](#system-requirements)
- [Code overview](#code-overview)
- [Install](#install)
- [Run the controller with learned weights](#run-the-controller-with-learned-weights)
- [Run learning algorithm](#run-learning-algorithm)
- [License](#license)

## System Requirements
This code has been tested with the following hardware and software:
- Intel® Core™ i9-9900K CPU @ 3.60GHz × 16
- GeForce RTX 2080
- Ubuntu 20.04.5 LTS
- Isaac Gym Preview 4
- Python 3.10.6

## Code overview

## Install

1. Make sure to have installed pytorch with cuda. In this project pytorch 1.10 with cuda-11.3 has been used:

```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

2. Download and install Isaac Gym Preview 4 from its [official page](https://developer.nvidia.com/isaac-gym). Registration is needed.
3. Unzip the file via:

```bash
tar -xf IsaacGym_Preview_4.tar.gz
```

4. Install the python package:

```bash
cd isaacgym_lib/python && pip install -e .
```

5. Verify the installation by running an example:

```bash
cd examples && python3 1080_balls_of_solitude.py
```

6. Clone this repository to your local machine:

```bash
git clone https://github.com/DFdz26/online-locomotion-rl
```

## Run the controller with learned weights

## Run learning algorithm

## License
All software is available under the [GPL-3](http://www.gnu.org/licenses/gpl.html) license.

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
