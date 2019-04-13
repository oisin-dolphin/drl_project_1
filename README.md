# drl_project_1

## Project Details

This project uses a dueling dqn algorithm to solve the unity bannana environment.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions.

Four discrete actions are available, corresponding to:

    0 - move forward.
    1 - move backward.
    2 - turn left.
    3 - turn right.


## Getting Started

I assume you're operating on Linux and have conda installed along with
CUDA and an nvidia graphics card with the latest drivers.


### Load the conda environment

```conda env create -f environment.yml```
```conda activate drlnd```

### Download the Unity Environment

Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)

Unzip and move it into the repo.

### Run jupyter notebook

```jupyter-notebook```

And then run all the cells to watch the agent train!

### Testing (very basic)

```pytest```
