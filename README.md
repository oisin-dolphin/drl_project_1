# drl_project_1

## Project Details

This project uses a dueling dqn algorithm to solve the unity bannana environment.

## Getting Started

I assume you're operating on Linux and have conda installed along with
CUDA and an nvidia graphics card with the latest drivers.


### Load the conda environment

```conda env create -f environment.yml```
```conda activate drlnd```

### Download the Unity Environment

Linux: click here [https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip]

Unzip and move it into the repo.

### Run jupyter notebook

```jupyter-notebook```

And then run all the cells to watch the agent train!

### Testing (very basic)

```pytest```