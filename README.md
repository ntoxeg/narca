# NARCA: NARS Controlled Agent
> An agent capable of playing various games in Gym environments, using NARS for planning

This repository hosts both a library that facilitates interfacing OpenNARS for Applications with OpenAI Gym and
a bunch of experiments using Griddly.

## Installation
The recommended way to install dependencies is with Poetry: `poetry install`. `poetry shell` will open a shell with the virtual environment activated.

[Griddly](https://github.com/Bam4d/Griddly) requires to install Vulkan SDK, refer to its README.

Install in editable mode to your environment: `pip install -e .`
There is also a PyPI package available: `pip install narca`.

## Running
There are multiple experiments under the `experiments` directory. Each one can be launched with `python <path to the script>`. You can observe the rendered environment
and run TensorBoard to see performance metrics. TensorBoard run data gets saved to `runs`.
There is also Neptune.ai integration that you can use, you need to set `NEPTUNE_PROJECT` and `NEPTUNE_API_TOKEN`.

Each Griddly experiment has its own subdirectory, each has at least the `main` experiment - this is supposed to hold the currently established simplest best performing agent.

### Notable experiments
Besides the usual `main` experiments there are some interesting variations as follows:
- `experiments/zelda/procgen`: uses a level generator that places a key and a door randomly around the level.
- `experiments/drunk_dwarf/curriculum_learning`: tries to find out whether there is a measurable gain in learning efficiency with a curriculum learning setup.

## AGI-22 presentation
Here you can watch the presentation of some results from this project as of August 2022: https://youtu.be/ciDH3oJbP0c?t=20868
