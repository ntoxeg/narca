# NARCA: NARS Controlled Agent
> An agent capable of playing various games in Gym environments, using NARS for planning

## Installation
You need to have OpenNARS-For-Applications installed somewhere. It is expected that you will export a `NARS_HOME` environment
variable to specify where it is built.

Dependencies are listed in the `env.yml` file, that can be used to create a Conda environment.
`lock.yml` is a lock-file that you can use to make a reproducible environment, Conda might fail to create it though. For standard virtual environments you just need to use `pip`.
You can also use Poetry, via `poetry install`, `poetry shell` will open a shell with the virtual environment activated.

[Griddly](https://github.com/Bam4d/Griddly) requires to install Vulkan SDK, refer to its README.

Install in editable mode to your environment: `pip install -e .`

## Running
There are multiple experiments under the `experiments` directory. Each one can be launched with `python <path to the script>`. You can observe the rendered environment
and run TensorBoard to see performance metrics. TensorBoard run data gets saved to `runs`.
There is also Neptune.ai integration that you can use, you need to set `NEPTUNE_PROJECT` and `NEPTUNE_API_TOKEN`.

Each Griddly environment has its own subdirectory, each has at least the `main` experiment - this is supposed to hold the currently established simplest best performing agent.

### Notable experiments
Besides the usual `main` experiments there are some interesting variations as follows:
- `experiments/zelda/procgen`: uses a level generator that places a key and a door randomly around the level.
- `experiments/drunk_dwarf/curriculum_learning`: tries to find out whether there is a measurable gain in learning efficiency with a curriculum learning setup.
