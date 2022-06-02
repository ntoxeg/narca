# NARCA: NARS Controlled Agent
> An agent capable of playing various games in Gym environments, using NARS for planning

## Installation
You need to have OpenNARS-For-Applications installed somewhere. It is expected that you will export a `NARS_HOME` environment
variable to specify where it is built.

Dependencies are listed in the `env.yml` file, that can be used to create a Conda environment.
`lock.yml` is a lock-file that you can use to make a reproducible environment, Conda might fail to create it though.

Current caveat: Griddly has to be installed from source for Python 3.10. You have to do this manually.

## Running
There are multiple experiments under the `experiments` directory. Each one can be launched with `python <path to the script>`. You can observe the rendered environment
and run TensorBoard to see performance metrics. TensorBoard run data gets saved to `runs`.
There is also Neptune.ai integration that you can use, you need to set `NEPTUNE_PROJECT` and `NEPTUNE_API_TOKEN`.
