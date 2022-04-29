# NARCA: NARS Controlled Agent
> An agent capable of playing various games in Gym environments, using NARS for planning

## Installation
You need to have OpenNARS-For-Applications installed somewhere. It is expected that you will export a `NARS_HOME` environment
variable to specify where it is built.

Dependencies are listed in the `env.yml` file, that can be used to create a Conda environment.
`lock.yml` is a lock-file that you can use to make a reproducible environment, Conda might fail to create it though.

Current caveat: Griddly has to be installed from source for Python 3.10. You have to do this manually.

## Running
Currently, there is only one experiment, `nars_zelda.py`, that you can simply run. You can observe the rendered environment
and run TensorBoard to see rewards per episode.
