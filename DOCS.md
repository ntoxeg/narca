# Documentation for NARS Controlled Agent

## Outline of experiment structure

Experiments consist of 4 main stages:
- Setting up the Gym environment and experiment parameters.
- Defining goals and optional background knowledge to be used.
- Adding optional callbacks, like ones that log to Neptune.
- Creating a new `Agent` instance and a `Runner` instance.
    - Using the `Runner.run` method to run the experiment.

## Parameters, goals and background knowledge
- `VIEW_RADIUS` regulates the radius from which information about objects is sent to the agent.
- `THINK_TICKS` does not really do anything at the moment.

Default run is 50 episodes long, with each of them capped at 100 steps.

### Drunk Dwarf
- `GOT_KEY` - that goal is satisfied after the agent acquires the key.
- `DOOR_OPENED` - satisfied after the agent opens the door.
- `COMPLETE` - satisfied when the agent enters the bed.

Background knowledge: `<(<$obj --> [ahead]> &/ ^move_forwards) =/> <$obj --> [reached]>>.` This is supposed to inform the agent that moving forward when something is ahead leads to the object being reached.