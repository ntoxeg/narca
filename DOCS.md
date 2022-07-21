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
  `<{key} --> [reached]) =/> GOT_KEY>.`
- `DOOR_OPENED` - satisfied after the agent opens the door.
  `<(GOT_KEY &/ <{door} --> [reached]>) =/> DOOR_OPENED>.`
- `COMPLETE` - satisfied when the agent enters the bed.
  `<(DOOR_OPENED &/ <{coffin_bed} --> [reached]>) =/> COMPLETE>.`
- `GOT_REWARD` - satisfied every time the agent receives a positive reward. Rewards are not directly observed.

Background knowledge: `<(<$obj --> [ahead]> &/ ^move_forwards) =/> <$obj --> [reached]>>.` This is supposed to inform the agent that moving forward when something is ahead leads to the object being reached.

## Perception configuration
The following are beliefs sent to the agent to inform it about what is happening in the environment and with itself.
`$VAR` in the following statements signifies string interpolation.

### Agent state
- `{SELF} --> [orient-$ORIENTATION]. :|:`, where `ORIENTATION` is the agetn's orientation as received from the Gym environment observation.

### Relative position beliefs
#### General direction
- `{$objname} --> [ahead]. :|:`, where `objname` is the ID of the object.
  This belief signifies that an object is in front of the agent, in a straight line.
- `{$objname} --> [leftward]. :|:` means that an object is in front of the agent, but to the left of it.
- `{$objname} --> [rightward]. :|:` means that an object is in front of  the agent, but to the right of it.

#### Distance
- `({$objname} * $delta) --> delta_forward. :|:`, represents how many cells forward the agent has to travel to make one of its coordinates equal to the object's. `delta` is simply an integer.
- `({$objname} * $delta) --> delta_sideways. :|:`, represents how many cells to the left or right the agent has to travel to make one of its coordinates equal to the object's. `delta` is 0 if the object is straight ahead, `L$distance` if the object is to the left, `R$distance` if the object is to the right.


### Differential beliefs
These beliefs rely on tracking state changes between the current and previous time steps.
- `{$objname} --> [new]. :|:` means that an object has appeared in the field of vision.
- `{$objname} --> [gone]. :|:` means that an object has disappeared from the field of vision.
- `{$objname} --> [closer]. :|:` means that the agent has decreased its Manhattan distance from the object.
- `{$objname} --> [further]. :|:` means that the agent has increased its Manhattan distance from the object.

### Special beliefs
- `{$objname} -> [reached]. :|:` signifies that the object has been reached by the agent.
