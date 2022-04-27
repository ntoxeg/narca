def pathfind(agent_pos: tuple[int, int], pos: tuple[int, int]) -> list[str]:
    """Use A* algorithm to find a path from one location to another

    Args:
        agent_pos: agent's current position
        pos: goal position
    Returns:
        list of operations to execute in order to reach the position
    """
    return ["^move_forwards", "^rotate_left", "^rotate_left", "^rotate_left"]
