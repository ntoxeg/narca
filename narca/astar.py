def pathfind(agent_pos: tuple[int, int], pos: tuple[int, int]) -> list[str]:
    """Use A* algorithm to find a path from one location to another

    Args:
        agent_pos: agent's current position
        pos: goal position
    Returns:
        list of operations to execute in order to reach the position.
        note: operations are absolute - conversion to rotate / forward / backward might be needed.
    """
    dx, dy = pos[0] - agent_pos[0], pos[1] - agent_pos[1]
    path_ops = []
    if dx > 0:
        path_ops.extend(["^right"] * abs(dx))
    else:
        path_ops.extend(["^left"] * abs(dx))
    if dy > 0:
        path_ops.extend(["^down"] * abs(dy))
    else:
        path_ops.extend(["^up"] * abs(dy))

    return path_ops
