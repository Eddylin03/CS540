import heapq

def calculate_manhattan_distance(idx: int, value: int, num_tiles: int) -> int:
    if value == 0 or value > num_tiles:
        return 0
    current_row, current_col = idx // 3, idx % 3
    goal_row, goal_col = (value - 1) // 3, (value - 1) % 3
    return abs(current_row - goal_row) + abs(current_col - goal_col)

def calculate_heuristic(state: list) -> int:

    num_tiles = sum(1 for x in state if x != 0)
    return sum(calculate_manhattan_distance(i, val, num_tiles) 
              for i, val in enumerate(state) 
              if val != 0)

def get_goal_state(state: list) -> list:

    num_tiles = sum(1 for x in state if x != 0)
    return list(range(1, num_tiles + 1)) + [0] * (9 - num_tiles)

def get_successors(state: list) -> list:

    successors = []
    empty_positions = [i for i, val in enumerate(state) if val == 0]
    
    for empty_pos in empty_positions:
        row, col = empty_pos // 3, empty_pos % 3
        # Try all four directions: up, down, left, right
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                swap_pos = new_row * 3 + new_col
                if state[swap_pos] != 0:  # Only move non-empty tiles
                    new_state = state.copy()
                    new_state[empty_pos], new_state[swap_pos] = new_state[swap_pos], new_state[empty_pos]
                    successors.append(new_state)
    
    return sorted(successors)

def print_succ(state: list) -> None:

    successors = get_successors(state)
    for succ in successors:
        h = calculate_heuristic(succ)
        print(f"{str(succ)} h={h}")

def solve(state: list) -> None:
    goal_state = get_goal_state(state)
    pq = []  
    states_list = [] 
    visited = set()
    max_queue_length = 0
 
    h = calculate_heuristic(state)
    heapq.heappush(pq, (h, state, (0, h, -1)))
    
    while pq:
        max_queue_length = max(max_queue_length, len(pq))
        f, current_state, (g, h, parent_index) = heapq.heappop(pq)
        current_tuple = tuple(current_state)
        if current_tuple in visited:
            continue
        current_index = len(states_list)
        states_list.append((current_state, parent_index, g))
        visited.add(current_tuple)
        if current_state == goal_state:
            print(True)
            path = []
            curr_index = current_index
            while curr_index != -1:
                state, parent_idx, moves = states_list[curr_index]
                path.append((state, calculate_heuristic(state), moves))
                curr_index = parent_idx
            
            for state, h_val, moves in reversed(path):
                print(f"{str(state)} h={h_val} moves: {moves}")
            
            print(f"Max queue length: {max_queue_length}")
            return
        for succ in get_successors(current_state):
            succ_tuple = tuple(succ)
            if succ_tuple not in visited:
                succ_h = calculate_heuristic(succ)
                succ_g = g + 1
                succ_f = succ_g + succ_h
                heapq.heappush(pq, (succ_f, succ, (succ_g, succ_h, current_index)))
    
    print(False)