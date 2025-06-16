import heapq
from typing import List, Tuple, Set
import time

class HanoiState:
    """
    Using tuples because they are immutable and I dont want anything to be changed
    Tuples are hashable and help avoid cycles
    """
    
    def __init__(self, pegs: Tuple[Tuple[int, ...], ...]):
        """
        Each peg is a tuple of disk numbers, bottom to top
        example: ((3,2,1), (), ()) means all disks on first peg, largest at bottom
        """

        self.pegs = pegs
        self.num_disks = sum(len(peg) for peg in pegs) # sum of number of disks initially 
    
    def __eq__(self, other):
        """
        This method is essential for A* to not WASTE time exploring multiple positions twice!
        Avoids infinite loops and memory efficient
        """
        return self.pegs == other.pegs
    
    def __hash__(self):
        """
        VERY important since this lets us access the states directly lightning fast :) O(n) vs O(1), used in pair with __eq__
        to quickly check with 2 states
        """
        return hash(self.pegs)
    
    def __str__(self):
        """
        printing for checking solution
        """
        result = []
        for i, peg in enumerate(self.pegs):
            result.append(f" {tuple(peg) if peg else '____'}")
        return " | ".join(result)
    
    def is_goal(self):
        """
        Goal state: all disks on the last peg, in correct order :)
        Moving everything to the last peg, so this method just checks if that is true
        """
        return (len(self.pegs[0]) == 0 and 
                len(self.pegs[1]) == 0 and 
                len(self.pegs[2]) == self.num_disks)
    
    def get_legal_moves(self):
        """
        The expansion operator - generates all possible next states
        
        Hanoi rules:
        1. Move only one disk at a time
        2. Only move the top disk from a peg
        3. Never put a larger disk on a smaller disk
        """
        moves = []
        
        # Try moving from each peg to each other peg
        for from_peg in range(3):
            if not self.pegs[from_peg]:  # Skip empty pegs
                continue
                
            top_disk = self.pegs[from_peg][-1]  # Top disk (last elem in tuple)
            
            for to_peg in range(3):
                if from_peg == to_peg:  # Can't move to same peg
                    continue
                
                # Check if move is legal (smaller disk on top)
                if not self.pegs[to_peg] or top_disk < self.pegs[to_peg][-1]:
                    moves.append(self._make_move(from_peg, to_peg))
        return moves
    
    def _make_move(self, from_peg: int, to_peg: int):
        """
        Creates a new state after making a move
        We don't modify the current state - we create a new one
        """
        new_pegs = list(self.pegs)  # Convert to list for modification
        
        # Remove disk from source peg
        disk = new_pegs[from_peg][-1]
        new_pegs[from_peg] = new_pegs[from_peg][:-1] # list slice
        
        # Add disk to destination peg
        new_pegs[to_peg] = new_pegs[to_peg] + (disk,)
        
        return HanoiState(tuple(new_pegs))


def heuristic(state: HanoiState):
    """
    This is the method responsible for the heuristics of A* algorithm 
    
    This heuristic counts disks not on the final peg. It's valid because:
    - Each disk not on the goal peg needs at least 1 move to get there
    - We never overestimate the actual cost
    - Valid heuristics guarantee optimal solutions in A*
    
    It is simple and works well, (You dont need a GPS to go to the kitchen ;))
    """
    # Count disks not on the rightmost (goal) peg
    return state.num_disks - len(state.pegs[2])


def astar_hanoi(initial_state: HanoiState):
    """
    A* Search Algorithm
    
    - it acts by exploring paths that seem the most promising
    - f(state) = g(state) + h(state)
    - g(state) = actual cost to reach this state (number of moves so far)
    - h(state) = estimated cost from this state to goal (our heuristic)
    - f(state) = estimated total cost of path through this state
    """
    
    # Priority queue: (f_score, unique_id, g_score, state, path)
    # unique_id prevents comparison of states when f_scores are equal
    open_set = [(heuristic(initial_state), 0, 0, initial_state, [initial_state])]
    closed_set: Set[HanoiState] = set()  # empty set to rememeber the states we've already seen
    nodes_explored = 0
    unique_id = 1
    
    print("Starting A* search...")
    print(f"Initial state: {initial_state}")
    print(f"Initial heuristic: {heuristic(initial_state)}")
    print()
    
    while open_set: # keep going until there are no more states to explore
        # Get the most promising state (lowest f_score)
        f_score, _, g_score, current_state, path = heapq.heappop(open_set) 
        
        if current_state in closed_set: # skip neighbors that have been explored aleady
            continue
            
        closed_set.add(current_state)
        nodes_explored += 1
        
        # Print progress every few nodes
        if nodes_explored % 10 == 0:
            print(f"Explored {nodes_explored} nodes, current f_score: {f_score}")
        
        # Check if we've reached the goal
        if current_state.is_goal():
            print(f"\nSolution found!")
            print(f"Nodes explored: {nodes_explored}")
            print(f"Solution length: {len(path) - 1} moves")
            return path, nodes_explored
        
        # Explore all possible moves from current state
        for next_state in current_state.get_legal_moves():
            if next_state in closed_set:
                continue
            
            # this essentially calculates the cost and builds paths for this neighboor
            new_g_score = g_score + 1  # Each move costs 1
            new_f_score = new_g_score + heuristic(next_state)
            new_path = path + [next_state]
            
            # add neighbor to queue with its priority and increment counter
            heapq.heappush(open_set, (new_f_score, unique_id, new_g_score, next_state, new_path))
            unique_id += 1
    
    return [], nodes_explored  # No solution found


def create_initial_state(num_disks: int):
    """
    Creates the starting state: all disks on the first peg
    Disks are numbered 1 to num_disks, where 1 is smallest
    """
    first_peg = tuple(range(num_disks, 0, -1))  # (3, 2, 1) for 3 disks
    return HanoiState((first_peg, (), ()))


def print_solution(solution: List[HanoiState]):
    if not solution:
        print("No solution found!")
        return
    
    print(f"\nSolution with {len(solution) - 1} moves:")
    print("-" * 42)
    
    for i, state in enumerate(solution):
        print(f"Step {i}: {state}")
        if i < len(solution) - 1:
            print()
    
    print("-"* 42)



test_cases = [6, 7, 8]  # Number of disks to test

for num_disks in test_cases:
    print(f"solving tower of hanoi with  {num_disks} disks")
    print()
    print()
    
    # Create initial state
    initial_state = create_initial_state(num_disks)
    
    # The minimum number of moves is 2^n - 1
    min_moves = 2**num_disks - 1
    print(f" minimum moves: {min_moves}")
    
    # Solve using A*
    start_time = time.time()
    solution, nodes_explored = astar_hanoi(initial_state)
    end_time = time.time()

    if solution:
        actual_moves = len(solution) - 1 # we include starting state so we want to get rid of it as it doesnt actually count as a move
        print(f"A* found solution with {actual_moves} moves")
        print(f"Optimal? {'Yes' if actual_moves == min_moves else 'No'}")
        print(f"Time taken: {end_time - start_time:.3f} seconds")
        
        # Print solution for smaller problems
        if num_disks <= 8:
            print_solution(solution)
    else:
        print("No solution found!")
    
    print()