# genetic_level_generator.py
import random
import settings
from collections import deque
import math
import statistics

# --- Helper Functions ---

# Modify the function signature and usage
def get_difficulty_params(generation, difficulty_stages_list): # Check this signature
    """Gets the number of Wumpuses and Pits for the current generation from the provided list."""
    if not difficulty_stages_list:
        print("Error: Difficulty stages list is empty in get_difficulty_params.")
        # Return initial defaults as a fallback if list is missing
        # Make sure INITIAL_WUMPUS and INITIAL_PITS are accessible here
        # (They should be if defined at the top level of the module)
        return settings.INITIAL_WUMPUS, settings.INITIAL_PITS 

    for end_gen, num_w, num_p in difficulty_stages_list:
        # Use generation number directly (0-indexed) vs end_gen (1-indexed boundary)
        if generation < end_gen: # If current gen is BEFORE the end of this stage
            return num_w, num_p
    # If generation is >= end_gen of the last stage, return last stage's params
    return difficulty_stages_list[-1][1], difficulty_stages_list[-1][2]

def get_empty_squares(grid, allow_start=False):
    """Returns list of (y, x) for empty squares. Optionally includes (0,0)."""
    empty = []
    for r in range(settings.GRID_SIZE):
        for c in range(settings.GRID_SIZE):
            if grid[r][c] == settings.EMPTY:
                if allow_start or (r, c) != (0, 0):
                    empty.append((r, c))
    return empty

def create_random_individual(num_wumpus, num_pits):
    """Creates a grid with specified numbers of Wumpus, Gold, Pits."""
    grid = [[settings.EMPTY for _ in range(settings.GRID_SIZE)] for _ in range(settings.GRID_SIZE)]
    potential_spots = get_empty_squares(grid, allow_start=False) # Initial placement away from start

    num_entities = num_wumpus + num_pits + 1 # +1 for Gold
    if len(potential_spots) < num_entities:
        raise ValueError(f"Grid size {settings.GRID_SIZE} too small for {num_entities} entities away from start.")

    random.shuffle(potential_spots)

    entities_to_place = ([settings.WUMPUS] * num_wumpus +
                         [settings.GOLD] * 1 +
                         [settings.PIT] * num_pits)
    random.shuffle(entities_to_place) # Mix entity types during placement

    for entity_type in entities_to_place:
         if not potential_spots:
              raise RuntimeError("Ran out of unique non-start spots during initial placement.")
         y, x = potential_spots.pop()
         grid[y][x] = entity_type

    return grid

def is_safe(grid, y, x):
    """Checks if a square is within bounds and not any hazard."""
    return 0 <= y < settings.GRID_SIZE and \
           0 <= x < settings.GRID_SIZE and \
           grid[y][x] != settings.WUMPUS and \
           grid[y][x] != settings.PIT

def find_safe_path(grid):
    """BFS for safe path from (0,0) to Gold. Returns (path_length, gold_pos)."""
    q = deque([(0, 0, 0)]) # (y, x, distance)
    visited = set([(0, 0)])
    gold_pos = None

    for r in range(settings.GRID_SIZE):
        for c in range(settings.GRID_SIZE):
            if grid[r][c] == settings.GOLD:
                gold_pos = (r, c)
                break
        if gold_pos: break

    if not gold_pos: return None, None

    if not is_safe(grid, 0, 0):
         return None, gold_pos

    while q:
        y, x, dist = q.popleft()

        if (y, x) == gold_pos:
            return dist, gold_pos

        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ny, nx = y + dy, x + dx
            if is_safe(grid, ny, nx) and (ny, nx) not in visited:
                visited.add((ny, nx))
                q.append((ny, nx, dist + 1))

    return None, gold_pos

def get_entity_positions(grid):
    """Helper to get positions of all entities."""
    # Now returns lists for Wumpus and Pit
    positions = {settings.WUMPUS: [], settings.PIT: [], settings.GOLD: None}
    for r in range(settings.GRID_SIZE):
        for c in range(settings.GRID_SIZE):
            entity = grid[r][c]
            if entity == settings.WUMPUS:
                positions[settings.WUMPUS].append((r, c))
            elif entity == settings.PIT:
                positions[settings.PIT].append((r, c))
            elif entity == settings.GOLD:
                positions[settings.GOLD] = (r, c)
    return positions

def calculate_fitness(grid, target_num_wumpus, target_num_pits):
    """Calculates fitness, considering target entity counts and multiple hazards."""
    fitness = 0
    entity_positions = get_entity_positions(grid)
    wumpus_positions = entity_positions[settings.WUMPUS]
    pit_positions = entity_positions[settings.PIT]
    gold_pos = entity_positions[settings.GOLD]
    all_hazard_pos = wumpus_positions + pit_positions

    # --- Validation Penalty (Optional but good) ---
    # Penalize if the grid doesn't have the *correct number* of entities
    # (Could happen due to bad crossover/mutation recovery)
    validation_penalty = 0
    # Use get_entity_positions AFTER potential modifications by crossover/mutation
    current_entities = get_entity_positions(grid)
    current_wumpus_count = len(current_entities[settings.WUMPUS])
    current_pit_count = len(current_entities[settings.PIT])
    current_gold_exists = current_entities[settings.GOLD] is not None

    if current_wumpus_count != target_num_wumpus:
        validation_penalty += -1000 * abs(current_wumpus_count - target_num_wumpus) # Increased penalty
    if current_pit_count != target_num_pits:
         validation_penalty += -1000 * abs(current_pit_count - target_num_pits) # Increased penalty
    if not current_gold_exists:
         validation_penalty += -2000 # Increased penalty

    fitness += validation_penalty
    if validation_penalty < 0: # If invalid, don't bother with other calcs? Or maybe still penalize bad placement?
         # Let's calculate others too, maybe it can recover.
         pass


    # 1. Solvability and Path Length
    path_length, found_gold_pos = find_safe_path(grid)
    gold_pos = found_gold_pos if found_gold_pos else gold_pos

    if path_length is not None:
        fitness += settings.SOLVABILITY_WEIGHT
        fitness += path_length * settings.PATH_LENGTH_WEIGHT
    else:
        fitness -= settings.SOLVABILITY_WEIGHT * 1.5

    # 2. Start Penalties (Applied per hazard)
    start_neighbors = [(0, 1), (1, 0), (1,1)]
    min_hazard_dist_from_start = float('inf')
    hazards_near_start = 0

    for hy, hx in all_hazard_pos:
        if (hy, hx) in start_neighbors or (hy,hx) == (0,0):
            hazards_near_start += 1
        dist = abs(hy - 0) + abs(hx - 0)
        min_hazard_dist_from_start = min(min_hazard_dist_from_start, dist if dist > 0 else float('inf')) # Ignore dist 0

    fitness += hazards_near_start * settings.HAZARD_START_PENALTY # Penalty scales with number near start

    if gold_pos and (gold_pos in start_neighbors or gold_pos == (0,0)):
        fitness += settings.GOLD_START_PENALTY

    # 3. Min Hazard Distance Reward
    if min_hazard_dist_from_start > 1.5:
         fitness += settings.MIN_HAZARD_DISTANCE_REWARD

    # 4. Hazard Adjacency to Gold Penalty (Applied per hazard)
    hazards_near_gold = 0
    if gold_pos:
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ny, nx = gold_pos[0] + dy, gold_pos[1] + dx
            if (ny, nx) in all_hazard_pos:
                hazards_near_gold += 1
        fitness += hazards_near_gold * settings.HAZARD_ADJACENT_GOLD_PENALTY

    # 5. Hazard Clustering Penalty
    if len(all_hazard_pos) > 1:
        total_dist_sq = 0
        pairs = 0
        for i in range(len(all_hazard_pos)):
            for j in range(i + 1, len(all_hazard_pos)):
                y1, x1 = all_hazard_pos[i]
                y2, x2 = all_hazard_pos[j]
                dist_sq = (y1 - y2)**2 + (x1 - x2)**2
                total_dist_sq += dist_sq
                pairs += 1
        if pairs > 0:
            avg_dist_sq = total_dist_sq / pairs
            if avg_dist_sq < 2.0: # Penalize if avg squared distance is < 2 (very close)
                 fitness += settings.HAZARD_CLUSTERING_PENALTY * (2.0 - avg_dist_sq)

    return fitness

# --- GA Operators ---

def tournament_selection(population, fitnesses):
    """Selects an individual using tournament selection."""
    actual_tournament_size = min(settings.TOURNAMENT_SIZE, len(population))
    if actual_tournament_size <= 0: return None
    selected_indices = random.sample(range(len(population)), actual_tournament_size)
    best_index = max(selected_indices, key=lambda index: fitnesses[index])
    return [row[:] for row in population[best_index]] # Return copy


def crossover(parent1, parent2, target_num_wumpus, target_num_pits):
    """
    Crossover ensuring offspring have the target number of entities for the current difficulty.
    """
    if random.random() > settings.CROSSOVER_RATE:
        return [row[:] for row in parent1], [row[:] for row in parent2]

    target_counts = {settings.WUMPUS: target_num_wumpus,
                     settings.GOLD: 1,
                     settings.PIT: target_num_pits}

    # Extract unique positions from parents for pooling
    pos_pool = {etype: set() for etype in target_counts}
    all_valid_parent_pos = set()

    for p in [parent1, parent2]:
         entities = get_entity_positions(p)
         for etype, pos_list_or_val in entities.items():
              if etype == settings.GOLD:
                   if pos_list_or_val:
                        pos_pool[etype].add(pos_list_or_val)
                        all_valid_parent_pos.add(pos_list_or_val)
              elif etype in pos_pool: # Wumpus, Pit
                   for pos in pos_list_or_val:
                        pos_pool[etype].add(pos)
                        all_valid_parent_pos.add(pos)

    # Convert sets to lists and shuffle for selection
    for etype in pos_pool:
        pos_pool[etype] = list(pos_pool[etype])
        random.shuffle(pos_pool[etype])

    offspring1_grid = [[settings.EMPTY for _ in range(settings.GRID_SIZE)] for _ in range(settings.GRID_SIZE)]
    offspring2_grid = [[settings.EMPTY for _ in range(settings.GRID_SIZE)] for _ in range(settings.GRID_SIZE)]
    assigned_pos1 = set()
    assigned_pos2 = set()

    # Find all empty squares in parents (potential fill spots)
    parent_empty = get_empty_squares(parent1, allow_start=True) + get_empty_squares(parent2, allow_start=True)
    random.shuffle(parent_empty)
    fallback_spots = [p for p in parent_empty if p not in all_valid_parent_pos]


    # Assign entities ensuring target counts and minimal overlap
    for entity_type, count in target_counts.items():
        available = pos_pool[entity_type][:] # Copy available positions

        # Assign to offspring 1
        placed_count = 0
        idx = 0
        while placed_count < count and idx < len(available):
            pos = available[idx]
            if pos not in assigned_pos1:
                offspring1_grid[pos[0]][pos[1]] = entity_type
                assigned_pos1.add(pos)
                placed_count += 1
            idx += 1

        # Fill remaining for offspring 1 using fallback empty spots if needed
        fb_idx = 0
        while placed_count < count and fb_idx < len(fallback_spots):
             pos = fallback_spots[fb_idx]
             if pos not in assigned_pos1:
                  offspring1_grid[pos[0]][pos[1]] = entity_type
                  assigned_pos1.add(pos)
                  placed_count += 1
             fb_idx +=1
        # If still not enough (highly unlikely), place randomly in remaining empty
        while placed_count < count:
            final_empty = get_empty_squares(offspring1_grid, allow_start=True)
            valid_final = [p for p in final_empty if p not in assigned_pos1]
            if not valid_final: break # Error
            pos = random.choice(valid_final)
            offspring1_grid[pos[0]][pos[1]] = entity_type
            assigned_pos1.add(pos)
            placed_count += 1


        # Assign to offspring 2 (using remaining from pool + fallbacks)
        placed_count = 0
        # Use remaining from original 'available' list first
        while placed_count < count and idx < len(available):
             pos = available[idx]
             if pos not in assigned_pos2:
                  offspring2_grid[pos[0]][pos[1]] = entity_type
                  assigned_pos2.add(pos)
                  placed_count += 1
             idx += 1
        # Fill remaining using fallback
        while placed_count < count and fb_idx < len(fallback_spots):
             pos = fallback_spots[fb_idx]
             if pos not in assigned_pos2:
                  offspring2_grid[pos[0]][pos[1]] = entity_type
                  assigned_pos2.add(pos)
                  placed_count += 1
             fb_idx += 1
        # Final random placement if needed
        while placed_count < count:
            final_empty = get_empty_squares(offspring2_grid, allow_start=True)
            valid_final = [p for p in final_empty if p not in assigned_pos2]
            if not valid_final: break
            pos = random.choice(valid_final)
            offspring2_grid[pos[0]][pos[1]] = entity_type
            assigned_pos2.add(pos)
            placed_count += 1

    return offspring1_grid, offspring2_grid


def mutate(grid, current_mutation_rate):
    """Mutation: Move entity or Swap two entities."""
    mutated_grid = [row[:] for row in grid]
    mutations_applied = 0

    for _ in range(settings.MAX_MUTATIONS_PER_INDIVIDUAL):
        if random.random() < current_mutation_rate:
            entity_positions = get_entity_positions(mutated_grid)
            # Flatten entity list: [(type, pos), (type, pos), ...]
            entities_present = []
            if entity_positions[settings.GOLD]: entities_present.append((settings.GOLD, entity_positions[settings.GOLD]))
            entities_present.extend([(settings.WUMPUS, pos) for pos in entity_positions[settings.WUMPUS]])
            entities_present.extend([(settings.PIT, pos) for pos in entity_positions[settings.PIT]])

            if not entities_present: continue

            if random.random() < 0.7 or len(entities_present) < 2:
                # Move Mutation
                entity_type, (old_y, old_x) = random.choice(entities_present)
                empty_spots = get_empty_squares(mutated_grid, allow_start=True)
                valid_new_spots = [pos for pos in empty_spots if pos != (old_y, old_x)]
                if not valid_new_spots: continue

                new_y, new_x = random.choice(valid_new_spots)
                mutated_grid[old_y][old_x] = settings.EMPTY
                mutated_grid[new_y][new_x] = entity_type
                mutations_applied += 1
            else:
                # Swap Mutation
                idx1, idx2 = random.sample(range(len(entities_present)), 2)
                type1, (y1, x1) = entities_present[idx1]
                type2, (y2, x2) = entities_present[idx2]
                mutated_grid[y1][x1] = type2
                mutated_grid[y2][x2] = type1
                mutations_applied += 1

    return mutated_grid

def print_grid_visualization(grid, actual_num_w, actual_num_p): # Changed params
     """Prints grid visualization showing ACTUAL counts found."""
     # Use actual counts in the title for clarity
     print(f"--- Grid (Actual: {actual_num_w}W, {actual_num_p}P, 1G Present: {get_entity_positions(grid)[settings.GOLD] is not None}) ---")
     print("-" * (settings.GRID_SIZE * 2 + 1))
     for r in range(settings.GRID_SIZE):
          row_str = ["|"]
          for c in range(settings.GRID_SIZE):
               char = '.'
               entity = grid[r][c]
               is_start_and_empty = (r == 0 and c == 0 and entity == settings.EMPTY)

               if entity == settings.WUMPUS: char = 'W'
               elif entity == settings.PIT: char = 'P'
               elif entity == settings.GOLD: char = 'G'

               if r==0 and c==0 and not is_start_and_empty: # If start has entity or clue
                   pass # Character already set
               elif is_start_and_empty:
                   char = 'S' # Explicitly mark empty start

               row_str.append(char)
          row_str.append("|")
          print(" ".join(row_str))
     print("-" * (settings.GRID_SIZE * 2 + 1))

# --- Main GA Function ---

def generate_dynamic_difficulty_stages(total_gens, num_stages,
                                     init_w, init_p,
                                     pit_incr, wump_freq):
    """Generates the DIFFICULTY_STAGES list automatically."""
    if num_stages <= 0:
        # Default to one stage covering all generations if input is invalid
        print("Warning: NUM_DIFFICULTY_STAGES must be > 0. Defaulting to 1 stage.")
        return [(total_gens, init_w, init_p)]
    if wump_freq <= 0:
        print("Warning: WUMPUS_INCREMENT_FREQUENCY must be > 0. Wumpus count will not increase.")
        wump_freq = float('inf') # Effectively disable wumpus increment


    stages = []
    base_gens_per_stage = total_gens // num_stages
    remainder_gens = total_gens % num_stages

    current_wumpus = init_w
    current_pits = init_p
    last_gen_end = 0

    # Calculate max possible entities based on grid size to prevent impossible scenarios
    max_placeable_entities = (settings.GRID_SIZE * settings.GRID_SIZE) - 1 # Exclude start square (0,0)

    for i in range(num_stages):
        gens_this_stage = base_gens_per_stage + (1 if i < remainder_gens else 0)
        current_gen_end = last_gen_end + gens_this_stage

        # Ensure the final stage ends exactly at total_gens
        if i == num_stages - 1:
            current_gen_end = total_gens

        # --- Add current stage data ---
        # Clamp entity counts if they exceed reasonable limits for the grid
        total_entities = current_wumpus + current_pits + 1 # +1 for Gold
        if total_entities > max_placeable_entities:
             print(f"Warning: Calculated entity count ({total_entities}) exceeds grid capacity ({max_placeable_entities}) at stage {i+1}. Clamping counts.")
             # Prioritize reducing pits first if clamping needed
             pits_to_remove = total_entities - max_placeable_entities
             clamped_pits = max(1, current_pits - pits_to_remove) # Ensure at least 1 pit if possible
             pits_removed = current_pits - clamped_pits
             wumpus_to_remove = max(0, (total_entities - max_placeable_entities) - pits_removed)
             clamped_wumpus = max(1, current_wumpus - wumpus_to_remove) # Ensure at least 1 Wumpus

             stages.append((current_gen_end, clamped_wumpus, clamped_pits))
             # Stop further increments if max capacity reached
             current_wumpus = clamped_wumpus
             current_pits = clamped_pits
             # Prevent future increments if already clamped heavily? Or let it try? Let it try for now.

        else:
             stages.append((current_gen_end, current_wumpus, current_pits))


        # --- Prepare counts for the *next* stage ---
        next_pits = current_pits + pit_incr
        next_wumpus = current_wumpus

        # Check if Wumpus count should increase (based on stage number, 1-indexed)
        if (i + 1) % wump_freq == 0:
            next_wumpus += 1

        # Update counts for the next iteration (only if not already clamped at max)
        if (current_wumpus + current_pits + 1) < max_placeable_entities:
             current_pits = next_pits
             current_wumpus = next_wumpus
        # Else keep counts clamped from previous check


        last_gen_end = current_gen_end

    print("Generated Difficulty Stages:")
    for end_g, nw, np in stages:
        print(f"  - Up to Gen {end_g}: {nw} Wumpus, {np} Pits")
    return stages

def generate_levels_over_generations():
    """Runs GA, saves best grids periodically based on increasing difficulty."""
    print("Starting Genetic Algorithm with Increasing Difficulty...")

    # --- Generate Difficulty Stages Dynamically ---
    DIFFICULTY_STAGES = generate_dynamic_difficulty_stages(
        total_gens=settings.NUM_GENERATIONS,
        num_stages=settings.NUM_DIFFICULTY_STAGES,
        init_w=settings.INITIAL_WUMPUS,
        init_p=settings.INITIAL_PITS,
        pit_incr=settings.PIT_INCREMENT_PER_STAGE,
        wump_freq=settings.WUMPUS_INCREMENT_FREQUENCY
    )
    if not DIFFICULTY_STAGES:
         print("Error: Could not generate difficulty stages. Exiting.")
         return [] # Return empty list
    # --- End Stage Generation ---


    saved_grids = [] # List to store results
    population = []

    prev_num_wumpus, prev_num_pits = -1, -1

    best_overall_fitness_ever = -float('inf')
    current_mutation_rate = settings.INITIAL_MUTATION_RATE

    for generation in range(settings.NUM_GENERATIONS):
        # Get difficulty for this generation
        current_num_wumpus, current_num_pits = get_difficulty_params(generation, DIFFICULTY_STAGES) # Pass the list

        # --- Population Initialization / Injection on Difficulty Change ---
        if current_num_wumpus != prev_num_wumpus or current_num_pits != prev_num_pits:
            print(f"\n--- Difficulty Change Detected at Gen {generation+1}: Target -> {current_num_wumpus}W, {current_num_pits}P ---")
            if not population: # First generation initialization
                 print("Initializing population...")
                 population = [create_random_individual(current_num_wumpus, current_num_pits) for _ in range(settings.POPULATION_SIZE)]
            else:
                 # Inject new individuals matching the NEW difficulty
                 num_to_inject = int(settings.POPULATION_SIZE * 0.3) # Inject 30% new blood
                 print(f"Injecting {num_to_inject} new individuals...")
                 # Option 1: Replace worst individuals
                 # Option 2: Replace random individuals
                 # Let's replace random for simplicity
                 indices_to_replace = random.sample(range(settings.POPULATION_SIZE), num_to_inject)
                 new_individuals = [create_random_individual(current_num_wumpus, current_num_pits) for _ in range(num_to_inject)]
                 inject_idx = 0
                 for pop_idx in indices_to_replace:
                      population[pop_idx] = new_individuals[inject_idx]
                      inject_idx += 1
                 # Could also re-sort population after injection if replacing worst

            # Update tracked difficulty
            prev_num_wumpus = current_num_wumpus
            prev_num_pits = current_num_pits
        elif not population: # Handle case if first stage check didn't init
             print("Error: Population not initialized.")
             return []

        # --- Evaluation (using current difficulty targets) ---
        fitnesses = [calculate_fitness(grid, current_num_wumpus, current_num_pits) for grid in population]

        # --- Diversity Check & Mutation Rate Adaptation ---
        # (Keep existing diversity check logic)
        diversity_boost = False
        if len(fitnesses) > 1:
            try:
                fitness_std_dev = statistics.stdev(fitnesses)
                if fitness_std_dev < settings.DIVERSITY_THRESHOLD: diversity_boost = True
            except statistics.StatisticsError: diversity_boost = True # Boost if all same
        else: diversity_boost = False

        current_mutation_rate = max(settings.FINAL_MUTATION_RATE, settings.INITIAL_MUTATION_RATE - generation * settings.MUTATION_RATE_DECAY)
        effective_mutation_rate = min(0.9, current_mutation_rate * (settings.MUTATION_BOOST_FACTOR if diversity_boost else 1.0))


        # --- Selection & Conditional Elitism ---
        sorted_indices = sorted(range(len(population)), key=lambda k: fitnesses[k], reverse=True)
        new_population = []

        elites_added = 0
        idx = 0
        # Iterate through sorted individuals to find valid elites
        while elites_added < settings.ELITE_COUNT and idx < len(sorted_indices):
            elite_candidate_index = sorted_indices[idx]
            elite_candidate_grid = population[elite_candidate_index]

            # Check if the candidate actually matches the CURRENT target counts
            candidate_entities = get_entity_positions(elite_candidate_grid)
            w_count = len(candidate_entities[settings.WUMPUS])
            p_count = len(candidate_entities[settings.PIT])
            g_exists = candidate_entities[settings.GOLD] is not None

            if w_count == current_num_wumpus and p_count == current_num_pits and g_exists:
                 # This elite is valid for the current difficulty
                 new_population.append([row[:] for row in elite_candidate_grid]) # Add copy
                 elites_added += 1
            # Else: Skip this candidate, it doesn't match current targets

            idx += 1
        
        if elites_added < settings.ELITE_COUNT:
             print(f"  Warning: Only found {elites_added} valid elites matching target counts.")


        # Track best of generation (among ALL individuals, even if count is wrong temporarily)
        current_best_index = sorted_indices[0]
        current_best_fitness = fitnesses[current_best_index]
        current_best_grid = population[current_best_index] # Grid with highest score this gen
        best_overall_fitness_ever = max(best_overall_fitness_ever, current_best_fitness)

        print(f"Generation {generation+1}/{settings.NUM_GENERATIONS} ({current_num_wumpus}W, {current_num_pits}P) - "
              f"Best Fitness: {current_best_fitness:.2f} (Overall Max: {best_overall_fitness_ever:.2f}) - "
              f"Mut Rate: {effective_mutation_rate:.3f}" + (" (Boosted)" if diversity_boost else ""))


        # --- Save grid periodically ---
        # Save the grid with the highest fitness *in this generation*,
        # regardless of whether it perfectly matches the target count yet.
        # The visualization will show its actual content.
        if (generation + 1) % settings.SAVE_GRID_EVERY_N_GEN == 0 or generation == settings.NUM_GENERATIONS - 1:
            print(f"--- Saving best grid layout from generation {generation + 1} (Fitness: {current_best_fitness:.2f}) ---")
            # Visualize the *actual* content of the best grid found this gen
            actual_entities = get_entity_positions(current_best_grid)
            actual_w = len(actual_entities[settings.WUMPUS])
            actual_p = len(actual_entities[settings.PIT])
            print_grid_visualization(current_best_grid, actual_w, actual_p) # Use actual counts for viz title

            # Store data for playing later
            saved_grids.append({
                "generation": generation + 1,
                "grid": [row[:] for row in current_best_grid],
                # Store the TARGET counts for context when playing
                "target_num_wumpus": current_num_wumpus,
                "target_num_pits": current_num_pits,
                "fitness": current_best_fitness,
                # Store actual counts found in this grid too
                "actual_num_wumpus": actual_w,
                "actual_num_pits": actual_p
            })

        # --- Reproduction ---
        while len(new_population) < settings.POPULATION_SIZE:
             parent1 = tournament_selection(population, fitnesses)
             parent2 = tournament_selection(population, fitnesses)
             if parent1 is None or parent2 is None:
                 if not population: break # Avoid error if population became empty
                 parent1 = parent1 or [row[:] for row in random.choice(population)]
                 parent2 = parent2 or [row[:] for row in random.choice(population)]

             # Crossover uses the TARGET counts for the CURRENT generation
             offspring1, offspring2 = crossover(parent1, parent2, current_num_wumpus, current_num_pits)

             # Mutation
             offspring1 = mutate(offspring1, effective_mutation_rate)
             offspring2 = mutate(offspring2, effective_mutation_rate)

             new_population.append(offspring1)
             if len(new_population) < settings.POPULATION_SIZE:
                 new_population.append(offspring2)

        population = new_population # Next generation

    print("\nGenetic Algorithm Finished.")
    return saved_grids

# --- Example Usage ---
if __name__ == "__main__":
    generated_data = generate_levels_over_generations()
    print(f"\nGenerated {len(generated_data)} level snapshots to play.")
    for data in generated_data:
        print(f" - Generation {data['generation']} (Fitness: {data['fitness']:.2f}, {data['num_wumpus']}W, {data['num_pits']}P)")
    # In a real scenario, main.py would now take 'generated_data' and run the games.