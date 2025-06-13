# game.py
import pygame
import random
import settings

class WumpusWorld:
    def __init__(self, initial_grid=None):
        if initial_grid:
            self.grid = [row[:] for row in initial_grid] # Make a copy
        else:
            # Fallback: Create a default random grid (e.g., 1 Wumpus, 3 Pits)
            # This part might not be used if main.py always provides a grid
            print("Warning: No initial grid provided to WumpusWorld, creating random.")
            nw, np = 1, 3 # Default fallback difficulty
            self.grid = [[settings.EMPTY for _ in range(settings.GRID_SIZE)] for _ in range(settings.GRID_SIZE)]
            self._fallback_place_entities(nw, np)

        self.player_pos = [0, 0]
        self.revealed_tiles = [[False for _ in range(settings.GRID_SIZE)] for _ in range(settings.GRID_SIZE)]
        # Reveal starting position immediately
        self.reveal_tile(self.player_pos[0], self.player_pos[1])
        # Check initial Wumpus presence
        self.wumpus_positions = self._find_wumpuses()
        self.wumpus_alive_count = len(self.wumpus_positions)
        self.initial_wumpus_count = self.wumpus_alive_count # Store for reference if needed
        #self.has_arrow = True # Start with one arrow (can be adjusted)
        self.arrow_count = self.wumpus_alive_count
        self.has_arrow=self.arrow_count
        self.load_assets()
        self.add_sensory_clues() # Add clues based on the initial grid

    def reveal_tile(self, x, y):
        """Marks a tile as revealed."""
        if 0 <= y < settings.GRID_SIZE and 0 <= x < settings.GRID_SIZE:
            self.revealed_tiles[y][x] = True

    def _fallback_place_entities(self, num_w, num_p):
        """Used only if no initial_grid is provided."""
        entities_to_place = ([settings.WUMPUS] * num_w +
                             [settings.GOLD] * 1 +
                             [settings.PIT] * num_p)
        for entity in entities_to_place:
            self.place_random(entity)

    def _find_wumpuses(self):
        """Finds all Wumpus locations on the current grid."""
        positions = []
        for r in range(settings.GRID_SIZE):
            for c in range(settings.GRID_SIZE):
                if self.grid[r][c] == settings.WUMPUS:
                    positions.append((r, c))
        return positions

    def load_assets(self):
        """Load sprites."""
        # (Keep existing load_assets, ensure paths are correct)
        try:
            player_img = pygame.image.load("assets/old/player.png")
            self.player_img = pygame.transform.scale(player_img, (settings.TILE_SIZE, settings.TILE_SIZE))

            wumpus_img = pygame.image.load("assets/old/wumpus.png")
            self.wumpus_img = pygame.transform.scale(wumpus_img, (settings.TILE_SIZE, settings.TILE_SIZE))

            pit_img = pygame.image.load("assets/old/pit.png")
            self.pit_img = pygame.transform.scale(pit_img, (settings.TILE_SIZE, settings.TILE_SIZE))

            gold_img = pygame.image.load("assets/old/gold.png")
            self.gold_img = pygame.transform.scale(gold_img, (settings.TILE_SIZE, settings.TILE_SIZE))

            breeze_img = pygame.image.load("assets/old/breeze.png")
            self.breeze_img = pygame.transform.scale(breeze_img, (settings.TILE_SIZE, settings.TILE_SIZE))

            stench_img = pygame.image.load("assets/old/stench.png")
            self.stench_img = pygame.transform.scale(stench_img, (settings.TILE_SIZE, settings.TILE_SIZE))
        except pygame.error as e:
             print(f"Error loading assets: {e}")
             pygame.quit()
             exit()

    def place_random(self, entity):
        """Places one entity randomly, avoiding (0,0)."""
        while True:
            x, y = random.randint(0, settings.GRID_SIZE - 1), random.randint(0, settings.GRID_SIZE - 1)
            if (y, x) != (0, 0) and self.grid[y][x] == settings.EMPTY:
                self.grid[y][x] = entity
                break

    def add_sensory_clues(self):
        """Adds stench/breeze clues around ALL Wumpuses/Pits."""
        # Clear existing clues first
        for y in range(settings.GRID_SIZE):
            for x in range(settings.GRID_SIZE):
                if self.grid[y][x] in [settings.BREEZE, settings.STENCH]:
                     self.grid[y][x] = settings.EMPTY

        # Add new clues
        for y in range(settings.GRID_SIZE):
            for x in range(settings.GRID_SIZE):
                if self.grid[y][x] == settings.WUMPUS:
                    self._add_adjacent_clue(x, y, settings.STENCH)
                elif self.grid[y][x] == settings.PIT:
                    self._add_adjacent_clue(x, y, settings.BREEZE)

    def _add_adjacent_clue(self, x, y, clue):
        """Places a clue in adjacent EMPTY squares."""
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < settings.GRID_SIZE and 0 <= ny < settings.GRID_SIZE and self.grid[ny][nx] == settings.EMPTY:
                self.grid[ny][nx] = clue
                # Handle potential overlap? If STENCH is there, should BREEZE overwrite or add?
                # Simple: Overwrite EMPTY only. Complex: Use lists or bitflags per cell.
                # Keep it simple for now.

    def shoot(self, direction):
        """Shoot arrow. Handles hitting any Wumpus."""
        if not self.has_arrow:
            print("No arrows left!")
            return False

        #self.has_arrow = False
        self.has_arrow -= 1
        px, py = self.player_pos
        ax, ay = px, py # Arrow start position

        # Determine direction vector
        if direction == "up": dy, dx = -1, 0
        elif direction == "down": dy, dx = 1, 0
        elif direction == "left": dy, dx = 0, -1
        elif direction == "right": dy, dx = 0, 1
        else:
            print("Invalid shoot direction")
            self.has_arrow = True # Give back arrow
            return False

        # Traverse path
        hit_wumpus = False
        while True:
             ax += dx
             ay += dy
             if not (0 <= ax < settings.GRID_SIZE and 0 <= ay < settings.GRID_SIZE):
                  break # Arrow left grid

             if self.grid[ay][ax] == settings.WUMPUS:
                  print(f"You hear a terrible scream from ({ay},{ax})!")
                  self.grid[ay][ax] = settings.EMPTY # Remove the Wumpus
                  self.wumpus_alive_count -= 1
                  print(f"Wumpuses remaining: {self.wumpus_alive_count}")
                  hit_wumpus = True
                  # Stench needs recalculation based on remaining wumpuses
                  self.add_sensory_clues() # Re-calculate all clues
                  break # Arrow stops after hitting

        if not hit_wumpus:
            print("Your arrow flies into the darkness... You missed!")

        return hit_wumpus


    def draw(self, screen):
        """Draw the game world, respecting the fog of war."""
        screen.fill(settings.FOG_COLOR) # Fill background with fog color initially

        for y in range(settings.GRID_SIZE):
            for x in range(settings.GRID_SIZE):
                rect = pygame.Rect(x * settings.TILE_SIZE, y * settings.TILE_SIZE, settings.TILE_SIZE, settings.TILE_SIZE)

                # --- Check if tile is revealed ---
                if self.revealed_tiles[y][x]:
                    # --- Draw Revealed Tile ---
                    # Optional: Slightly different background for revealed empty tiles?
                    # pygame.draw.rect(screen, (30, 30, 30), rect) # Example background

                    # Draw grid lines for revealed tiles
                    pygame.draw.rect(screen, (50, 50, 50), rect, 1)

                    # Draw content of the revealed tile
                    tile_content = self.grid[y][x]

                    if tile_content == settings.WUMPUS:
                        screen.blit(self.wumpus_img, rect)
                    elif tile_content == settings.PIT:
                        screen.blit(self.pit_img, rect)
                    elif tile_content == settings.GOLD:
                        screen.blit(self.gold_img, rect)
                    elif tile_content == settings.STENCH:
                        screen.blit(self.stench_img, rect)
                    elif tile_content == settings.BREEZE:
                        screen.blit(self.breeze_img, rect)
                    # Else: It's an empty revealed tile, background/gridline is enough
                    # --- End Revealed Tile Drawing ---

                # else:
                    # If tile is not revealed, the background fill covers it.
                    # No need to explicitly draw fog again unless you want a different color.
                    # pass # Covered by screen.fill(settings.FOG_COLOR)

        # --- Draw Player and UI on TOP of everything ---
        # Draw player on top
        px, py = self.player_pos
        player_rect = pygame.Rect(px * settings.TILE_SIZE, py * settings.TILE_SIZE, settings.TILE_SIZE, settings.TILE_SIZE)
        screen.blit(self.player_img, player_rect)

        # UI: Arrow count
        try:
            font = pygame.font.Font(None, 36)
            #arrow_text = font.render(f"Arrows: {1 if self.has_arrow else 0}", True, (255, 255, 255))
            arrow_text = font.render(f"Arrows: {self.arrow_count}", True, (255, 255, 255))
            screen.blit(arrow_text, (10, 10))
            # Removed Senses display as per previous request
        except pygame.error as e:
            print(f"Font rendering error for UI: {e}")