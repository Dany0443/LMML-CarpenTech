import sys
import json
import numpy as np

class Strategy:
    """
    Standard UCB with sample mean for multi-armed bandit.
    Preserves all observations including rare high-reward events (walu runs).
    """
    
    def __init__(self, c: float = 2.0):
        self.num_spots = 10
        self.c = c
        self.reset()
    
    def reset(self):
        """Reset strategy state for a new day."""
        self.counts = np.zeros(self.num_spots, dtype=int)
        self.sums = np.zeros(self.num_spots, dtype=float)
        self.total_casts = 0
        self.total_catch = 0.0
    
    def select_spot(self) -> int:
        """
        Select a spot using UCB on sample mean.
        """
        # Initial exploration: try each spot at least once
        unexplored = np.where(self.counts == 0)[0]
        if len(unexplored) > 0:
            return int(unexplored[0])
        
        means = self.sums / self.counts
        t = self.total_casts + 1
        log_t = np.log(t)
        
        ucb_values = means + self.c * np.sqrt(log_t / self.counts)
        
        return int(np.argmax(ucb_values))
    
    def update(self, spot_idx: int, catch: float):
        """
        Update strategy with observed catch.
        """
        self.counts[spot_idx] += 1
        self.sums[spot_idx] += catch
        self.total_casts += 1
        self.total_catch += catch


def main():
    """Main entry point. Reads test-input.json, then for each cast prints spot to stdout and reads catch from stdin."""
    # Read configuration from test-input.json
    config_path = "test-input.json"
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        # Fallback for local testing with train-input.json
        try:
            with open("train-input.json", "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            sys.stderr.write("Error: test-input.json not found\n")
            sys.exit(1)
    
    n_spots = config.get("n_spots", config.get("num_spots", 10))
    n_casts = config.get("n_casts", config.get("casts_per_day", 1000))
    n_days = config.get("n_days", config.get("num_days", 100))
    
    C = 2.0
    strategy = Strategy(c=C)
    strategy.num_spots = n_spots
    
    for day in range(n_days):
        strategy.reset()
        
        for cast in range(n_casts):
            spot_idx = strategy.select_spot()
            print(spot_idx, flush=True)
            
            # Read catch from stdin
            line = sys.stdin.readline()
            if not line:
                sys.stderr.write(f"EOF on day {day}, cast {cast}\n")
                sys.exit(1)
            
            try:
                catch = float(line.strip())
            except ValueError:
                sys.stderr.write(f"Invalid catch value: {line}\n")
                sys.exit(1)
            
            strategy.update(spot_idx, catch)


if __name__ == "__main__":
    main()