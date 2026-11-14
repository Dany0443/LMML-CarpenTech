import json
import numpy as np
from typing import List, Tuple, Optional
import sys

class ReefSimulator:
    def __init__(self, seed: int = 42, reef_type: str = "practice"):
        self.seed = seed
        self.reef_type = reef_type
        self.rng = np.random.default_rng(seed)
        self.num_spots = 10
        self.num_casts = 1000
        self.day = 0
        self.cast_count = 0
        self.spot_permutation = None
        self.spot_params = None
        self._initialize_reef()
        
    def _initialize_reef(self):
        """Initialize the reef with spot parameters. Practice and evaluation reefs differ."""
        if self.reef_type == "practice":
            # Practice reef: 10 spots with different reward distributions
            # Each spot has a base rate and occasional "walu run" (large catch)
            # Using a mixture distribution: mostly small catches, rare large ones
            self.spot_params = [
                {"base_mean": 2.0, "base_std": 1.0, "run_prob": 0.02, "run_mean": 50.0, "run_std": 10.0},
                {"base_mean": 3.0, "base_std": 1.5, "run_prob": 0.015, "run_mean": 80.0, "run_std": 15.0},
                {"base_mean": 1.5, "base_std": 0.8, "run_prob": 0.03, "run_mean": 40.0, "run_std": 8.0},
                {"base_mean": 4.0, "base_std": 2.0, "run_prob": 0.01, "run_mean": 100.0, "run_std": 20.0},
                {"base_mean": 2.5, "base_std": 1.2, "run_prob": 0.025, "run_mean": 60.0, "run_std": 12.0},
                {"base_mean": 3.5, "base_std": 1.8, "run_prob": 0.012, "run_mean": 90.0, "run_std": 18.0},
                {"base_mean": 1.0, "base_std": 0.5, "run_prob": 0.04, "run_mean": 30.0, "run_std": 6.0},
                {"base_mean": 5.0, "base_std": 2.5, "run_prob": 0.008, "run_mean": 120.0, "run_std": 25.0},
                {"base_mean": 2.8, "base_std": 1.4, "run_prob": 0.018, "run_mean": 70.0, "run_std": 14.0},
                {"base_mean": 3.2, "base_std": 1.6, "run_prob": 0.014, "run_mean": 85.0, "run_std": 17.0},
            ]
        else:
            # Evaluation reef: different parameters (hidden from strategy)
            # Using a different seed to generate different parameters
            eval_rng = np.random.default_rng(self.seed + 10000)
            self.spot_params = []
            for i in range(10):
                base_mean = eval_rng.uniform(1.0, 5.0)
                base_std = eval_rng.uniform(0.5, 2.5)
                run_prob = eval_rng.uniform(0.005, 0.04)
                run_mean = eval_rng.uniform(30.0, 150.0)
                run_std = eval_rng.uniform(5.0, 30.0)
                self.spot_params.append({
                    "base_mean": base_mean,
                    "base_std": base_std,
                    "run_prob": run_prob,
                    "run_mean": run_mean,
                    "run_std": run_std
                })
        
        # Shuffle spots for the first day
        self._shuffle_spots()
    
    def _shuffle_spots(self):
        """Reshuffle spot indices for a new day."""
        self.spot_permutation = self.rng.permutation(self.num_spots)
        self.day += 1
        self.cast_count = 0
    
    def reset_day(self):
        """Reset for a new day - reshuffle spots."""
        self._shuffle_spots()
    
    def cast(self, spot_idx: int) -> float:
        """
        Cast at a spot and return the catch in kg.
        spot_idx: 0-9 (the visible spot index after shuffling)
        """
        if spot_idx < 0 or spot_idx >= self.num_spots:
            raise ValueError(f"Invalid spot index: {spot_idx}. Must be 0-9.")
        
        if self.cast_count >= self.num_casts:
            raise ValueError(f"Daily cast limit ({self.num_casts}) reached.")
        
        # Map visible spot index to actual spot parameter
        actual_spot = self.spot_permutation[spot_idx]
        params = self.spot_params[actual_spot]
        
        # Determine if this is a "walu run" (rare large catch)
        is_run = self.rng.random() < params["run_prob"]
        
        if is_run:
            catch = self.rng.normal(params["run_mean"], params["run_std"])
        else:
            catch = self.rng.normal(params["base_mean"], params["base_std"])
        
        # Ensure non-negative catch
        catch = max(0.0, catch)
        
        self.cast_count += 1
        return catch
    
    def get_state(self) -> dict:
        """Get current simulator state for strategy."""
        return {
            "day": self.day,
            "cast_count": self.cast_count,
            "num_spots": self.num_spots,
            "num_casts": self.num_casts,
            "casts_remaining": self.num_casts - self.cast_count
        }
    
    def is_day_done(self) -> bool:
        return self.cast_count >= self.num_casts


def run_simulation(strategy_class, seed: int = 42, reef_type: str = "practice", num_days: int = 100) -> List[float]:
    """
    Run a full simulation with the given strategy.
    Returns list of daily catches.
    """
    simulator = ReefSimulator(seed=seed, reef_type=reef_type)
    daily_catches = []
    
    for day in range(num_days):
        strategy = strategy_class()
        simulator.reset_day()
        day_catch = 0.0
        
        while not simulator.is_day_done():
            state = simulator.get_state()
            spot_idx = strategy.select_spot(state)
            catch = simulator.cast(spot_idx)
            strategy.update(spot_idx, catch)
            day_catch += catch
        
        daily_catches.append(day_catch)
    
    return daily_catches


if __name__ == "__main__":
    # Quick test
    sim = ReefSimulator(seed=42, reef_type="practice")
    print(f"Reef type: {sim.reef_type}")
    print(f"Number of spots: {sim.num_spots}")
    print(f"Casts per day: {sim.num_casts}")
    print(f"Spot params: {sim.spot_params}")
    
    # Test one day
    sim.reset_day()
    total = 0
    for _ in range(1000):
        total += sim.cast(0)  # Always cast at spot 0
    print(f"Total catch (always spot 0): {total:.2f} kg")