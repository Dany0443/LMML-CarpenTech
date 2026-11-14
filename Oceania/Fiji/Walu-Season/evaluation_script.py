import sys
import json
import subprocess
import numpy as np
from typing import List, Dict, Any
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reef import ReefSimulator


def run_evaluation(strategy_script: str, dataset: str = "train", seed: int = 42, num_days: int = 100) -> Dict[str, Any]:
    """
    Run evaluation of a strategy script against the simulator.
    
    Args:
        strategy_script: Path to the strategy Python script (main.py)
        dataset: "train" for practice reef, "eval" for evaluation reef
        seed: Random seed for reproducibility
        num_days: Number of days to simulate
    
    Returns:
        Dict with evaluation results
    """
    reef_type = "practice" if dataset == "train" else "eval"
    
    # Start the strategy process
    proc = subprocess.Popen(
        [sys.executable, strategy_script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    simulator = ReefSimulator(seed=seed, reef_type=reef_type)
    daily_catches = []
    errors = []
    
    try:
        for day in range(num_days):
            # Send reset message
            reset_msg = json.dumps({"type": "reset"}) + "\n"
            proc.stdin.write(reset_msg)
            proc.stdin.flush()
            
            # Wait for ready
            ready_line = proc.stdout.readline()
            if not ready_line:
                errors.append(f"Day {day}: Strategy process ended unexpectedly")
                break
            
            try:
                ready_msg = json.loads(ready_line.strip())
                if ready_msg.get("type") != "ready":
                    errors.append(f"Day {day}: Expected 'ready', got {ready_msg}")
            except json.JSONDecodeError:
                errors.append(f"Day {day}: Invalid JSON in ready response: {ready_line}")
            
            simulator.reset_day()
            day_catch = 0.0
            cast_count = 0
            
            while not simulator.is_day_done():
                # Send select message
                state = simulator.get_state()
                select_msg = json.dumps({"type": "select", "state": state}) + "\n"
                proc.stdin.write(select_msg)
                proc.stdin.flush()
                
                # Wait for cast response
                cast_line = proc.stdout.readline()
                if not cast_line:
                    errors.append(f"Day {day}, Cast {cast_count}: Strategy process ended unexpectedly")
                    break
                
                try:
                    cast_msg = json.loads(cast_line.strip())
                    if cast_msg.get("type") != "cast":
                        errors.append(f"Day {day}, Cast {cast_count}: Expected 'cast', got {cast_msg}")
                        break
                    spot_idx = cast_msg.get("spot")
                    if spot_idx is None or not (0 <= spot_idx < 10):
                        errors.append(f"Day {day}, Cast {cast_count}: Invalid spot index: {spot_idx}")
                        break
                except json.JSONDecodeError:
                    errors.append(f"Day {day}, Cast {cast_count}: Invalid JSON in cast response: {cast_line}")
                    break
                
                # Simulate cast
                try:
                    catch = simulator.cast(spot_idx)
                except ValueError as e:
                    errors.append(f"Day {day}, Cast {cast_count}: Simulator error: {e}")
                    break
                
                # Send update message
                update_msg = json.dumps({"type": "update", "spot": spot_idx, "catch": catch}) + "\n"
                proc.stdin.write(update_msg)
                proc.stdin.flush()
                
                # Wait for ack
                ack_line = proc.stdout.readline()
                if not ack_line:
                    errors.append(f"Day {day}, Cast {cast_count}: Strategy process ended unexpectedly")
                    break
                
                try:
                    ack_msg = json.loads(ack_line.strip())
                    if ack_msg.get("type") != "ack":
                        errors.append(f"Day {day}, Cast {cast_count}: Expected 'ack', got {ack_msg}")
                except json.JSONDecodeError:
                    errors.append(f"Day {day}, Cast {cast_count}: Invalid JSON in ack response: {ack_line}")
                
                day_catch += catch
                cast_count += 1
            
            if errors:
                break
            
            # Send done message
            done_msg = json.dumps({"type": "done"}) + "\n"
            proc.stdin.write(done_msg)
            proc.stdin.flush()
            
            # Wait for done response
            done_line = proc.stdout.readline()
            if done_line:
                try:
                    done_response = json.loads(done_line.strip())
                except json.JSONDecodeError:
                    pass
            
            daily_catches.append(day_catch)
        
        # Close stdin and wait for process to finish
        proc.stdin.close()
        proc.wait(timeout=10)
        
    except subprocess.TimeoutExpired:
        proc.kill()
        errors.append("Strategy process timed out")
    except Exception as e:
        errors.append(f"Evaluation error: {e}")
        proc.kill()
    
    # Read any stderr output
    stderr_output = proc.stderr.read()
    if stderr_output:
        print(f"Strategy stderr:\n{stderr_output}", file=sys.stderr)
    
    # Calculate statistics
    if daily_catches:
        avg_catch = np.mean(daily_catches)
        std_catch = np.std(daily_catches)
        min_catch = np.min(daily_catches)
        max_catch = np.max(daily_catches)
    else:
        avg_catch = std_catch = min_catch = max_catch = 0.0
    
    return {
        "dataset": dataset,
        "seed": seed,
        "num_days": len(daily_catches),
        "daily_catches": daily_catches,
        "avg_catch": float(avg_catch),
        "std_catch": float(std_catch),
        "min_catch": float(min_catch),
        "max_catch": float(max_catch),
        "errors": errors,
        "passed": len(errors) == 0 and len(daily_catches) == num_days
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluation_script.py <strategy_script> [--dataset train|eval] [--seed N] [--days N]")
        sys.exit(1)
    
    strategy_script = sys.argv[1]
    dataset = "train"
    seed = 42
    num_days = 100
    
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--dataset" and i + 1 < len(sys.argv):
            dataset = sys.argv[i + 1]
        elif arg == "--seed" and i + 1 < len(sys.argv):
            seed = int(sys.argv[i + 1])
        elif arg == "--days" and i + 1 < len(sys.argv):
            num_days = int(sys.argv[i + 1])
    
    if not os.path.exists(strategy_script):
        print(f"Error: Strategy script not found: {strategy_script}")
        sys.exit(1)
    
    print(f"Running evaluation on {dataset} reef with seed {seed} for {num_days} days...")
    print(f"Strategy script: {strategy_script}")
    print()
    
    results = run_evaluation(strategy_script, dataset, seed, num_days)
    
    print(f"Results:")
    print(f"  Days completed: {results['num_days']}/{num_days}")
    print(f"  Average catch: {results['avg_catch']:.2f} kg")
    print(f"  Std deviation: {results['std_catch']:.2f} kg")
    print(f"  Min catch: {results['min_catch']:.2f} kg")
    print(f"  Max catch: {results['max_catch']:.2f} kg")
    print(f"  Passed: {results['passed']}")
    
    if results['errors']:
        print(f"  Errors: {len(results['errors'])}")
        for err in results['errors'][:5]:
            print(f"    - {err}")
    
    # Save results
    output_file = f"eval_results_{dataset}_seed{seed}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    if results['passed'] and results['avg_catch'] >= 5300:
        print("\n✓ TARGET ACHIEVED: Average catch >= 5300 kg/day")
    elif results['passed']:
        print(f"\n✗ TARGET MISSED: Average catch {results['avg_catch']:.2f} kg/day < 5300 kg/day")
    else:
        print("\n✗ EVALUATION FAILED: Errors occurred during simulation")
    
    sys.exit(0 if results['passed'] else 1)


if __name__ == "__main__":
    main()