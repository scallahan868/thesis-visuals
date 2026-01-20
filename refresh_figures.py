import subprocess
import sys

scripts = [
    "training_missed_deliveries.py",
    "training_missed_deliveries_curr.py",
    "training_iteration_time_curr.py",
    "training_curriculum.py",
    "training_iteration_time.py",
    "training_iteration_time_with_agents.py",
]

for script in scripts:
    print(f"\n=== Running {script} ===")
    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        print(f"❌ {script} failed with exit code {result.returncode}")
        break
    else:
        print(f"✅ {script} completed successfully")

print("\nAll plotting scripts finished.")
