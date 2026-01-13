# run_all helper

This repo contains three plotting/processing scripts used for thesis visuals. `run_all.py` is a small runner that executes them sequentially.

Files added:

- `run_all.py` - Python script that runs the three training scripts in order. Exits non-zero on the first failure unless `--continue-on-error` is passed.
- `run_all.ps1` - PowerShell wrapper to call the Python runner.

Usage

PowerShell (recommended on Windows):

```powershell
# run and stop on first error
.\run_all.ps1

# run and continue even if a script fails
.\run_all.ps1 -ContinueOnError
```

Direct Python:

```powershell
python run_all.py
python run_all.py --continue-on-error
```

Notes

- The runner uses the same Python interpreter that executes `run_all.py` (use `python` to run it with your chosen interpreter).
- If any of the training scripts are missing, the runner will report and exit with code 2 (unless `--continue-on-error` is used).
