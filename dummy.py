import subprocess
print("Running pytest again to ensure tests pass after changes")
subprocess.run(["uv", "run", "pytest", "tests/test_performance_opts.py"])
