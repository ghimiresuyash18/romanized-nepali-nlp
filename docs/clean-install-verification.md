# Clean-Install Verification

Verified with Python 3.13.3:

```text
python3 -m pytest -q
163 passed
ruff check .
All checks passed!
python3 scripts/demo.py
preprocessing and sentiment prediction completed
```

Base install now declares NumPy and scikit-learn runtime dependencies. Use `python3 -m pip install -e ".[dev,ml]"` for development and chart tooling.
