# Ensures the repository root is importable so `import symveig` works
# during pytest runs without requiring `pip install`.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
