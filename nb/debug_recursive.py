# debug_recursive.py

import pandas as pd
from IPython import embed
import os

# Load your previously pickled series and exogenous variables
series_path = "series.pkl"
exog_path = "future_exog_step_0.pkl"

# Make sure both files exist
if not os.path.exists(series_path) or not os.path.exists(exog_path):
    print("❌ One or both pickle files are missing.")
    print("Expected files:")
    print(f" - {series_path}: {'✅' if os.path.exists(series_path) else '❌ not found'}")
    print(f" - {exog_path}: {'✅' if os.path.exists(exog_path) else '❌ not found'}")
    exit()

# Load them
series = pd.read_pickle(series_path)
future_exog = pd.read_pickle(exog_path)

print("✅ Pickle files loaded successfully.")
print("🔍 series index tail:", series.index[-3:])
print("🔍 future_exog index head:", future_exog.index[:3])
print("🔍 future_exog shape:", future_exog.shape)

# Launch an interactive shell
embed()

# Optional cleanup after exiting shell (not necessary)
print("👋 Exiting debug session.")
thon