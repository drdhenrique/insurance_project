# %%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

# %%

df_raw = pd.read_csv("../data/raw/test.csv", index_col=[0])


# %%
df = df_raw.sample(5000, random_state=31415)
# %%