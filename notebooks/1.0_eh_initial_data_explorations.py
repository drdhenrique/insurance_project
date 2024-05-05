# %%

# imports

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from IPython.core.display import HTML
from sklearn.model_selection import train_test_split

# %%

# auxiliary functions

%matplotlib inline

plt.style.use( 'bmh' )
plt.rcParams['figure.figsize'] = [25, 12]
plt.rcParams['font.size'] = 24
display( HTML( '<style>.container { width:100% !important; }</style>') )
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option( 'display.expand_frame_repr', False )
sns.set_theme(style="ticks", color_codes=True)

# %%
df_raw = pd.read_csv("../data/raw/train.csv", index_col=[0])



# %%

# is there any NA?
df_raw.isna().sum()

# nop! 
# %%

# description of raw data

df_raw.describe(include='all')

# %%
# description of raw data

df_raw.info()