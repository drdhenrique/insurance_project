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

# description of raw data

df_raw.describe(include='all')

# %%

# is there any NA?
df_raw.isna().sum()

# nop! 

# %%

X = df_raw.drop('target', axis = 1)
y = df_raw['target']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, stratify= y, random_state= 31415)

X_train, X_dev, y_train, y_dev = train_test_split(X_train,y_train, test_size= 0.25, stratify= y_train, random_state= 31415)