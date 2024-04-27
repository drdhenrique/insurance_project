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

# data import 
# sampling a subset to work on
# save the information .describe() from raw_data


df_raw = pd.read_csv("../data/raw/train.csv", index_col=[0])

X = df_raw.drop('target', axis = 1)
y = df_raw['target']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, stratify= y)

desc_raw_data = df_raw.describe()

desc_raw_data.to_csv('../data/interim/desc_raw_data')

# %%
df = pd.read_csv('../data/interim/df.csv', index_col=[0])
df.shape
# %%

1/(y_train.value_counts()[0]/y_train.value_counts()[1])
