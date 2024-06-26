# %%

# imports

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
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
sns.set_theme()

# %%
df_raw = pd.read_csv("../data/raw/train.csv", index_col=[0])



# %%

# is there any NA?
df_raw.isna().sum().sum()

# nop! But in the data description on kaggle, missing are assigned to -1, so further exploration needed!

# %%
list(df_raw.columns) 

# %%
cond1 = df_raw['target'] == -1
sum(cond1)

# %%
# counting the -1 values

for i in list(df_raw.columns):
    cond1 = df_raw[i] == -1

    if sum(cond1) != 0:
        print(f'The total of -1 in the "{i}" column is', df_raw[cond1].shape[0]) 
        
# %%
# description of raw data

df_raw.describe(include='all')

# %%
# description of raw data

df_raw.info()
# %%

# there's four variable groups:
# ind - individual
# reg - region
# car - car :D
# calc - a calculated feature

# there's also some identifiers:
# bin - binary variable
# cat - categorical variable

df_car = df_raw[[x for x in df_raw.columns if 'car' in x]]
df_calc = df_raw[[x for x in df_raw.columns if 'calc' in x]]
df_reg = df_raw[[x for x in df_raw.columns if 'reg' in x]]
df_ind = df_raw[[x for x in df_raw.columns if 'ind' in x]]
df_bin = df_raw[[x for x in df_raw.columns if 'bin' in x]]
df_cat = df_raw[[x for x in df_raw.columns if 'cat' in x]]

# %%
sns.countplot(data=df_raw, x = 'target', hue= 'target')
# %%

# Univariated analysis - binary columns

df_bin.head()
# %%
plt.matshow(df_bin.corr());

# %%

sns.heatmap(df_bin.corr().round(2))
# %%
sns.countplot(data=df_car, x = 'ps_car_02_cat');
# %%
sns.countplot(data=df_car, x = 'ps_car_03_cat');
# %%
sns.countplot(data=df_car, x = 'ps_car_04_cat');
# %%
sns.countplot(data=df_car, x = 'ps_car_05_cat');
# %%
sns.countplot(data=df_car, x = 'ps_car_06_cat');
# %%
sns.countplot(data=df_car, x = 'ps_car_07_cat');
# %%
sns.countplot(data=df_car, x = 'ps_car_08_cat');
# %%
sns.countplot(data=df_car, x = 'ps_car_09_cat');
# %%
sns.countplot(data=df_car, x = 'ps_car_10_cat');
# %%
sns.histplot(data=df_car, x = 'ps_car_11_cat');


# %%

# visualizing features vs target

# %%