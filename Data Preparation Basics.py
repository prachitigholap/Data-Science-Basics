# -*- coding: utf-8 -*-


# Data Preparation Basics

Filtering and selecting
"""

import numpy as np
import pandas as pd

from pandas import Series, DataFrame

series_obj = Series(np.arange(8), index = ['row 1', 'row 2', 'row 3', 'row 4', 'row 5','row 6', 'row 7', 'row 8'])
series_obj

series_obj['row 7']

# extracting index via data
series_obj[[0, 7]]

np.random.seed(16)

DF_obj = DataFrame(np.random.rand(36).reshape((6,6)),
                   index = ['row 1', 'row 2', 'row 3', 'row 4', 'row 5','row 6'],
                   columns = ['column 1', 'column 2', 'column 3', 'column 4', 'column 5','column 6'])
DF_obj

DF_obj.loc[['row 2', 'row 5'], ['column 5', 'column 2']]

series_obj['row 3' : 'row 7']

DF_obj < .2

series_obj[series_obj > 6]

series_obj['row 1'] = 8
series_obj

"""Treating Missing values"""

missing = np.nan

series_obj = Series(['row 1', 'row 2', missing, 'row 3', 'row 4', missing, 'row 5', 'row 6', missing, 'row 7', 'row 8', missing, 'row 10'])
series_obj

series_obj.isnull()

np.random.seed
DF_obj = DataFrame(np.random.rand(36).reshape(6,6))
DF_obj

DF_obj.iloc[1:3, 0] = missing
DF_obj.iloc[0:5, 5] = missing
DF_obj

filledf = DF_obj.fillna(0)
filledf

filledf = DF_obj.fillna({0:0.1234, 5:0.2345})
filledf

fill_DF = DF_obj.fillna("0.456", inplace=True)
fill_DF

np.random.seed
DF_obj = DataFrame(np.random.rand(36).reshape(6,6))
DF_obj.iloc[1:3, 0] = missing
DF_obj.iloc[0:5, 5] = missing
DF_obj

# DF_obj.dropna(axis=0, inplace=True)
# DF_obj

"""# Removing Duplicates

"""

import pandas as pd
import numpy as np

from pandas import Series, DataFrame

df = DataFrame({ 'column 1': [1,1,2,2,3,3,3],
                 'column 2': ['a', 'a', 'b', 'b', 'c', 'c', 'c'],
                 'column 3': ['A', 'A', 'B', 'B', 'C', 'C', 'C']
                 })
df

df.duplicated()

df.drop_duplicates()

df = DataFrame({ 'column 1': [1,1,2,2,3,3,3],
                 'column 2': ['a', 'a', 'b', 'b', 'c', 'c', 'c'],
                 'column 3': ['A', 'A', 'B', 'B', 'C', 'D', 'C']
                 })
df

df.drop_duplicates(['column 3'])
#removing duplicates from one column

"""# Concatenation and Transformation

"""

import pandas as pd
import numpy as np

from pandas import Series, DataFrame

df1 = pd.DataFrame(np.arange(36).reshape(6,6))
df1

df2 = pd.DataFrame(np.arange(15).reshape(5,3))
df2

pd.concat([df1, df2], axis=1)

newdf = pd.concat([df1, df2])
newdf

newdf.drop_duplicates([0])

df1.drop([0,2])

df1.drop([0,2], axis=1)

Series_obj = Series(np.arange(6))
Series_obj.name = 'added_variable'
Series_obj

variable_added = DataFrame.join(df1, Series_obj)
variable_added

added_datatable = variable_added.append([Series_obj], ignore_index=True)
added_datatable

sorted = df1.sort_values( by=(5), ascending=[False])
sorted
