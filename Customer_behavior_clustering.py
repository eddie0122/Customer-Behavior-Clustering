import pandas as pd
import numpy as np
from pandas import DataFrame, Series

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import scipy.cluster.hierarchy as sch

import os
import re
import gc

### Read File
folder = './data/'
file_path = folder + 'bhv_dataset_raw.pkl'
df = pd.read_pickle(file_path)

### DataFrame Information
df.info()

### DataFrame Example
df.head(15)


### How Many "Order Finish" Mobile Behavior Does Each Session IDs Contains ? ###

### Flag "Order Finish"(Code: 2542570)
df['ord_seq'] = df['bhr_cd'].where(df['bhr_cd'] == 2542570, np.nan)
df['ord_seq'] = df['ord_seq'].where(df['ord_seq'].isnull(), 1)
df.head(15)

### Aggregate The Count of "Order Finish" By Session ID Level
df_agg_of = df.groupby(['session_id']).agg({'ord_seq':'count'})
df_agg_of.sort_values(['ord_seq'], ascending=[False])


### Splite Session ID By Order Finish Code
df['ord_seq'] = df['bhr_seq'] + df['ord_seq']
df.head(20)

df['ord_seq_1'] = df.groupby(['session_id'])['ord_seq'].fillna(method='bfill')
df['ord_seq_1'].fillna(10000, inplace=True)
df.head(20)

df['ord_seq_1'] = df['ord_seq_1'].astype(int).astype(str)
df['ord_seq_1'] = df['ord_seq_1'].str.pad(5, side ='left', fillchar ='0')
df.head(20)

df['session_id_new'] = df['session_id'] + '_' + df['ord_seq_1']
df.head(20)

df.loc[0, 'session_id_new']

### Rename & Tidy-Up Columns
df.drop(['session_id', 'ord_seq', 'ord_seq_1'], axis=1, inplace=True)
df.rename(columns={'session_id_new':'session_id'}, inplace=True)
df = df[['customer_id', 'date_id', 'session_id', 'bhr_seq', 'bhr_cd']]
df.head(20)

# ### Export Temporary
# file_path = folder + 'bhr_dataset.pkl'
# df.to_pickle(file_path)


### Aggregate Behavior Data By Customer IDs
cols = ['customer_id', 'bhr_cd']
df_agg_cust = df.groupby(cols).agg({'bhr_seq':'count'})
df_agg_cust.head(20)

### Aggregate Behavior Data By Customer IDs w/o No Duplicated Behaviors in Sessions
df_no_duplicated = df.drop_duplicates(['session_id', 'bhr_cd'])
df_agg_cust_no = df_no_duplicated.groupby(cols).agg({'bhr_seq':'count'})
df_agg_cust_no.head(20)

df_agg_cust.reset_index(inplace=True)
df_agg_cust_no.reset_index(inplace=True)

df_agg_cust['duplicated'] = 1
df_agg_cust_no['duplicated'] = 0

df_agg_total = df_agg_cust.append(df_agg_cust_no).reset_index(drop=True)


### The Average Mobile Behavior Between Duplication & No Duplication in Sessions
df_agg_total['bhr_seq_log'] = np.log(df_agg_total['bhr_seq'] + 1)

fig, ax = plt.subplots(2, 1, figsize=(20, 14))
sns.barplot(data=df_agg_total, x='bhr_cd', y='bhr_seq', hue='duplicated', ax=ax[0])
ax[0].set_title('The Average Mobile Behavior Between Duplication & No Duplication in Sessions')

sns.barplot(data=df_agg_total, x='bhr_cd', y='bhr_seq_log', hue='duplicated', ax=ax[1])
ax[1].set_title('The Average Mobile Behavior Between Duplication & No Duplication in Sessions : Log Transform')

plt.show()

### Pivoting Data w/ Mobile Behavior : No Duplication
idx = 'customer_id'
col = 'bhr_cd'
val = 'bhr_seq'
df_bhr_pvt = pd.pivot_table(df_agg_cust_no, index=idx, columns=col,
                            values=val, aggfunc='sum', fill_value=0)
df_bhr_pvt


### Distribution of "Order Finish"(Code: 2542570)
fig, ax = plt.subplots(3, 1, figsize=(16, 21))
sns.distplot(df_bhr_pvt.loc[:, 2542570], ax=ax[0])
ax[0].set_title('Customers Order'' Finish(Code: 2542570) Distribution')

sns.distplot(df_bhr_pvt.loc[df_bhr_pvt.loc[:, 2542570] > 1, 2542570], ax=ax[1])
ax[1].set_title('Customers Order'' Finish(Code: 2542570) Distribution : More Than 1')

sns.distplot(df_bhr_pvt.loc[df_bhr_pvt.loc[:, 2542570] > 3, 2542570], ax=ax[2])
ax[2].set_title('Customers Order'' Finish(Code: 2542570) Distribution : More Than 3')

plt.show()

df_bhr_pvt.loc[:, 2542570].describe()


### Create Dataset For Clustering
# Due to Memory Size, I Sliced Customers Who Have Order Finish(Code: 2542570) More Than 3
dataset_raw = df_bhr_pvt[df_bhr_pvt.loc[:, 2542570] > 3].copy()

### Stratified Sampling By Order Finish Count
val = dataset_raw.loc[:, 2542570]
val = pd.qcut(val, q=[0, 0.25, 0.5, 0.75, 1], labels=[0, 1, 2, 3])
dataset_raw['stratified'] = val.astype(int)

_, dataset_sample = train_test_split(dataset_raw, test_size=0.15,
                                     stratify=dataset_raw['stratified'],
                                     random_state=2021)

dataset = dataset_sample.values
dataset

dataset_raw['stratified'].value_counts()
dataset_sample['stratified'].value_counts()

### Memory Retrieve
del [[dataset_raw, df, df_agg_cust, df_agg_cust_no, df_agg_of, df_agg_total, df_bhr_pvt, df_no_duplicated]]
gc.collect()
dataset_raw = DataFrame()
df = DataFrame()
df_agg_cust = DataFrame()
df_agg_cust_no = DataFrame()
df_agg_of = DataFrame()
df_agg_total = DataFrame()
df_bhr_pvt = DataFrame()
df_no_duplicated = DataFrame()


### Linkage Matrix & Dendrogram
data_linkage = sch.linkage(dataset, metric='cosine', method='complete')

plt.figure(figsize=(16, 8))
data_dend = sch.dendrogram(data_linkage, p=30, truncate_mode='lastp')
plt.title('Customer Mobile Behavior Similarity Dendrogram : Cosine Distance(Complete)')
plt.xlabel('Cluster Sample Number')
plt.ylabel('Cluster Distance')

ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [0.61, 0.61], '--', c='r')

plt.show()

### Make Clusters Into 7 Pieces
cluster_number = sch.cut_tree(data_linkage, 7)
dataset_sample['cluster'] = cluster_number
dataset_sample

### Make Clusters Into 7 Pieces : Pie Plot
recap = dataset_sample['cluster'].value_counts()
explodes = [ix * 0.02 for ix in range(7)]

plt.figure(figsize=(14, 8))
plt.title('The Percentage of Clusters')
plt.pie(recap, explode=explodes, labels=recap.index, autopct='%1.1f%%')
plt.show()

DataFrame(recap).T

# dataset_sample = dataset_sample[[1542568, 1542670, 1542671, 1552569, 1552570, 1552571, 1552573, 1552576, 1552579, 1552581, 1552582, 1552668, 1552768, 1552868, 1552869, 1552870, 1552871, 1552872, 1552873, 1552874, 1552875, 1552968, 1562568, 1562570, 1562670, 1562671, 1562672, 1562868, 1562871, 1572668, 1572669, 1572670, 1572671, 1572672, 1572673, 1572768, 1572769, 1582568, 1582668, 1582768, 1582769, 2542568, 2542569, 2542570, 3542568, 3542569, 3542570, 3542571, 3552568, 3552668, 3552768, 3562568, 3562569, 3562668, 3562669, 3562768, 3562769, 3572668, 3572669, 4542568, 4542668, 4552568, 4552768, 4552868, 4552869, 4552870, 4552871, 4552872, 4552874, 4552875, 4552969, 4552970, 4552971, 4552972, 4553068, 4553070, 4553071, 4553074, 4553075, 4553076, 4553077, 4553078, 4553079, 4553170, 4553268, 4553269, 4562568, 4562768, 4562769, 4572568, 4572569, 4572573, 4582568, 4582573, 4582575, 4592568, 4592668, 4602568, 4602569, 'stratified', 'cluster']]
### Centroids of Clusters
def cosineDist(_array1, _array2):
    _val = np.dot(_array1, _array2) / (np.linalg.norm(_array1, axis=1) * np.linalg.norm(_array2))
    return _val

cols = dataset_sample.columns[:-2].tolist()
centroids_mean = dataset_sample.groupby(['cluster'])[cols].mean()
centroids_mean

centroids = []
for cluster_num in range(7):
    array = dataset_sample.loc[dataset_sample['cluster'] == cluster_num, cols].copy()
    distances = cosineDist(array.values, centroids_mean.loc[cluster_num])
    array['similarity'] = distances

    array_max = array[array['similarity'] == array['similarity'].max()]
    centroids.append(array_max.iloc[0].values[:-1])

DataFrame(centroids)

### Calculate Distances From Cluster Centroids
array_data = dataset_sample[cols].values

for cluster_num, cluster_cntr in enumerate(centroids):
    val = cosineDist(array_data, cluster_cntr)

    col = 'distance_{}'.format(cluster_num)
    dataset_sample[col] = val

dataset_sample

### Heatmap : Similarity to Centroids
cols = ['distance_' + str(ix) for ix in range(7)]
df_heat = dataset_sample.set_index(['cluster'])[cols].copy()

val_max = df_heat.max(axis=1)
for ix in cols:
    df_heat[ix] = df_heat[ix] / val_max

ix = 0
fig, ax = plt.subplots(4, 2, figsize=(8 * 2, 6 * 4))
for ix1 in range(4):
    for ix2 in range(2):
        try:
            sns.heatmap(df_heat.loc[ix], vmax=1, vmin=0.5, cmap='Blues', yticklabels=False, ax=ax[ix1,ix2])
            ax[ix1,ix2].set_title('Cluster {} : Cosine Similary From Centroids'.format(ix))
            ax[ix1,ix2].set_xticks([ix_num + .5 for ix_num in range(7)])
            ax[ix1,ix2].set_xticklabels(list(range(7)))
            ax[ix1,ix2].set_xlabel('Cluster Numbers')
            ax[ix1,ix2].set_ylabel('')
        except:
            pass
        ix += 1
plt.show()
