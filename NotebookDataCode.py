#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import datetime as dt


# In[2]:


data = pd.read_csv ("Online Retail - Online Retail.csv")
data.head()


# In[3]:


data.describe()


# In[4]:


plt.boxplot(data["Quantity"])
plt.title("Quantity Quartiles")


# In[5]:


plt.boxplot(data["UnitPrice"])
plt.title("Price Breakdown")


# In[6]:


data= data[pd.notnull(data['InvoiceNo'])]
data= data[pd.notnull(data['StockCode'])]
data= data[pd.notnull(data['Description'])]
data= data[pd.notnull(data['Quantity'])]
data= data[pd.notnull(data['InvoiceDate'])]
data= data[pd.notnull(data['UnitPrice'])]
data= data[pd.notnull(data['CustomerID'])]
data= data[pd.notnull(data['Country'])]


# In[7]:


data['CustomerID'] = data['CustomerID'].astype(int)
data = data[(data['Quantity']>0)]


# In[8]:


data.info()


# In[9]:


data=data[['CustomerID','InvoiceDate','InvoiceNo','Quantity','UnitPrice']]


# In[10]:


data['TotalPrice'] = data['Quantity'] * data['UnitPrice']


# In[11]:


data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
data['InvoiceDate'].min(),data['InvoiceDate'].max()


# In[12]:


PRESENT = dt.datetime(2011,12,10)


# In[13]:


rfm= data.groupby('CustomerID').agg({'InvoiceDate': lambda date: (PRESENT - date.max()).days,
                                        'InvoiceNo': lambda num: len(num),
                                        'TotalPrice': lambda price: price.sum()})


# In[14]:


rfm.columns


# In[15]:


rfm.columns=['recency','frequency','monetary']


# In[16]:


rfm['recency'] = rfm['recency'].astype(int)


# In[17]:


rfm.head()


# In[18]:


rfm['r_quartile'] = pd.qcut(rfm['recency'], 4, ['1','2','3','4'])
rfm['f_quartile'] = pd.qcut(rfm['frequency'], 4, ['4','3','2','1'])
rfm['m_quartile'] = pd.qcut(rfm['monetary'], 4, ['4','3','2','1'])


# In[19]:


rfm.head()


# In[20]:


rfm['RFM_Segment_Concat'] = rfm.r_quartile.astype(str)+ rfm.f_quartile.astype(str) + rfm.m_quartile.astype(str)
rfm['RFM_Score'] = rfm.r_quartile.astype(int)+rfm.f_quartile.astype(int)+rfm.m_quartile.astype(int)


# In[21]:


rfm


# In[22]:


rfm[rfm['RFM_Score']==3].sort_values('monetary', ascending=False).head()


# In[23]:


rfm.sort_values('RFM_Score', ascending=True)


# In[24]:


from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# In[25]:


def check_values(df):
    col_desc = []
    data = {
        'features': [col for col in df.columns],
        'data_type': [df[col].dtype for col in df.columns],
        'nan_total': [df[col].isna().sum() for col in df.columns],
        'nan_pct': [round(df[col].isna().sum()/len(df)*100,2) for col in df.columns],
        'unique': [df[col].nunique() for col in df.columns],
        'values_ex': [df[col].drop_duplicates().sample(df[col].nunique()).values if df[col].nunique() <= 5 else df[col].drop_duplicates().sample(2).values for col in df.columns]
    }
    return pd.DataFrame(data)


# In[26]:


get_ipython().run_cell_magic('time', '', 'check_values(data)')


# In[27]:


data[data.InvoiceNo.str.startswith('C')]


# In[28]:


get_ipython().run_cell_magic('time', '', "df_clean = data.dropna(subset=['CustomerID'])\ncancelled = df_clean[df_clean.InvoiceNo.str.startswith('C')].index\ndf_clean = df_clean.drop(index=cancelled)\ndf_clean.loc[:,'Date'] = pd.to_datetime(df_clean['InvoiceDate'])\ndf_clean.loc[:,'TotalSum'] = df_clean['Quantity'] * df_clean['UnitPrice']\ndf_clean[['Quantity', 'UnitPrice', 'TotalSum']].head()")


# In[29]:


print('df_clean length:',len(df_clean))


# In[30]:


get_ipython().run_cell_magic('time', '', 'check_values(df_clean)')


# In[31]:


record_date = df_clean.Date.max() + timedelta(days=1)
df_rfm = df_clean.groupby(['CustomerID']).agg({
    'Date': lambda x: (record_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalSum': 'sum'})

df_rfm.rename(columns={
    'Date':'RecentTrans',
    'InvoiceNo':'Frequency',
    'TotalSum':'MonetaryValue'}, inplace=True)

df_rfm


# In[32]:


get_ipython().run_cell_magic('time', '', 'fig, ax = plt.subplots(1,3, figsize=(12,4))\nfor i, col in enumerate(df_rfm.columns):\n    sns.histplot(df_rfm[col], ax=ax[i])\nplt.show()')


# In[33]:


skew_columns = (df_rfm.skew().sort_values(ascending=False))

skew_columns = skew_columns.loc[skew_columns > 0.75]
skew_columns


# In[34]:


get_ipython().run_cell_magic('time', '', "df_transf = df_rfm.copy()\nfor col in skew_columns.index.tolist():\n    df_transf[col] = np.log1p(df_transf[col])\n\n\nfig, ax = plt.subplots(1,3, figsize=(12,4))\nfor i, col in enumerate(df_transf.columns):\n    sns.histplot(df_transf[col], ax=ax[i])\nplt.suptitle('After Transformation')\nplt.show()")


# In[35]:


(df_transf.skew().sort_values(ascending=False))


# In[36]:


scaler = StandardScaler()
df_train = df_transf.copy()
for col in df_transf.columns:
    df_train[col] = scaler.fit_transform(df_train[[col]])
df_train.describe().round(4)


# In[37]:


### BEGIN SOLUTION
km_list = []

for clust in range(2,12):
    km = KMeans(n_clusters=clust, random_state=26)
    km = km.fit(df_train)
    
    km_list.append(pd.Series({'clusters': clust, 
                              'inertia': km.inertia_,
                              'model': km}))

plot_data = (pd.concat(km_list, axis=1)
             .T
             [['clusters','inertia']]
             .set_index('clusters'))

ax = plot_data.plot(marker='o',ls='-')
ax.set_xticks(range(0,12,2))
ax.set_xlim(0,12)
ax.set_title('Clusters Number vs Inertia')
ax.set(xlabel='Cluster', ylabel='Inertia');
### END SOLUTION


# In[38]:


### BEGIN SOLUTION
km_list = []

for clust in range(2,12):
    km = KMeans(n_clusters=clust, random_state=26)
    km = km.fit(df_train)
    
    km_list.append(pd.Series({'clusters': clust, 
                              'distortion': sum(np.min(cdist(df_train, km.cluster_centers_,
                                        'euclidean'), axis=1)) / df_train.shape[0],
                              'model': km}))

plot_data = (pd.concat(km_list, axis=1)
             .T
             [['clusters','distortion']]
             .set_index('clusters'))

ax = plot_data.plot(marker='o',ls='-')
ax.set_xticks(range(0,12,2))
ax.set_xlim(0,12)
ax.set_title('Clusters Number vs Distortion')
ax.set(xlabel='Cluster', ylabel='Distortion');
### END SOLUTION


# In[39]:


def plot_silhouette_analysis(X):
    for n_clusters in range(2,12):
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8, 7)

        ax.set_xlim([-0.1, 1])
      
        ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])

      
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

       
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
     
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):

            ith_cluster_silhouette_values =                 sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            y_lower = y_upper + 10  # 10 for the 0 samples

        ax.set_title("Silhouette plot of the various clusters.")
        ax.set_xlabel("Silhouette of coefficient values")
        ax.set_ylabel("Cluster label")

        ax.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax.set_yticks([]) 
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                    fontsize=10, fontweight='bold')


# In[40]:


plot_silhouette_analysis(df_train)
plt.show()


# In[41]:


def kmeans(df, clusters_number):
    '''
    Implement k-means clustering on dataset
    
    INPUT:
        dataset : dataframe. Dataset for k-means to fit.
        clusters_number : int. Number of clusters to form.
        end : int. Ending range of kmeans to test.
    OUTPUT:
        Cluster results and t-SNE visualisation of clusters.
    '''
    
    kmeans = KMeans(n_clusters = clusters_number, random_state = 1)
    kmeans.fit(df)

    cluster_labels = kmeans.labels_

    df_new = df.assign(Cluster = cluster_labels)

    model = TSNE(random_state=1)
    transformed = model.fit_transform(df)

    plt.title('Flattened Graph of {} Clusters'.format(clusters_number))
    sns.scatterplot(x=transformed[:,0], y=transformed[:,1], hue=cluster_labels, style=cluster_labels, palette="Set1")
    
    return df_new, cluster_labels


# In[42]:


def plot_ag(df, clusters_number, df_agg):

    cluster_labels = df_agg['Cluster']

    model = TSNE(random_state=1)
    transformed = model.fit_transform(df)

    plt.title('Flattened Graph of {} Clusters'.format(clusters_number))
    sns.scatterplot(x=transformed[:,0], y=transformed[:,1], hue=cluster_labels, style=cluster)


# In[43]:


get_ipython().run_cell_magic('time', '', "df_km_2, labels_2 = kmeans(df_train, 2)\ndf_label_2 = df_rfm.assign(Cluster = labels_2)\nplt.title('Flattened Graph of 2 Clusters with KMeans')\nplt.show()")


# In[44]:


ag = AgglomerativeClustering(n_clusters=2, linkage='ward')
ag = ag.fit(df_train)
cluster_labels = ag.fit_predict(df_train)
df_agg_2 = df_train.assign(Cluster = cluster_labels)


# In[45]:


plot_ag(df_train, clusters_number=2, df_agg=df_agg_2)
plt.title('Flattened Graph of 2 Clusters with AgglomerativeClustering')
plt.show()


# In[46]:


get_ipython().run_cell_magic('time', '', "df_km_3, labels_3 = kmeans(df_train, 3)\ndf_label_3 = df_rfm.assign(Cluster = labels_3)\nplt.title('Flattened Graph of 3 Clusters with KMeans')\nplt.show()")


# In[47]:


ag = AgglomerativeClustering(n_clusters=3, linkage='ward')
ag = ag.fit(df_train)
cluster_labels = ag.fit_predict(df_train)
df_agg_3 = df_train.assign(Cluster = cluster_labels)


# In[48]:


plot_ag(df_train, clusters_number=3, df_agg=df_agg_3)
plt.title('Flattened Graph of 3 Clusters with AgglomerativeClustering')
plt.show()


# In[49]:


get_ipython().run_cell_magic('time', '', "df_km_4, labels_4 = kmeans(df_train, 4)\ndf_label_4 = df_rfm.assign(Cluster = labels_4)\nplt.title('Flattened Graph of 4 Clusters with KMeans')\nplt.show()")


# In[50]:


ag = AgglomerativeClustering(n_clusters=4, linkage='ward')
ag = ag.fit(df_train)
cluster_labels = ag.fit_predict(df_train)
df_agg_4 = df_train.assign(Cluster = cluster_labels)


# In[51]:


get_ipython().run_cell_magic('time', '', "plot_ag(df_train, clusters_number=4, df_agg=df_agg_4)\nplt.title('Flattened Graph of 4 Clusters with AgglomerativeClustering')\nplt.show()")


# In[52]:


def snake_plot(normalised_df_rfm, df_rfm_kmeans, df_rfm_original=df_rfm):
    '''
    Transform dataframe and plot snakeplot
    '''
    normalised_df_rfm = pd.DataFrame(normalised_df_rfm, 
                                       index=df_rfm_original.index, 
                                       columns=df_rfm_original.columns)
    normalised_df_rfm['Cluster'] = df_rfm_kmeans['Cluster']

    df_melt = pd.melt(normalised_df_rfm.reset_index(), 
                        id_vars=['CustomerID', 'Cluster'],
                        value_vars=['RecentTrans', 'Frequency', 'MonetaryValue'], 
                        var_name='Metric', 
                        value_name='Value')

    plt.xlabel('Metric')
    plt.ylabel('Value')
    
    return sns.pointplot(data=df_melt, x='Metric', y='Value', hue='Cluster')


# In[53]:


get_ipython().run_cell_magic('time', '', "fig, ax = plt.subplots(1,3, figsize=(16,4))\nplt.subplot(1,3,1)\nax[0]=snake_plot(df_train, df_km_2)\nax[0].set_title('cluster = 2')\nplt.subplot(1,3,2)\nax[1]=snake_plot(df_train, df_km_3)\nax[1].set_title('cluster = 3')\nplt.subplot(1,3,3)\nax[2]=snake_plot(df_train, df_km_4)\nax[2].set_title('cluster = 4')\nplt.suptitle('KMeans')\nplt.show()")


# In[54]:


def rfm_values(df):
    '''
    Calcualte average RFM values and size for each cluster

    '''
    df_new = df.groupby(['Cluster']).agg({
        'RecentTrans': 'mean',
        'Frequency': 'mean',
        'MonetaryValue': ['mean', 'count']
    }).round(0)
    
    return df_new


# In[55]:


rfm_values(df_label_3)


# In[61]:


import matplotlib.pyplot as plt
import numpy as np


# In[65]:


import matplotlib.pyplot as plt
import numpy as np

y = np.array([980, 1844, 1515])

plt.pie(y)
plt.show() 


# In[66]:


rfm_values(df_label_2)


# In[67]:


import matplotlib.pyplot as plt
import numpy as np

y = np.array([2435,1904])

plt.pie(y)
plt.show() 


# In[68]:


rfm_values(df_label_4)


# In[69]:


import matplotlib.pyplot as plt
import numpy as np

y = np.array([1222,867,872,1378])

plt.pie(y)
plt.show() 


# In[71]:


df_label_3.describe()
