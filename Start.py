#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize']=(12,8)

df=pd.read_csv(r'C:\Users\Sanjay\Desktop\movies.csv') 


# In[2]:


df.head()


# In[3]:


# missing values
for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{}-{}%'.format(col,pct_missing))


# In[4]:


# data types
df.dtypes


# In[5]:


# Change data types of columns
df['budget'] = df['budget'].fillna(0) #replace na with 0
df['gross'] = df['gross'].fillna(0)

df['budget'] = df['budget'].astype('Int64')# after 'fiilna' 'int' instead of 'Int' can be used

df['gross'] = df['gross'].astype('Int64')


# In[6]:


df


# In[7]:


df


# In[8]:


df['yearcorrect']=df['released'].str.split('[(,\s)]{1}').str[3]
# Alternative code: df['released'].str.extract(pat = '([0-9]{4})').astype('str')


# In[9]:


df


# In[10]:


df.sort_values(by=['gross'],inplace=False, ascending=False)  #sorting by gross column


# In[11]:


pd.set_option('display.max_rows',None) # Showing more rows(optional)


# In[ ]:





# In[12]:


# Drop any duplicates
df['company'].drop_duplicates().sort_values(ascending=False)


# In[13]:


#Scatter plot with budget vs gross
plt.scatter(x=df['budget'], y=df['gross'])
plt.title('Budget vs Gross earnings')
plt.xlabel('Gross Earnings')
plt.ylabel('Budget for Film')


# In[14]:


# Regression using seaborn
sns.regplot(x='budget',y='gross', data=df, scatter_kws={"color":"red"}, line_kws={"color":"blue"})


# In[ ]:


# Looking at correlation

df.corr(method='spearman')


# In[ ]:


df.corr(method='kendall')


# In[ ]:


df.corr(method='pearson') #default


# In[ ]:


df.corr() # default pearson


# In[16]:


#Correlation heatmap
correlation_matrix=df.corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation matrix for numeric features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()


# In[17]:


# numerization of the object data
df_numerized=df
for col_name in df_numerized.columns:
    if (df_numerized[col_name].dtype=='object'):
        df_numerized[col_name]=df_numerized[col_name].astype('category')
        df_numerized[col_name]=df_numerized[col_name].cat.codes
        
df_numerized


# In[18]:


# Heatmap with all columns
correlation_matrix=df_numerized.corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation matrix for numeric features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()


# In[ ]:


correlation_mat=df_numerized.corr()
corr_pairs=correlation_mat.unstack()
corr_pairs


# In[ ]:


sorted_pairs=corr_pairs.sort_values()
sorted_pairs


# In[ ]:


# High Correlation
high_corr= sorted_pairs[(sorted_pairs)> 0.5]
high_corr


# In[20]:


# Using Linear regression
from sklearn.linear_model import LinearRegression
model=LinearRegression()


# In[23]:


# Remove infinite values
df_new = df_numerized[np.isfinite(df_numerized).all(1)]


# In[24]:


print(df_new)


# In[25]:


model.fit(df_new[['budget','votes','score']],df_new['gross'])


# In[26]:


print(model.intercept_, model.coef_)


# In[27]:


df_new.head()


# In[28]:


# To pedict revenue based on certain values of budget votes and score
model.predict([[300000,50000,7.4,]]) #The predicted revenue is $3244921.7


# In[ ]:




