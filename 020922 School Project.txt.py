#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
missing_values = ["s"]
df = pd.read_csv('https://raw.githubusercontent.com/CunyLaguardiaDataAnalytics/datasets/master/2014-15_To_2016-17_School-_Level_NYC_Regents_Report_For_All_Variables.csv',
                 na_values = missing_values)


# In[2]:


df.shape


# In[3]:


df.tail()


# ## Removing Columns 

# In[4]:


drop_cols = ['Number Scoring CR', 'Percent Scoring CR']
df.drop(drop_cols, inplace = True, axis=1)


# In[5]:


df.tail()


# In[6]:


def borough(row):
    if 'M' in row['School DBN']:
        return 'Manhattan'
    elif 'K' in row['School DBN']:
        return 'Brooklyn'
    elif 'Q' in row['School DBN']:
        return 'Queens'
    elif 'X' in row['School DBN']:
        return 'Bronx'
    elif 'R' in row['School DBN']:
        return 'Staten Island'
    else:
        return 'Other'
    
df['Borough'] = df.apply(lambda row: borough(row), axis=1)


# ## Confirm Borough column was added 

# In[7]:


df.tail()


# In[8]:


df.describe()


# In[9]:


df.groupby(['Borough'])['School Name'].nunique()


# ## Finding Percentage of Null Values

# In[10]:


df.isna().mean() * 100


# #### 35% of schools are missing test score data

# ### Finding Schools with missing Data by Borough

# In[11]:


no_data = df[df['Mean Score'].isnull()]
no_data.groupby(['Borough'])['School Name'].nunique()


# #### Only 2 schools have complete data

# ### Finding Missing Regent Exam Entries by School 

# In[12]:


regents_null = df[df['Regents Exam'].isnull()]
regents_null.groupby(['School Name'])['School DBN'].count()


# ### Determining if I should drop all entries for Inwood

# In[13]:


df[df['School Name'] == 'Inwood Academy for Leadership Charter School'].count()


# In[14]:


df_inwood = df[df['School Name'] == 'Inwood Academy for Leadership Charter School']
df_inwood.tail()


# #### Not dropping all, many rows have complete values

# ### Finding number of schools with missing Data

# In[15]:


df[df['Mean Score'].isnull()].count()


# ### Drop rows with null values by Mean score

# In[16]:


df_clean = df.dropna(subset=['Mean Score'])
df_clean.describe()


# ### Confirm row count is 137109 as expected

# In[17]:


df_clean.shape


# ## Review Tests by Number of Students and Year

# In[18]:


df_clean.groupby(['Regents Exam'])['Total Tested'].count().sort_values(ascending=False)


# In[19]:


df_clean.groupby(['Regents Exam'])['Total Tested'].sum().sort_values(ascending=False)


# ### Common Core Algebra was adminstered the most often to the most students

# In[20]:


df_clean.groupby(['Year'])['Total Tested'].sum()


# ### There was a decrease in the number of students tested each year

# # Data Visualization Mean Scores by Borough

# In[21]:


sns.boxplot(x='Borough', y='Mean Score', data=df_clean, hue='Year').set_title('Mean Score by Borough')


# In[22]:


df_clean_2015 = df_clean[df_clean['Year']== 2015]
df_clean_2016 = df_clean[df_clean['Year']== 2016]
df_clean_2017 = df_clean[df_clean['Year']== 2017]


# In[23]:


df_clean_2015['Mean Score'].mean()


# In[24]:


df_clean_2016['Mean Score'].mean()


# In[25]:


df_clean_2017['Mean Score'].mean()


# In[26]:


sns.boxplot(x='Year', y='Mean Score', data=df_clean, hue='Borough').set_title('Mean Score by Year')
plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)


# ## Most schools seem to have improved their average scores, the Bronx has the lowest average scores

# ## How Many Schools had more than 50% of their students score 80 or higher?

# In[27]:


all_percent_above_80 = df_clean[df_clean['Percent Scoring 80 or Above'] > 50.0]
all_percent_above_80_2015 = all_percent_above_80[all_percent_above_80['Year'] == 2015]
all_percent_above_80_2016 = all_percent_above_80[all_percent_above_80['Year'] == 2016]
all_percent_above_80_2017 = all_percent_above_80[all_percent_above_80['Year'] == 2017]


# In[51]:


all_percent_above_80.groupby(['Regents Exam'])['School Name'].count().sort_values(ascending=False)


# In[52]:


all_percent_above_80.groupby(['Borough'])['School Name'].count().sort_values(ascending=False)


# # Select School for Analysis

# In[29]:


df_school = df[df['School Name'] == 'New Explorations into Science, Technology and Math']
df_school.describe()


# ## Finding Missing Data for My School

# In[30]:


df_school.isna().mean() * 100


# In[31]:


df_school[df_school['Mean Score'].isnull()]


# ### 20% (86 Rows) Missing Data

# In[32]:


df_school_clean = df_school.dropna()
df_school_clean.describe()


# In[33]:


df_school_clean.head()


# ## Determining Grade Level

# In[34]:


df_school_clean.groupby(['School Level'])['School DBN'].count()


# ### All entries for K - 12

# ## Review Tests by Number of Students and Year

# In[35]:


df_school_clean.groupby(['Regents Exam'])['Total Tested'].count().sort_values(ascending=False)


# In[36]:


df_school_clean.groupby(['Regents Exam'])['Total Tested'].sum().sort_values(ascending=False)


# ### U.S. History and Government was administered most often to the most students. This is different than the overall data set.

# In[37]:


df_school_clean.groupby(['Year'])['Total Tested'].sum()


# ### There was a significant decrease in the number of students taking exams, this is in line with overall data set.

# ## Reviewing data where 50% or more of their students score 80 or higher

# In[38]:


percent_above_80 = df_school_clean[df_school_clean['Percent Scoring 80 or Above'] > 50.0]
percent_above_80_2015 = percent_above_80[percent_above_80['Year'] == 2015]
percent_above_80_2016 = percent_above_80[percent_above_80['Year'] == 2016]
percent_above_80_2017 = percent_above_80[percent_above_80['Year'] == 2017]


# In[39]:


percent_above_80.groupby(['Regents Exam'])['Regents Exam'].count().sort_values(ascending=False)


# In[55]:


percent_above_80.groupby(['Regents Exam'])['Total Tested'].sum().sort_values(ascending=False)


# ## Mean Score Review

# In[40]:


df_school_clean.groupby(['Regents Exam'])['Mean Score'].mean().sort_values(ascending=False)


# In[58]:


plt.barh(df_school_clean['Regents Exam'], df_school_clean['Mean Score'])


# In[41]:


df_school_clean_2015 = df_school_clean[df_school_clean['Year']== 2015]
df_school_clean_2016 = df_school_clean[df_school_clean['Year']== 2016]
df_school_clean_2017 = df_school_clean[df_school_clean['Year']== 2017]


# In[59]:


df_school_clean_2015.groupby(['Regents Exam'])['Mean Score'].mean().sort_values(ascending=False)


# In[60]:


df_school_clean_2016.groupby(['Regents Exam'])['Mean Score'].mean().sort_values(ascending=False)


# In[61]:


df_school_clean_2017.groupby(['Regents Exam'])['Mean Score'].mean().sort_values(ascending=False)


# In[65]:


df_school_clean_2015['Mean Score'].mean()


# In[66]:


df_school_clean_2016['Mean Score'].mean()


# In[67]:


df_school_clean_2017['Mean Score'].mean()


# ### This schools mean is higher than the data set as a whole. They had the highest scores on average in 2015, fell in 2016 and improved in 2017

# ## Reviewing which tests were dropped over the 3 years

# In[42]:


df_school_clean_2015.groupby(['Regents Exam'])['Regents Exam'].count().sort_values(ascending=False)


# In[43]:


df_school_clean_2016.groupby(['Regents Exam'])['Regents Exam'].count().sort_values(ascending=False)


# In[44]:


df_school_clean_2017.groupby(['Regents Exam'])['Regents Exam'].count().sort_values(ascending=False)


# ## Reviewing number of total students reduced over the 3 years

# In[45]:


df_school_clean_2015.groupby(['Regents Exam'])['Total Tested'].sum().sort_values(ascending=False)


# In[56]:


df_school_clean_2016.groupby(['Regents Exam'])['Total Tested'].sum().sort_values(ascending=False)


# In[57]:


df_school_clean_2017.groupby(['Regents Exam'])['Total Tested'].sum().sort_values(ascending=False)


# ### insights are embedded above

# In[ ]:




