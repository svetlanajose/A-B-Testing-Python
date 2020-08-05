#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# ## Table of Contents
# 
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# 
# <a id='probability'></a>
# #### Part I - Probability
# 

# In[3]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[4]:


df = pd.read_csv('ab_data.csv')
df.head()


# b. Use the cell below to find the number of rows in the dataset.

# In[5]:


df.shape[0]


# c. The number of unique users in the dataset.

# In[6]:


df['user_id'].nunique()


# d. The proportion of users converted.

# In[7]:


df.query('converted == 1')['user_id'].nunique()/df['user_id'].nunique()


# e. The number of times the `new_page` and `treatment` don't match.

# In[8]:


df.query('group == "treatment"').shape[0]


# In[9]:


df.query('landing_page == "new_page"').shape[0]


# In[10]:


df.query('group == "treatment"').query('landing_page == "new_page"').shape[0]


# In[11]:


(147276-145311)+(147239-145311)


# f. Do any of the rows have missing values?

# In[12]:


df.info() #Answer is No missing values


# `2.` For the rows where **treatment** does not match with **new_page** or **control** does not match with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to figure out how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[13]:


df.head()


# In[14]:


del_1 = df[df['landing_page']=="new_page"].query('group =="control"')


# In[15]:


arr1 = del_1.index.values


# In[16]:


len(arr1)


# In[17]:


del_2 = df[df['landing_page']=="old_page"].query('group =="treatment"')
arr2 = del_2.index.values
len(arr2)


# In[18]:


arr = np.concatenate([arr1,arr2])


# In[19]:


len(arr)==(1928 + 1965)


# In[20]:


1928 + 1965 #number of rows to be deleted


# In[21]:


arr_list = arr.tolist()  


# In[22]:


df2 = df.drop(arr_list)


# In[23]:


df2.info()


# In[24]:


294478-290585


# In[25]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[26]:


df2.head()


# In[27]:


df2['user_id'].nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[28]:


df2['user_id'].value_counts()


# c. What is the row information for the repeat **user_id**? 

# In[29]:


df2.query('user_id == 773192')


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[30]:


df2 = df2.drop(1899)


# In[31]:


df2.info()


# `4.` Use **df2** in the cells below to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[32]:


df2.head()


# In[33]:


df2['converted'].sum()/df2['converted'].count()


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[34]:


df2[df2['group']=="control"]['converted'].sum()/df2[df2['group']=="control"]['converted'].count()


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[35]:


df2[df2['group']=="treatment"]['converted'].sum()/df2[df2['group']=="treatment"]['converted'].count()


# d. What is the probability that an individual received the new page?

# In[36]:


df2[df2['landing_page']=="new_page"].count()/df2.count()


# e. Is sufficient evidence to conclude that the new treatment page leads to more conversions?

# The proability of a person converting regardless of the page they receive is 0.119.
# Given the individual was in the Control group, probability of converting is 0.120.
# Given the individual was in the Treatment group, probability of converting is 0.118.
# Probability that an individual received the new page is 0.5.
# 
# Since the probability of an individual receiving the new page is exactly .5, and inspite of that, the probability of an individual converting from the Treatment group (0.118) is lower than probability of an individual converting from the Control group(0.120), indicating that this new treatment page DOES NOT lead to more conversions. 

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# H0 =  P𝑛𝑒𝑤 - P𝑜𝑙𝑑  <= 0  #New page is equal or worse 
# H1 =  P𝑛𝑒𝑤 - P𝑜𝑙𝑑  > 0   #New page is better

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# <br><br>

# a. What is the **conversion rate** for $p_{new}$ under the null? 

# In[37]:


pnew = (df2.converted == 1).mean()
pnew


# b. What is the **conversion rate** for $p_{old}$ under the null? <br><br>

# In[38]:


pold = (df2.converted == 1).mean()
pold


# c. What is $n_{new}$, the number of individuals in the treatment group?

# In[39]:


nnew = df2.query('group == "treatment"')['user_id'].nunique()
nnew #sample size


# d. What is $n_{old}$, the number of individuals in the control group?

# In[40]:


nold = df2.query('group == "control"')['user_id'].nunique()  #sample size
nold


# e. Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[41]:


new_page_converted = np.random.choice([0, 1], size=nnew, p=[1-pnew, pnew])
new_page_converted


# f. Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[42]:


old_page_converted = np.random.choice([0, 1], size=nold, p=[1-pold, pold])
old_page_converted


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[43]:


new_page_converted.mean()-old_page_converted.mean()


# h. Create 10,000 $p_{new}$ - $p_{old}$ values using the same simulation process you used in parts (a) through (g) above. Store all 10,000 values in a NumPy array called **p_diffs**.

# In[44]:


p_diffs = []
for _ in range(10000):
    new_page_converted = np.random.choice([0, 1], size=nnew, p=[1-pnew, pnew])
    old_page_converted = np.random.choice([0, 1], size=nold, p=[1-pold, pold])
    p_diffs.append(new_page_converted.mean()-old_page_converted.mean())


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[45]:


np.array(p_diffs)


# In[46]:


plt.hist(p_diffs)


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[47]:


actual = (df2.query('group =="treatment"')['converted'].mean())-(df2.query('group == "control"')['converted'].mean())
actual


# In[48]:


plt.hist(p_diffs)
plt.axvline(actual,color='r')


# In[49]:


p_val = (p_diffs>actual).mean()
p_val


# k. What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# Here the p-value is 0.902 which is greater than the cut off value of 0.05. In this analysis, we fail to reject the null hypothesis which implies that the new page is only as equal or worse than the old landing page. 
# However, the difference is not signifant enough to conclude a differnce.  

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[50]:


import statsmodels.api as sm

convert_old = df2.query('landing_page == "old_page" and converted == 1').shape[0]
convert_new = df2.query('landing_page == "new_page" and converted == 1').shape[0]


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](https://docs.w3cub.com/statsmodels/generated/statsmodels.stats.proportion.proportions_ztest/) is a helpful link on using the built in.

# In[51]:


z_score, p_value = sm.stats.proportions_ztest([convert_new,convert_old], [nnew,nold], alternative='larger')
print(z_score,p_value)


# In[52]:


from scipy.stats import norm
norm.ppf(1-(0.05/2))


# In[53]:


df2.head()


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# Since the z-score of -1.31 does not exceed the critical value of 1.96 at a 95% confidence interval, and with the p-value so high at 0.905 which is greater than 0.05, we fail to reject the null hypothesis.

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you achieved in the A/B test in Part II above can also be achieved by performing regression.<br><br> 
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# Logistics Regression as it is used for predicting one out of two outcomes.

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives. However, you first need to create in df2 a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[54]:


df2['intercept'] = 1
df2[['drop', 'ab_page']] = pd.get_dummies(df2['group'])
df2.drop('drop', axis=1, inplace=True)
df2.head()


# c. Use **statsmodels** to instantiate your regression model on the two columns you created in part b., then fit the model using the two columns you created in part **b.** to predict whether or not an individual converts.

# In[59]:


from scipy import stats
stats.chisqprob = lambda chisq, df2: stats.chi2.sf(chisq, df2)


# In[60]:


logit = sm.Logit(df2['converted'],df2[['intercept','ab_page']])
results1 = logit.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[61]:


results1.summary()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br> 
# 
# What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in **Part II**?

# The p-value associated with ab_page is 0.190, which is greater than the cut off of 0.05 and so we fail to reject the null hypothesis. 
# The p-value here differes from Part II because, in Part II the null hypothesis was the conversion rate is equal irrespective of the landing page, and the alternate hypothesis was that the conversion rate is not equal for both landing pages. 
# In Part III, the logistic regression estimates how the conversion rate varies based on the landing page. 
# Additonally, regression in not a one-sided test, whereas the simulation and z-test are one-sided.

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# In most cases, it is a good idea to add more factors, other than the landing page, into the regression model to understand if they influence the conversions rate. For eg: demographic information like a specific age category, gender, country, interests may have a direct influence on the conversion rate. 
# However, the disadvantage to adding additonal terms is that the other factors may be correlated to each other and not just the dependant variable. For eg: age category can be related to interests leading to multicollinearity. Also, adding more factors increases chances of having more outliers which could impact the summary statistics used to conclude the hypothesis.

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in.
# 
# Does it appear that country had an impact on conversion?  

# In[62]:


countries_df = pd.read_csv('countries.csv')
countries_df.head()


# In[63]:


df_new = countries_df.set_index('user_id').join(df2.set_index('user_id'),how='inner')
df_new.head()


# In[64]:


df_new['country'].unique()


# In[65]:


df_new[['CA','UK','US']] = pd.get_dummies(df_new['country'])
df_new=df_new.drop('US',axis=1)
df_new.head()


# In[66]:


logit = sm.Logit(df_new['converted'],df_new[['intercept','CA','UK']])
results=logit.fit()
results.summary()


# With the p-values as 0.129 and 0.456 for CA and UK, which are still much higher than 0.05, we can say that the country does not have an impact on the conversion rate.

# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[67]:


df_new['UK_ab_page'] = df_new['UK']*df_new['ab_page']
df_new['CA_ab_page'] = df_new['CA']*df_new['ab_page']
logit3 = sm.Logit(df_new['converted'], df_new[['intercept', 'ab_page', 'UK', 'CA', 'UK_ab_page', 'CA_ab_page']])
results = logit3.fit()
results.summary()


# From the above p-values, we can see that even after analyzing results that involve interaction between landing page and country, we still have all p-values as 0.132,0.760,0.642,0.238,0.383 which are all higher than 0.05, indicating that none of these factors are statistically significant in predicting the conversion of users. In other words, the landing page and country of the user does not impact user conversion rate.

# <a id='conclusions'></a>
# ## Conclusion
# 
# > This project was aimed at helping a company decide whether or not they should launch a new page based on testing if the new page increases user conversion rate compared to the previous page. 
# 
# > In Part II, we failed to reject the null hypothesis, which was that the conversion rate of the new page is less than or equal to the old page. We also performed the z-test, which also failed to reject the null hypothesis.
# 
# > In Part III, we used logistic regression to calculate p-value, with which we also failed to reject the null hypothesis.
# >We also took into consideration the country of the user and a combination of landing page with country. In both cases, once again, the p-value did not indicate influence on the conversion rate. 
# 
# >Hence from a practical standpoint, we can conclude by saying that there is no test evidence to prove that the new landing page will increase conversion rate. Hence, investing in launching this new page would solely be a waste of time and money for the company and it is better not do so. 
# 
# 

# In[68]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])

