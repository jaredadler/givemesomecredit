# "Give Me Some Credit" Kaggle Analysis

## Overview

See the [Kaggle competition background](https://www.kaggle.com/c/GiveMeSomeCredit/overview), which includes the data sets.

As described on the Kaggle page, in the datasets (`cs-training.csv`, `cs-test.csv`):

- each row represents a customer
- the column `SeriousDlqin2yrs` is 1 if the customer experiences financial distress in the future and 0 if not
- the remaining columns are features describing the customer, their investments, demographics, etc

The goal is to use those features to predict the likelihood that a customer experiences financial distress in the future (let's call it "defaulting"). The main metric used to evaluate how well a Kaggle competitor achieves that goal is AUC, area under the curve (ROC curve). It looks like a straightforward classification modeling problem.

## Basic Data Cleaning

Read the data:

``` python

cs_test = pd.read_csv(f"{tld}data/cs-test.csv", index_col = 0)
cs_train = pd.read_csv(f"{tld}data/cs-training.csv", index_col = 0)

```

``` python

print(f"Number of observations in train: {len(cs_train)}")
print(f"Number of observations in test: {len(cs_test)}")

```

Number of observations in train: 150000
Number of observations in test: 101503

### Check ground truth

Let's look at the balance of ground truth labels (default or didn't default)

``` python

print(cs_train["SeriousDlqin2yrs"].value_counts(True).to_markdown())
print(cs_test["SeriousDlqin2yrs"].value_counts(True).to_markdown())

```

|    |   SeriousDlqin2yrs |
|---:|-------------------:|
|  0 |            0.93316 |
|  1 |            0.06684 |


So 6.7% of customers in the dataset did default. It's an imbalanced dataset which may need to be handled appropriately in modeling.

| SeriousDlqin2yrs   |
|--------------------|

#### No labels in the test dataset

The label column `SeriousDlqin2yrs` is null for test dataset. 

> Note: I assume it must be empty because that dataset is used for the competition? Originally I thought this could be used as the test dataset for evaluation and the best AUC score can be calculated from `cs-test.csv` at the end, but it seems for the competition style at the end we would just calculated the probability of financial distress for each observation and submit it. I can do that if needed but for now will just ignore that dataset.

### Missing data

First let's check for missing values:

``` python

print(pd.DataFrame(cs_train.isna().sum()).reset_index().to_markdown(showindex = False))
print(pd.DataFrame(cs_test.isna().sum()).reset_index().to_markdown(showindex = False))

```

Train dataset:

| index                                |     0 |
|:-------------------------------------|------:|
| SeriousDlqin2yrs                     |     0 |
| RevolvingUtilizationOfUnsecuredLines |     0 |
| age                                  |     0 |
| NumberOfTime30-59DaysPastDueNotWorse |     0 |
| DebtRatio                            |     0 |
| MonthlyIncome                        | 29731 |
| NumberOfOpenCreditLinesAndLoans      |     0 |
| NumberOfTimes90DaysLate              |     0 |
| NumberRealEstateLoansOrLines         |     0 |
| NumberOfTime60-89DaysPastDueNotWorse |     0 |
| NumberOfDependents                   |  3924 |


#### Some missing values in demographic features

We can see that we do not know the income for about 20% of customers and don't know the number of dependents for 2.6% of customers.

Let's make sure that the incidince of missing data is not itself predictive of default:

``` python

cs_train["missing_data"] = cs_train.isna().sum(axis = 1)
#missing data case does not appear to be predictive of fraud
print(pd.Series(cs_train.groupby("missing_data")["SeriousDlqin2yrs"].value_counts(True), name = "incidence")\
	.reset_index().to_markdown(showindex = False))
del(cs_train["missing_data"])

```

|   missing_data |   SeriousDlqin2yrs |   incidence |
|---------------:|-------------------:|------------:|
|              0 |                  0 |   0.930514  |
|              0 |                  1 |   0.0694859 |
|              1 |                  0 |   0.942264  |
|              1 |                  1 |   0.0577363 |
|              2 |                  0 |   0.954383  |
|              2 |                  1 |   0.0456167 |

Users who have missing data are slightly less likely to default (of course not directly causally), but the difference doesn't seem big enought to worry about. So for modeling purposes in the analysis of features we can decide the best way to impute values and during modeling step we can decide whether to ignore observations with missing data or not.

## Analysis of features

In this section I'll go through the features one-by-one to 

- understand what type of feature it is
- visualize the shape of the distribution of features
- see if there is any obvious relationship between the feature and possibility of defaulting
- decide how to clean up the feature (remove outliers, convert the feature to a different type)

### RevolvingUtilizationOfUnsecuredLines

Definition: Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits

Describing the feature:

``` python

print(cs_train.groupby("SeriousDlqin2yrs")["RevolvingUtilizationOfUnsecuredLines"].describe().T.to_markdown())

```

|       |              0 |            1 |
|:------|---------------:|-------------:|
| count | 139974         | 10026        |
| mean  |      6.16886   |     4.36728  |
| std   |    256.126     |   131.836    |
| min   |      0         |     0        |
| 25%   |      0.0269834 |     0.398219 |
| 50%   |      0.133288  |     0.838853 |
| 75%   |      0.487686  |     1        |
| max   |  50708         |  8328        |


- it's continuous
- there are upper-bound outliers
- there might be high-modal values like 0 and 1


``` python

cs_train["RevolvingUtilizationOfUnsecuredLines"].apply(lambda x: x == 1.0).sum()

```

17

Actually there aren't many observations with this feature equal to 1.

``` python

f, ax = plt.subplots(figsize=(10, 8))

cs_train.loc[cs_train["RevolvingUtilizationOfUnsecuredLines"] <= 1.0]["RevolvingUtilizationOfUnsecuredLines"]\
  .hist(by=cs_train['SeriousDlqin2yrs'], bins = 50, ax = ax)

plt.suptitle("RevolvingUtilizationOfUnsecuredLines <= 1 \n Left: no default Right: default")

plt.savefig(f"{tld}diagrams/RevolvingUtilizationOfUnsecuredLines_hist_01.png")

```

![](https://github.com/jaredadler/givemesomecredit/blob/master/diagrams/RevolvingUtilizationOfUnsecuredLines_hist_01.png)

But we can see that some users are very close to 1. I think a value of 1 represents a user having totally maxed out their credit line. So it makes sense that this is more common for those who will default, and we can expect this feature to be predictive as seen by the difference in distribution.

It's unclear to me what a value greater than 1 should mean, but since they are rare let's just lasso them in to be equal to the 95th quantile and check the same distribution:

``` python

utilization_max = cs_train["RevolvingUtilizationOfUnsecuredLines"].quantile(q = 0.95)
print(f"value of feature RevolvingUtilizationOfUnsecuredLines at 95th quantile: {utilization_max}")

f, ax = plt.subplots(figsize=(10, 8))

cs_train['RevolvingUtilizationOfUnsecuredLines']\
  .apply(lambda x: utilization_max if x > utilization_max else x)\
  .hist(by=cs_train['SeriousDlqin2yrs'], bins = 20, ax = ax)

plt.suptitle("RevolvingUtilizationOfUnsecuredLines with outliers lassoed to q=0.95 \n (Left: no default Right: default)")

plt.savefig(f"{tld}diagrams/RevolvingUtilizationOfUnsecuredLines_hist_02.png")

```

![](https://github.com/jaredadler/givemesomecredit/blob/master/diagrams/RevolvingUtilizationOfUnsecuredLines_hist_02.png)

The shape doesn't change significantly when lassoing the outliers. So we can probably do this (plus flag them in case we want to drop) and use this feature as-is in the model since even visually we can tell it will be important.

### Age

Definition: Age of borrower in years

Describing the feature:

``` python

print(cs_train.groupby("SeriousDlqin2yrs")["age"].describe().T.to_markdown())

```

|       |           0 |          1 |
|:------|------------:|-----------:|
| count | 139974      | 10026      |
| mean  |     52.7514 |    45.9266 |
| std   |     14.7911 |    12.9163 |
| min   |      0      |    21      |
| 25%   |     42      |    36      |
| 50%   |     52      |    45      |
| 75%   |     63      |    54      |
| max   |    109      |   101      |

Obviously some bogus values in the dataset (age = 0) to deal with. But otherwise looks easy to handle.

``` python

f, ax = plt.subplots(figsize=(10, 8))

cs_train['age']\
  .hist(by=cs_train['SeriousDlqin2yrs'], bins = 20, ax = ax, range=[0, 120])

plt.suptitle("Age of the customer/borrower \n (Left: no default Right: default)")

plt.savefig(f"{tld}diagrams/age_hist_01.png")


```

![](https://github.com/jaredadler/givemesomecredit/blob/master/diagrams/age_hist_01.png)

Visually we can assume normal distribution for this feature and we cannot observe any major differences, althought from the median we can see a 7 year difference. defaulters might skew a bit younger.

Age being lower than 18 is extremely rare so for the cases where it is so we can just drop the value and impute the median borrower age.

### NumberOfTime30-59DaysPastDueNotWorse

Definition: Number of times borrower has been 30-59 days past due but no worse in the last 2 years.

Describing the feature:

``` python

print(cs_train.groupby("SeriousDlqin2yrs")["NumberOfTime30-59DaysPastDueNotWorse"].describe().T.to_markdown())

```

|       |             0 |           1 |
|:------|--------------:|------------:|
| count | 139974        | 10026       |
| mean  |      0.280109 |     2.38849 |
| std   |      2.94607  |    11.7345  |
| min   |      0        |     0       |
| 25%   |      0        |     0       |
| 50%   |      0        |     0       |
| 75%   |      0        |     2       |
| max   |     98        |    98       |

We can see that being past due at all is a rare event (above 75th percentile for non-defaulters), and being past due more than once is even more rare although there are some extreme high values.

From this it makes sense to conver this to a binary feature (past due once or more) and plot it to see whether it's predictive.

``` python

cs_train["30-50 Days Past due more than once"] = cs_train["NumberOfTime30-59DaysPastDueNotWorse"].apply(lambda x: x > 0)

f, ax = plt.subplots(figsize=(10, 8))
cs_train.groupby("30-50 Days Past due more than once")["SeriousDlqin2yrs"].value_counts()\
        .unstack()\
        .plot(kind = "bar", ax = ax)

plt.title("30-59 Days Past Due More than Once")

plt.savefig(f"{tld}diagrams/NumberOfTime30-59DaysPastDueNotWorse_01.png")

del cs_train["30-50 Days Past due more than once"]

```

![](https://github.com/jaredadler/givemesomecredit/blob/master/diagrams/NumberOfTime30-59DaysPastDueNotWorse_01.png)

This is intuitive, people who later default are more likely to be past due on payments. Converting it to binary should be enough to prepare this feature for modeling.

### DebtRatio

Definition: Monthly debt payments, alimony,living costs divided by monthy gross income

Describing the feature:

``` python

print(cs_train.groupby("SeriousDlqin2yrs")["DebtRatio"].describe().T.to_markdown())

```

|       |             0 |            1 |
|:------|--------------:|-------------:|
| count | 139974        | 10026        |
| mean  |    357.151    |   295.121    |
| std   |   2083.28     |  1238.36     |
| min   |      0        |     0        |
| 25%   |      0.173707 |     0.193979 |
| 50%   |      0.362659 |     0.428227 |
| 75%   |      0.865608 |     0.892371 |
| max   | 329664        | 38793        |

So another feature that should be between 0 and 1 but with some major upper bound outliers. A super high value may just be bogus data but let's lasso it in to be equal to 1 and look at the distribution:

``` python

f, ax = plt.subplots(figsize=(10, 8))

cs_train["DebtRatio"].apply(lambda x: 1 if x > 1 else x).hist(by = cs_train["SeriousDlqin2yrs"], bins = 25, ax = ax)

plt.suptitle("Debt Ratio \n (Left: no default Right: default)")

plt.savefig(f"{tld}diagrams/DebtRatio_hist_01.png")


```

![](https://github.com/jaredadler/givemesomecredit/blob/master/diagrams/DebtRatio_hist_01.png)

- Having a high number of people with a debt ratio of 0 makes little sense to me although I may just not understand the feature.
- A large number of customers with debt ratio of 1 or higher is also hard to understand. We may want to create an additional binary feature from this feature that represents whether the debt ratio is 1 or higher.

### MonthlyIncome

Definition: Monthly income

Describing the feature:

``` python

print(cs_train.groupby(["SeriousDlqin2yrs"])["MonthlyIncome"].describe().T.to_markdown())

```

|       |                0 |         1 |
|:------|-----------------:|----------:|
| count | 111912           |   8357    |
| mean  |   6747.84        |   5630.83 |
| std   |  14813.5         |   6171.72 |
| min   |      0           |      0    |
| 25%   |   3461           |   2963    |
| 50%   |   5466           |   4500    |
| 75%   |   8333           |   6800    |
| max   |      3.00875e+06 | 250000    |


Some people are making more than 250,000 per month in income. Whether true or not, they should be treated as outliers and let's set an upper bound for this feature. 

``` python

f, ax = plt.subplots(figsize=(10, 8))

cs_train["MonthlyIncome"]\
.apply(lambda x: 25000 if x > 25000 else x)\
.hist(by = cs_train["SeriousDlqin2yrs"], bins = 50, ax = ax)

plt.suptitle("Monthly income \n (Left: no default Right: default)")

plt.savefig(f"{tld}diagrams/MonthlyIncome_hist_01.png")

```

![](https://github.com/jaredadler/givemesomecredit/blob/master/diagrams/MonthlyIncome_hist_01.png)


From comparing medians again we can see that defaulters may be slightly less wealthy. Otherwise there doesn't seem to be much difference between defaulters and non-defaulters in terms of income level. 

We can lasso outliers based on a fixed upper bound (99th quantile = 25000) or to handle outliers and make the distribution more normal could just take the log of monthly income and use that as a feature.

For observations where monthly income is missing, we fill in the missing values with the overall median (around 5000).

### NumberOfOpenCreditLinesAndLoans

Definition: Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards)

Describing the feature:

``` python

cs_train.groupby(["SeriousDlqin2yrs"])["NumberOfOpenCreditLinesAndLoans"].describe().T

```

|       |            0 |           1 |
|:------|-------------:|------------:|
| count | 139974       | 10026       |
| mean  |      8.49362 |     7.88231 |
| std   |      5.10523 |     5.6536  |
| min   |      0       |     0       |
| 25%   |      5       |     4       |
| 50%   |      8       |     7       |
| 75%   |     11       |    11       |
| max   |     58       |    57       |

So most people don't have more than 10 open credit lines, but some people have a lot more. Another feature where lassoing in upper bound outliers is probably enough to make the feature useful. Surprisingly there doesn't seem to be an obvious difference between defaulters and non-defaulters in terms of how many credit lines they have open.

Note: It does not make sense to me that someone can have a value of zero here but still be able to default on their credit.

Let's check what kind of upper bound we could set:

``` python

cs_train["NumberOfOpenCreditLinesAndLoans"].quantile(q = 0.99)

```

24.0

And plot based on that:

``` python

f, ax = plt.subplots(figsize=(10, 8))

cs_train["NumberOfOpenCreditLinesAndLoans"]\
.apply(lambda x: 24.0 if x >= 24.0 else x)\
.hist(by = cs_train["SeriousDlqin2yrs"], bins = 10, ax = ax)

plt.suptitle("Number of open credit lines and loans \n (Left: no default Right: default)")

plt.savefig(f"{tld}diagrams/NumberOfOpenCreditLinesAndLoans_hist_01.png")

```

![](https://github.com/jaredadler/givemesomecredit/blob/master/diagrams/NumberOfOpenCreditLinesAndLoans_hist_01.png)

### NumberOfTimes90DaysLate

Definition: Number of times borrower has been 90 days or more past due.

Describing the feature:

``` python

print(cs_train.groupby(["SeriousDlqin2yrs"])["NumberOfTimes90DaysLate"].describe().T.to_markdown())

```

|       |             0 |           1 |
|:------|--------------:|------------:|
| count | 139974        | 10026       |
| mean  |      0.135225 |     2.09136 |
| std   |      2.90909  |    11.7628  |
| min   |      0        |     0       |
| 25%   |      0        |     0       |
| 50%   |      0        |     0       |
| 75%   |      0        |     1       |
| max   |     98        |    98       |


Another feature like NumberOfTime30-59DaysPastDueNotWorse. Probably best to convert the feature to binary just like it.

``` python

cs_train["90 days late more than once"] = cs_train["NumberOfTimes90DaysLate"].apply(lambda x: x > 0)

f, ax = plt.subplots(figsize=(10, 8))
cs_train.groupby("90 days late more than once")["SeriousDlqin2yrs"].value_counts()\
        .unstack()\
        .plot(kind = "bar", ax = ax)

plt.title("90 days late more than once")

plt.savefig(f"{tld}diagrams/NumberOfTimes90DaysLate_01.png")

del cs_train["90 days late more than once"]


```

![](https://github.com/jaredadler/givemesomecredit/blob/master/diagrams/NumberOfTimes90DaysLate_01.png)

Similar shape to NumberOfTime30-59DaysPastDueNotWorse as well, it should be a very predictive feature and makes a lot of sense to use in a tree-based model.

### NumberRealEstateLoansOrLines

Definition: Number of mortgage and real estate loans including home equity lines of credit

Describing the feature:

``` python

print(cs_train.groupby(["SeriousDlqin2yrs"])["NumberRealEstateLoansOrLines"].describe().T.to_markdown())

```

|       |            0 |           1 |
|:------|-------------:|------------:|
| count | 139974       | 10026       |
| mean  |      1.02037 |     0.98853 |
| std   |      1.10551 |     1.42572 |
| min   |      0       |     0       |
| 25%   |      0       |     0       |
| 50%   |      1       |     1       |
| 75%   |      2       |     2       |
| max   |     54       |    29       |

This feature seems similar to NumberOfOpenCreditLinesAndLoans. I'm not experienced enough with credit risk to know whether they should be handled differently or whether their meanings overlap in some way.

But since they're similar in terms of descriptive statistics, let's do the same thing in setting an upper bound and lassoing outliers.

Also, since the median is 1 for both defaulters and non-defaulters, maybe we can add some additional binary features based on this feature:

- customer has no real estate: 1 if the value of this feature is 0
- customer has many real estate: 1 if the value of this feature is higher than 1


``` python

cs_train["NumberRealEstateLoansOrLines"].quantile(q = 0.99)

```

4.0


``` python

f, ax = plt.subplots(figsize=(10, 8))

cs_train["NumberRealEstateLoansOrLines"]\
.apply(lambda x: 10 if x > 10 else x)\
.hist(by = cs_train["SeriousDlqin2yrs"], bins = 20, ax = ax)


plt.suptitle("Number of real estate loans or lines \n (Left: no default Right: default)")

plt.savefig(f"{tld}diagrams/NumberRealEstateLoansOrLines_hist_01.png")

```

![](https://github.com/jaredadler/givemesomecredit/blob/master/diagrams/NumberRealEstateLoansOrLines_hist_01.png")

I don't see any obvious pattern here when comparing defaulters to non-defaulters except that non-defaulters are more likely to have more real estate loans. Maybe that is related to an unobserved feature like credit score. Anyway, this feature should be useful after lassoing outliers.

### NumberOfTime60-89DaysPastDueNotWorse

Definition: Number of times borrower has been 60-89 days past due but no worse in the last 2 years.

Describing the feature:

``` python

print(cs_train.groupby(["SeriousDlqin2yrs"])["NumberOfTime60-89DaysPastDueNotWorse"].describe().T.to_markdown())

```

|       |             0 |           1 |
|:------|--------------:|------------:|
| count | 139974        | 10026       |
| mean  |      0.126666 |     1.82805 |
| std   |      2.90093  |    11.7531  |
| min   |      0        |     0       |
| 25%   |      0        |     0       |
| 50%   |      0        |     0       |
| 75%   |      0        |     1       |
| max   |     98        |    98       |

Another feature similar to NumberOfTime30-59DaysPastDueNotWorse and NumberOfTimes90DaysLate. So we can conver this to a binary feature as well.

``` python

cs_train["60-89 days past due not worse"] = cs_train["NumberOfTime60-89DaysPastDueNotWorse"].apply(lambda x: x > 0)

f, ax = plt.subplots(figsize=(10, 8))
cs_train.groupby("60-89 days past due not worse")["SeriousDlqin2yrs"].value_counts()\
        .unstack()\
        .plot(kind = "bar", ax = ax)

plt.title("60-89 days past due not worse")

plt.savefig(f"{tld}diagrams/NumberOfTime60-89DaysPastDueNotWorse_01.png")

del cs_train["60-89 days past due not worse"]

```

![](https://github.com/jaredadler/givemesomecredit/blob/master/diagrams/NumberOfTime60-89DaysPastDueNotWorse_01.png)

So again a feature like NumberOfTime30-59DaysPastDueNotWorse and NumberOfTimes90DaysLate where we see that defaulters are more likely to be past due than non-defaulters and as a binary feature it should be predictive especially in a tree-based model.

### NumberOfDependents

Definition: Number of dependents in family excluding themselves (spouse, children etc.)

Describing the feature:

``` python

print(cs_train.groupby(["SeriousDlqin2yrs"])["NumberOfDependents"].describe().T.to_markdown())

```

|       |             0 |           1 |
|:------|--------------:|------------:|
| count | 136229        | 9847        |
| mean  |      0.743417 |    0.948208 |
| std   |      1.1059   |    1.21937  |
| min   |      0        |    0        |
| 25%   |      0        |    0        |
| 50%   |      0        |    0        |
| 75%   |      1        |    2        |
| max   |     20        |    8        |


Looks like most people have 0 or 1 kids, makes sense, although there are some outliers so let's set the upper bound on them to be safe too.

``` python

dependents_max = cs_train["NumberOfDependents"].quantile(q = 0.99)
print(f"Max dependents: {dependents_max}")

```

Max dependents: 4.0

``` python

f, ax = plt.subplots(figsize=(10, 8))

cs_train["NumberOfDependents"]\
  .apply(lambda x: dependents_max if x > dependents_max else x)\
  .hist(by = cs_train["SeriousDlqin2yrs"], bins = 5, ax = ax)

plt.suptitle("Number of dependents \n (Left: no default Right: default)")

plt.savefig(f"{tld}diagrams/NumberOfDependents_hist_01.png")


```

![](https://github.com/jaredadler/givemesomecredit/blob/master/diagrams/NumberOfDependents_hist_01.png)

So no obvious pattern coming from this feature. Besieds removing the upper bound outliers there isn't much needed to prepare this feature for modeling.

For observations where this value is missing, we can fill with the median or maybe mode (0 in this case).


## Data preprocessing

Based on what we saw from the analysis, here are the transformations we'll do to the features to prepare them for modeling and further examination:

``` python

def lassoOutlierAtX(pd_series, max_x):
    return pd_series.apply(lambda x: max_x if x > max_x else x)

def logTransform(pd_series):
    return pd_series.apply(lambda x: np.log(x + 1))

def convertBinaryAtX(pd_series, max_x):
    return pd_series.apply(lambda x: int(x >= max_x))

def convertBinaryAtEqualX(pd_series, equal_x):
    return pd_series.apply(lambda x: 1 if x == equal_x else 0)

# set upper bound for lassoing and flagging outliers, quantile value based on analysis
utilization_max = cs_train["RevolvingUtilizationOfUnsecuredLines"].quantile(q = 0.95)
pastdue_3059_max = cs_train["NumberOfTime30-59DaysPastDueNotWorse"].quantile(q = 0.99)
debtratio_max = cs_train["DebtRatio"].quantile(q = 0.9)
monthlyincome_max = cs_train["MonthlyIncome"].quantile(q = 0.99)
opencredit_max = cs_train["NumberOfOpenCreditLinesAndLoans"].quantile(q = 0.99)
late_90_max = cs_train["NumberOfTimes90DaysLate"].quantile(q = 0.99)
openrealestate_max = cs_train["NumberRealEstateLoansOrLines"].quantile(q = 0.99)
pastdue_6089_max = cs_train["NumberOfTime60-89DaysPastDueNotWorse"].quantile(q = 0.99)
dependents_max = cs_train["NumberOfDependents"].quantile(q = 0.99)

# use median to fill null values (and flag them separately)
monthlyincome_median = cs_train["MonthlyIncome"].quantile(q = 0.5)
dependents_median = cs_train["NumberOfDependents"].quantile(q = 0.5)
age_median = cs_train["age"].quantile(q = 0.5)

for target_df in [cs_train, cs_test]:
    target_df["RevolvingUtilizationOfUnsecuredLines_clean"] =\
      lassoOutlierAtX(target_df["RevolvingUtilizationOfUnsecuredLines"], utilization_max)
    
    target_df["age_clean"] = target_df["age"].apply(lambda x: age_median if x <= 10 else x)

    target_df["NumberOfTime30-59DaysPastDueNotWorse_clean"] =\
      convertBinaryAtX(target_df["NumberOfTime30-59DaysPastDueNotWorse"], 1.0)

    target_df["DebtRatio_iszero_clean"] = convertBinaryAtEqualX(target_df["DebtRatio"], 0.0)
    target_df["DebtRatio_isoneorhigher_clean"] = convertBinaryAtX(target_df["DebtRatio"], 1.0)
    target_df["DebtRatio_clean"] = lassoOutlierAtX(target_df["DebtRatio"], 1.0)


    # has missing data
    target_df["MonthlyIncome_clean"] = lassoOutlierAtX(target_df["MonthlyIncome"], monthlyincome_max)
    target_df["MonthlyIncome_log_clean"] = logTransform(target_df["MonthlyIncome"].fillna(monthlyincome_median))
    target_df["MonthlyIncome_missingflg"] = target_df["MonthlyIncome"].isna().apply(lambda x: int(x))
    target_df["MonthlyIncome_clean"] = target_df["MonthlyIncome_clean"].fillna(monthlyincome_median)

    target_df["NumberOfOpenCreditLinesAndLoans_clean"] =\
      lassoOutlierAtX(target_df["NumberOfOpenCreditLinesAndLoans"], opencredit_max)
    target_df["NumberOfOpenCreditLinesAndLoans_log_clean"] =\
      logTransform(target_df["NumberOfOpenCreditLinesAndLoans"])

    target_df["NumberOfTimes90DaysLate_clean"] =\
      convertBinaryAtX(target_df["NumberOfTimes90DaysLate"], 1.0)

    target_df["NumberRealEstateLoansOrLines_clean"] =\
      lassoOutlierAtX(target_df["NumberRealEstateLoansOrLines"], openrealestate_max)
    target_df["NumberRealEstateLoansOrLines_noloans_clean"] =\
      convertBinaryAtEqualX(target_df["NumberRealEstateLoansOrLines"], 0.0)
    target_df["NumberRealEstateLoansOrLines_manyloans_clean"] =\
      convertBinaryAtX(target_df["NumberRealEstateLoansOrLines"], 2.0)

    target_df["NumberOfTime60-89DaysPastDueNotWorse_clean"] =\
      convertBinaryAtX(target_df["NumberOfTime60-89DaysPastDueNotWorse"], 1.0)

    # has missing data
    target_df["NumberOfDependents_clean"] =\
      lassoOutlierAtX(target_df["NumberOfDependents"], dependents_max)
    target_df["NumberOfDependents_missingflg"] = target_df["NumberOfDependents"].isna().apply(lambda x: int(x))
    target_df["NumberOfDependents_clean"] = target_df["NumberOfDependents_clean"].fillna(dependents_median)

    def flagOutlierAtX(pd_series, at_x):
        return pd_series.apply(lambda x: int(x >= at_x))

    target_df["RevolvingUtilizationOfUnsecuredLines_outlierflg"] =\
      flagOutlierAtX(target_df["RevolvingUtilizationOfUnsecuredLines"], utilization_max)

    target_df["NumberOfTime30-59DaysPastDueNotWorse_outlierflg"] =\
      flagOutlierAtX(target_df["NumberOfTime30-59DaysPastDueNotWorse"], pastdue_3059_max)

    target_df["DebtRatio_outlierflg"] =\
      flagOutlierAtX(target_df["DebtRatio"], debtratio_max)

    target_df["MonthlyIncome_outlierflg"] =\
      flagOutlierAtX(target_df["MonthlyIncome"], monthlyincome_max)

    target_df["NumberOfOpenCreditLinesAndLoans_outlierflg"] =\
      flagOutlierAtX(target_df["NumberOfOpenCreditLinesAndLoans"], opencredit_max)

    target_df["NumberOfTimes90DaysLate_outlierflg"] =\
      flagOutlierAtX(target_df["NumberOfTimes90DaysLate"], late_90_max)

    target_df["NumberRealEstateLoansOrLines_outlierflg"] =\
      flagOutlierAtX(target_df["NumberRealEstateLoansOrLines"], openrealestate_max)

    target_df["NumberOfTime60-89DaysPastDueNotWorse_outlierflg"] =\
      flagOutlierAtX(target_df["NumberOfTime60-89DaysPastDueNotWorse"], pastdue_6089_max)

    target_df["NumberOfDependents_outlierflg"] =\
      flagOutlierAtX(target_df["NumberOfDependents"], dependents_max)

 ```

## Feature correlation

Since several features look better for a tree-based model, it is not terribly important, but let's take a quick look at correlation between features:

``` python

f, ax = plt.subplots(figsize=(10, 8))
corr = cs_train[[i for i in cs_train.columns if "clean" in i]].corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

plt.title("Correlation between features")

plt.savefig(f"{tld}diagrams/featurecorrelation_01.png")

```

![](https://github.com/jaredadler/givemesomecredit/blob/master/diagrams/featurecorrelation_01.png)


Some notable relationships:

- age is negatively correlated with number of credit lines, this is intuitive
- age is negatively correlated with number of dependents
- number of real estate lines is positively correlated with income
- utilization is positively correlated with delinquency, this makes sense but curiously there isn't a similar relationship with debt ratio
- number of real estate lines is positively correlated with income, debt ratio
- being late / past due features are all positively correlated with each other

No action should be taken on the features based on the observation of these relationships but it is good to examine it anyways.

## Simple first model

As is usually a good idea, let's build a simple decision tree and see how the tree learns what features to use at the top of the tree to make the most important splits.

To do this we need to handle the unbalanced dataset. I will use SMOTE to oversample the default class so that the class sizes are equal. Just to mimic the eventual modeling approach I'll hold out 20% of the dataset and train the decision tree classifier on the oversampled augmentation of the remaining 80%.

``` python

features = cs_train[[i for i in cs_train.columns if "clean" in i]].values
labels = cs_train["SeriousDlqin2yrs"].values

train_features, test_features, train_labels, test_labels =\
  train_test_split(features, labels, test_size=0.2, random_state=69)

# balance dataset using SMOTE
train_features_resampled, train_labels_resampled = SMOTE().fit_resample(train_features, train_labels)

dtclassifier = DecisionTreeClassifier(random_state=69, max_depth = 5)
decisiontree = dtclassifier.fit(train_features_resampled, train_labels_resampled)

```

And visualize:

``` python

viz = dtreeviz(decisiontree, train_features, train_labels,
                target_name="default",
                feature_names=cs_train[[i for i in cs_train.columns if "clean" in i]].columns,
                class_names=["no_default", "default"])

viz.save(f"{tld}/diagrams/dtreeviz.svg")

```

Click [here](https://raw.githubusercontent.com/jaredadler/givemesomecredit/master/diagrams/dtreeviz.svg) for the decision tree visualization.

We can see that the model learns that the most important split is based on the delinquency features (being late on payments), and then for those who are not as delinquent it starts to use the utilization and number of credit lines features for lower splits in the tree.

Notice that the demographic features are not used at all for a simple tree like this. Perhaps that means going back to look harder at demographic relationships would be worth doing to generate new features (e.g. monthly income divided by # of dependents) that might be more important to this kind of model.

## Random Forest Classifier

Now let's try something more complex like random forest and optimize for the Kaggle-specified metric, AUC

I will first use an implementation of random forest which automatically handles an imbalanced dataset by oversampling the minority class, and randomly run through some hyperparameters, holding on to the best model by AUC.

``` python

def sample_hyperparameters_brf():
    """
    Yield possible hyperparameter choices.
    """
    while True:
        yield {
        "n_estimators": choice([30, 50, 70, 100, 130, 150, 180, 200]),
        "criterion": choice(["gini", "entropy"]),
        "max_depth": choice([None, 3, 4, 5, 6, 7]),
        "sampling_strategy": choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, "not majority", "not minority"])
    }

def trainBalancedRFC(train_X, train_Y, test_X, test_Y, hp):
    balanceRFC = BalancedRandomForestClassifier(n_estimators = hp["n_estimators"],
                                                criterion = hp["criterion"],
                                                max_depth = hp["max_depth"],
                                                sampling_strategy = hp["sampling_strategy"])
    balanceRFCModel = balanceRFC.fit(train_X, train_Y)
    predictions = balanceRFCModel.predict_proba(test_X)[:,1]
    target_auc = roc_auc_score(test_labels, predictions)
    print(f"AUC: {target_auc}")
    return hp, balanceRFCModel, target_auc

tuning_results_brf = {"auc": 0.0}

for hyperparams in itertools.islice(sample_hyperparameters_brf(), 100):
    t_hp, t_model, t_auc = trainBalancedRFC(train_features, train_labels, test_features, test_labels, hyperparams)
    if t_auc > tuning_results_brf["auc"]:
        tuning_results_brf = {"hyperparameters": t_hp, "model": t_model, "auc": t_auc}

```

``` python

print(f"Best AUC from Random Forest Classifier: {tuning_results_brf['auc']}")

```

Best AUC from Random Forest Classifier: 0.8601514308672688


## Gradient Boosted Tree Classifier

As an alternative let's also train a classifier using XGBoost to see if we can beat the AUC from random forest. To do this I'll use GridsearchCV from SKlearn to combine grid search and cross validation ensuring the best possible model from the dataset over a wide range of hyperparameters. 

``` python

#split train-test-val
train_features_xgb, testval_features_xgb, train_labels_xgb, testval_labels_xgb =\
  train_test_split(features, labels, test_size=0.2, random_state=69)

# balance dataset using SMOTE
train_features_resampled_xgb, train_labels_resampled_xgb =\
  SMOTE().fit_resample(train_features_xgb, train_labels_xgb)

xgb_param_grid = {
        "nthread": [4],
        "num_rounds": [30],
        "n_estimators": [50, 100, 200],
        "colsample_bytree": [0.3, 0.5, 0.7]
    }

xgb_model_def = XGBClassifier(verbosity = 2, eval_metric = "auc")
xgb_model = GridSearchCV(estimator = xgb_model_def, param_grid = xgb_param_grid)
xgb_model.fit(train_features_xgb, train_labels_xgb)
predictions_xgb = xgb_model.predict_proba(testval_features_xgb)[:,1]
target_auc_xgb = roc_auc_score(testval_labels_xgb, predictions_xgb)

print(f"Best AUC from XGBoost: {target_auc_xgb}")

```

Best AUC from XGBoost: 0.8628090398803969

## Next Steps

- Check performance: if we could get the labels for `cs-test.csv`, we should check to see whether these top-performing models are robust against a new dataset. When trying to hack a metric like this it is possible that we are overfitting to something unique to the data in `cs-training.csv`, especially for example if `cs-test.csv` and `cs-training.csv` are from two different time periods.

- Engineer more features: if this is a real world scenario, there must certainly be more raw data available than this handful of features in order to predict default. For example, we do not have any included features based on rolling statistics such as recent spending, recent number of transactions. We also don't have any features flagging recent high-value purchases. So with more time I'd go back to see if it's possible to engineer these kind of features and check whether they have any predictive power

- Set up for deployment: imagining this as a real business problem, we don't know yet how we'd actually use this model. Maybe AUC is less important than minimizing false positives. Maybe the real challenge is how to calculate these features in batch, not in getting the best AUC score. And of course so far we just have notebook-style work, deploying the trained model in a container would require a little more work to set up.