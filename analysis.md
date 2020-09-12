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
print(pd.Series(cs_train.groupby("missing_data")["SeriousDlqin2yrs"].value_counts(True), name = "incidence").reset_index().to_markdown(showindex = False))
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

![](https://github.com/jaredadler/givemesomecredit/blob/master/diagrams/RevolvingUtilizationOfUnsecuredLines_hist_01.png)


