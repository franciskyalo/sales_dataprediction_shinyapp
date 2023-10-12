
## SALES PREDICTION USING MACHINE LEARNING

Sales are crucial for companies for several reasons. Firstly, they
directly impact revenue, which is the lifeblood of any business. Revenue
generated from sales covers operational costs, investments, and
ultimately determines profitability. Secondly, sales are a key indicator
of customer demand and market interest in a company’s products or
services. Understanding customer preferences and trends helps in
refining offerings and staying competitive.

Analyzing how other resources affect sales is equally vital. It provides
insights into the efficiency and effectiveness of operations. For
example, marketing efforts directly influence a customer’s decision to
make a purchase. Proper allocation of resources ensures that these areas
are optimized, ultimately driving more sales. Additionally, monitoring
resource allocation helps in identifying areas that may need improvement
or reallocation, leading to better cost management and higher
profitability. By assessing the interplay between resources and sales, a
company can make informed decisions to enhance its overall performance
and competitiveness in the market.

### MAIN OBJECTIVE

The main objective of this project is to come up with
`prescriptive model` that will
`explain how different resources affect sales` and also come up with a
`predictive model` that will be `deployed with a shiny app` for users to
interact with it.

### METRICS FOR SUCCESS

The project will be considered a success if the prediction model has a
Mean Absolute Percentage Error of less than `10%`.

## DATA UNDERSTANDING

``` r
# importing the required libraries 

library(tidyverse)
library(dlookr)
library(MLmetrics)
library(corrplot)
library(caret)
```

``` r
# reading in the data 

df = read_csv("sales data.csv")
```

``` r
df %>% glimpse()
```

    ## Rows: 200
    ## Columns: 5
    ## $ Id        <dbl> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 1~
    ## $ TV        <dbl> 230.1, 44.5, 17.2, 151.5, 180.8, 8.7, 57.5, 120.2, 8.6, 199.~
    ## $ Radio     <dbl> 37.8, 39.3, 45.9, 41.3, 10.8, 48.9, 32.8, 19.6, 2.1, 2.6, 5.~
    ## $ Newspaper <dbl> 69.2, 45.1, 69.3, 58.5, 58.4, 75.0, 23.5, 11.6, 1.0, 21.2, 2~
    ## $ Sales     <dbl> 22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2, 4.8, 10.6, 8.6~

``` r
# checking the first few rows of the dataset

df %>% head()
```

    ## # A tibble: 6 x 5
    ##      Id    TV Radio Newspaper Sales
    ##   <dbl> <dbl> <dbl>     <dbl> <dbl>
    ## 1     1 230.   37.8      69.2  22.1
    ## 2     2  44.5  39.3      45.1  10.4
    ## 3     3  17.2  45.9      69.3   9.3
    ## 4     4 152.   41.3      58.5  18.5
    ## 5     5 181.   10.8      58.4  12.9
    ## 6     6   8.7  48.9      75     7.2

``` r
# checking the last few rows of the dataset

df %>% tail()
```

    ## # A tibble: 6 x 5
    ##      Id    TV Radio Newspaper Sales
    ##   <dbl> <dbl> <dbl>     <dbl> <dbl>
    ## 1   195 150.   35.6       6    17.3
    ## 2   196  38.2   3.7      13.8   7.6
    ## 3   197  94.2   4.9       8.1   9.7
    ## 4   198 177     9.3       6.4  12.8
    ## 5   199 284.   42        66.2  25.5
    ## 6   200 232.    8.6       8.7  13.4

The data set has 200 instances and 5 columns. Columns include the Id
column of the instance, the amount of Tv spending, radio spending
amount, newspaper spending amount and sales(target variable)

``` r
# getting a summary of the dataset 

df %>% summary()
```

    ##        Id               TV             Radio          Newspaper     
    ##  Min.   :  1.00   Min.   :  0.70   Min.   : 0.000   Min.   :  0.30  
    ##  1st Qu.: 50.75   1st Qu.: 74.38   1st Qu.: 9.975   1st Qu.: 12.75  
    ##  Median :100.50   Median :149.75   Median :22.900   Median : 25.75  
    ##  Mean   :100.50   Mean   :147.04   Mean   :23.264   Mean   : 30.55  
    ##  3rd Qu.:150.25   3rd Qu.:218.82   3rd Qu.:36.525   3rd Qu.: 45.10  
    ##  Max.   :200.00   Max.   :296.40   Max.   :49.600   Max.   :114.00  
    ##      Sales      
    ##  Min.   : 1.60  
    ##  1st Qu.:10.38  
    ##  Median :12.90  
    ##  Mean   :14.02  
    ##  3rd Qu.:17.40  
    ##  Max.   :27.00

## DATA CLEANING

1.  **Checking and Dealing with Missing Values**: Handling missing data
    is crucial for accurate modeling. Techniques include imputation
    (filling missing values with estimates like mean, median, or mode),
    deletion (removing rows or columns with missing data), or using
    advanced methods like interpolation. Understanding the reason for
    missingness (e.g., random or systematic) can inform the choice of
    approach.

2.  **Checking for and Dealing with Duplicated Values**: Identifying and
    removing duplicate records is vital for maintaining data integrity.
    Duplicates can lead to biased results and overfitting. This process
    involves identifying identical rows and deciding whether to keep the
    first occurrence or remove duplicates entirely, based on the context
    of the dataset.

3.  **Checking if Dataset has Appropriate Data Types**: Ensuring that
    each column has the correct data type is crucial for efficient
    storage and accurate analysis. For instance, categorical variables
    should be represented as categories, dates as date objects, and
    numerical variables as appropriate numeric types (e.g., integers or
    floats). Incorrect data types can lead to errors or inefficient
    memory usage.

4.  **Checking and Dealing with Outliers**: Outliers can distort
    statistical analyses and model performance. Detecting outliers
    involves using techniques like visualizations (box plots, scatter
    plots) and statistical tests. Depending on the context, outliers can
    be removed, transformed, or analyzed separately. Robust modeling
    techniques, like Random Forests or Support Vector Machines, can
    handle outliers better than some other models.

1.`Checking and dealing with missing values`

``` r
# checking for duplicated values in the dataset

df %>% diagnose() %>% select(missing_count, missing_percent)
```

    ## # A tibble: 5 x 2
    ##   missing_count missing_percent
    ##           <int>           <dbl>
    ## 1             0               0
    ## 2             0               0
    ## 3             0               0
    ## 4             0               0
    ## 5             0               0

For the dataset, there are `zero` cases of missing values.

2.`Checking for duplicated values`

``` r
# checking for duplicated values 

df %>% duplicated() %>%  sum()
```

    ## [1] 0

There are no duplicates in the dataset

3.`Checking if the dataset has the correct column types`

``` r
# checking for column types 

df %>% diagnose() %>% select(variables, types)
```

    ## # A tibble: 5 x 2
    ##   variables types  
    ##   <chr>     <chr>  
    ## 1 Id        numeric
    ## 2 TV        numeric
    ## 3 Radio     numeric
    ## 4 Newspaper numeric
    ## 5 Sales     numeric

All the columns have the expected data types and there is no need to
convert them.

4.`Checking for outliers`

``` r
# dropping the id column as it does provide any information

df <- df %>% select(-Id)
```

``` r
# checking for outliers 

df %>% diagnose_outlier()
```

    ## # A tibble: 4 x 6
    ##   variables outliers_cnt outliers_ratio outliers_mean with_mean without_mean
    ##   <chr>            <int>          <dbl>         <dbl>     <dbl>        <dbl>
    ## 1 TV                   0              0          NaN      147.         147. 
    ## 2 Radio                0              0          NaN       23.3         23.3
    ## 3 Newspaper            2              1          107.      30.6         29.8
    ## 4 Sales                0              0          NaN       14.0         14.0

``` r
## plot the outliers 

df %>% select(Newspaper) %>% plot_outlier()
```

![](salesprediction_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

``` r
# writing a function to remove outliers 

remove_outliers <- function(data) {
  data %>%
    mutate(across(everything(), ~ ifelse(. > quantile(., 0.99), NA, .))) %>%
    drop_na()
}
```

``` r
# removing outliers 

new_df <- remove_outliers(df)
```

``` r
# checking if there are outliers 

new_df %>% diagnose_outlier()
```

    ## # A tibble: 4 x 6
    ##   variables outliers_cnt outliers_ratio outliers_mean with_mean without_mean
    ##   <chr>            <int>          <dbl>         <dbl>     <dbl>        <dbl>
    ## 1 TV                   0              0           NaN     144.         144. 
    ## 2 Radio                0              0           NaN      22.7         22.7
    ## 3 Newspaper            0              0           NaN      29.6         29.6
    ## 4 Sales                0              0           NaN      13.8         13.8

## EXPLORATORY DATA ANALYSIS

### UNIVARIATE ANALYSIS
