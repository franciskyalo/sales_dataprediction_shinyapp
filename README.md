![R Language](https://img.shields.io/badge/language-R-blue?logo=r&logoColor=white) ![RStudio](https://img.shields.io/badge/IDE-RStudio-blue?logo=rstudio&logoColor=white) ![R Shiny](https://img.shields.io/badge/R_Shiny-Web_Application_Framework-blue?logo=r&logoColor=white)


# COGNISALES: A SHINY APP FOR PREDICTING SALES USING MACHINE LEARNING📈📉

![bandicam 2023-10-17 05-26-45-100 (1)](https://github.com/franciskyalo/sales_dataprediction_shinyapp/assets/94622826/d855835e-0776-4b16-9c60-30d668abf3e7)

## Overview🔎 

Sales play a pivotal role in a company's success for a multitude of reasons. Firstly, they wield a direct influence on revenue, serving as the lifeblood of any thriving business. The revenue generated from sales not only covers operational expenses but also fuels investments, ultimately dictating profitability. Secondly, sales serve as a crucial gauge of customer demand and the level of market interest in a company’s offerings. This invaluable insight into customer preferences and market trends is instrumental in refining products or services, ensuring a competitive edge.

## Business understanding📌

Equally imperative is the analysis of how other resources impact sales. This examination provides profound insights into the efficiency and effectiveness of operational processes. For instance, marketing initiatives directly influence a customer's decision to engage in a transaction. Strategic allocation of resources guarantees that these domains are optimized, leading to an upsurge in sales. Furthermore, vigilant monitoring of resource allocation aids in identifying areas that may require enhancement or reassignment, culminating in superior cost management and heightened profitability. Through a comprehensive evaluation of the dynamic interplay between resources and sales, a company can make judicious decisions aimed at augmenting its overall performance and bolstering its competitive stance in the market.

### Main objective📌 

The primary aim of this project is to develop a prescriptive model elucidating the impact of various resources on sales. Additionally, we endeavor to construct a predictive model that will be integrated into a Shiny app, providing users with an interactive platform to engage with these insights.


## Data understanding📊

The data set has 200 instances and 5 columns. Columns include the Id column of the instance, the amount of Tv spending, radio spending amount, newspaper spending amount and sales(target variable)


## Modelling👨🏿‍💻 

Linear regression model will be used for coming up with the predictive model. The following process will be considered in coming up with the ultimate model:

- An initial model will be fit with all the predictor variables
- An second model will be fit excluding variables that did not have significant predictive influence on the target varible
- Adding some interaction terms and test if they significantly improve the accuracy of the model
- Investigating model diagnostics for the final model to ensure that it does violate any.

## Evaluation📃 

The model will be evaluated on new unseen data to investigate the accuracy of the model. Mean Absolute Error will be used to evaluate on how much the model is expected to perform when predicting on a new dataset 

## Recommendations💵 

1. Optimizing TV Advertising Expenditure:
Consider fine-tuning investments in TV advertising. With a nearly 0.01884-unit increase in sales per unit spent (holding other factors constant), focus on channels, time slots, or programs that have demonstrated efficacy in driving sales.

2. Maximizing Radio Advertising Impact:
Increase emphasis on radio advertising, as a one-unit increase in expenditure corresponds to an approximate 0.02552-unit rise in sales (with other factors constant). Given its proven positive impact, radio advertising should continue to be a key element of the marketing strategy.

3. Exploring Synergistic Opportunities:
Exploit the significant interaction effect observed between TV and Radio advertising. Consider synergistic campaigns that leverage both channels together. This combined approach appears to yield a notably amplified positive effect on sales compared to individual use.

4. Dynamic Campaign Monitoring and Adjustment:
Maintain a vigilant eye on the performance of advertising initiatives and be prepared to adapt strategies accordingly. This may involve reallocating resources based on channel effectiveness and taking into account the collaborative influence of TV and Radio advertising.

## Deployment of the model as a shiny app.

The model will be deployed a shiny app as **shown by the gif above**.This app is designed to facilitate sales predictions based on a pre-loaded model. Here's a brief overview of how the app works:

**File Upload and Prediction:**

Users are encouraged to upload a CSV file containing sales data.
After uploading, the app processes this data using a pre-trained predictive model.
The model generates sales predictions based on the provided information.

**Displaying Predictions:**

The predicted sales data is presented in a tabular format within the main panel.
The table allows users to interact with the data, such as filtering and searching for specific information.

**Downloadable Predictions:**

Users have the option to download the predictions as a CSV file.
A button labeled "Download the Predictions⤵️" facilitates this process.




