# Predicting Covid19 Hospitalization in Los Angeles


<img class="center" alt="banner" src="https://images.unsplash.com/photo-1623701197215-3fe8e52f618e?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8OHx8Y292aWR8ZW58MHx8MHx8&auto=format&fit=crop&w=500&q=60">
<figcaption class="" data-selectable-paragraph="">Unsplash- Yoav Aziz</figcaption>

## Project Overview

As the world started grappling with the ramifications of COVID-19, healthcare systems across the globe started dealing with the new overwhelming burden of caring for the people infected with the disease. For instance, in the US governments, all levels – Federal, State, and local, had to make decisions so they can help the hospitals as they struggled to shoulder the crisis. 

In this case study, I'm going to focus on the government of Los Angeles County (LA), California. This county is the most populated in the US, with approximately 10 million residents. I'm going to use historical data to predict the number of patients that will need hospitalization in the near future; specifically, I will create a model that can predict the number of hospitalizations in LA County <u>two weeks</u> from the present moment.

## Data Source

In this study, the target is the number of hospitalizations. To predict such attribute, I imagined what the independent attributes could be for predicting this specific dependent attribute.

The following list shows four sources of data that can be useful for predicting hospitalizations: 
* Historical data of LA County COVID-19 hospitalizations (https://data.chhs.ca.gov/dataset/covid-19-hospital-data) 
* Historical data of COVID-19 Cases and Deaths in LA County (https://data.chhs.ca.gov/dataset/covid-19-time-series-metrics-by-county-and-state) 
* Historical data of COVID-19 Vaccinations in LA County (https://data.chhs.ca.gov/dataset/covid-19-vaccine-progress-dashboard-data-by-zip-code) 
* The dates of US public holidays (these can be accessed via Google)

**Note:** the Datasets were downloaded for the analysis on 13-12-2022 

## File Description

~~~~~
Predicting Covid19 Hospitalization

    |   Predicting Covid19 Hospitalization.ipynb
    |   README.md
    |          
    +---data
    |       covid-hospitalizations-data-dictionary.xlsx
    |       covid19cases_test.csv
    |       covid19hospitalbycounty.csv
    |       covid19vaccinesbyzipcode_test.csv
    |       
    \---visuals
        |   Compare Test_Prediction.png
        |   DT DATAvsModel.png
        |   DT Test_Prediction.png
        |   HospDT
        |   HospDT.pdf
        |   LR DATAvsModel.png
        |   LR Test_Prediction.png
        |   MLP DATAvsModel.png
        |   MLP Test_Prediction.png
~~~~~


## Predictive analytics: approach

### Designing a dataset to support predictions

I deisgned a dataset based on the following characteristics:
1. The dataset must support the prediction needs. For instance, in this case, I want to use historical data to predict hospitalizations in two weeks. 
2. the dataset must be filled with all of the data we have collected. In this example, the data includes **covid19hospitalbycounty.csv**, **covid19cases_test.csv**, **covid19vaccinesbyzipcode_test.csv**, and the dates of US public holidays.

In this study, I loaded the three datasets mentioned earlier.  After examining the three datasets, I come up with a list of explanatory independent attributes for the prediction. In defining the attributes in the following list, I have used the **t** variable to represent time. For instance, **t0** shows **t=0**, and the attribute shows information about the same day as the row:

* `n_Hosp_t0`: The number of hospitalizations at t=0
* `s_Hosp_tn7_0`: The slope of the curve of hospitalizations for the period t=-7 to t=0 
* `av7_Case_tn6_0`: The seven-day average of the number of cases for the period t=-6 to t=0 
* `s_Case_tn14_0`: The slope of the curve of cases for the period t=-14 to t=0 
* `av7_Death_tn6_0`: The seven-day average of the number of deaths for the period t=-6 to t=0 
* `s_Death_tn14_0`: The slope of the curve of deaths for the period t=-14 to t=0 
* `p_FullVax_t0`: The percentage of fully vaccinated people at t=0 
* `s_FullVax_tn14_0`: The slope of the curve of the percentage of fully vaccinated people for the period t=-14 to t=0
* `n_days_MajHol`: The number of days from the previous major holiday 

The dependent attribute (i.e., target) is also coded similarly as `n_Hosp_t14`, which is the number of hospitalizations at t=14.

I created a placeholder `day_df`. Then I wrote functions to use the data from the three sources to fill up the attirbutes listed above. 

### Feature Selection

After filling up the placeholder, I run two feature selection techniques. **Linear Regression** and **Random Forest**. I chose **Linear Regression** because it captures the linear relationship (if any) between the attributes and the target variable. I chose **Random Forest** to select features that may have non-linear relationships and coudl be useful in a more complex model. Details are provided in the notebook.

### Predictions
In this section, I run 3 prediction models, namely **Linear regression**, **Decision Tree**, and **Multilayer Perceptron**. I used GridSearch to select the optimal parameters for the **Decision Tree**, and **Multilayer Perceptron** models.

The figures below shows the results of the predictions of each model on the training and test sets, respectively:

#### Linear Regression
![image](./visuals/LR%20DATAvsModel.png)

![image](./visuals/LR%20Test_Prediction.png)

#### Decision Tree

![image](./visuals/DT%20DATAvsModel.png)
![image](./visuals/DT%20Test_Prediction.png)

#### Multilayer Perceptron
![image](./visuals/MLP%20DATAvsModel.png)
![image](./visuals/MLP%20Test_Prediction.png)

#### Comparing the three models

The first figure in each section shows how well the models fitted the data. However, with the decision tree and MLP, the fit between the training data and the model shoudl not be trust, as these algorithms can easily overfit the training set.

From the below figure, we can see that MLP predicts better than Linear regression and Decision Tree.

![image](./visuals/Compare%20Test_Prediction.png)


## Tools and Packages
---
```sh
# Packages
- Matplotlib
- Numpy
- Pandas
- datetime
- Seaborn
- statsmodels
- Scikit-learn
- graphviz

# Technologies and ML models
- Feature Selection
    - Linear Regression
    - Random Forest
    - Decision Tree
- Predictions
    - Linear Regression
    - Decision Tree
    - Multilayer Perceptron
- Hyperparameter tuning
    - GridSearch
- Decision Tree plot visualization
    - Graphiz

```

