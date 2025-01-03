# Rossmann Pharmaceuticals Sales Forecasting

## Overview

This repository contains the code and analysis of the Rossmann Pharmaceuticals Sales Forecasting project. The primary objective of Task 1 is to explore customer purchasing behavior across Rossmann stores, uncover key insights, and prepare data for predictive modeling. 

Through this exploratory data analysis (EDA), we aim to understand the relationships between various factors—such as promotions, holidays, competition, and assortment type—and customer purchasing behavior. The findings will guide further machine learning and deep learning tasks.

---

## Objectives

1. **Data Cleaning**: 
   - Detect and handle outliers.
   - Manage missing data using appropriate imputation techniques.

2. **Exploratory Data Analysis**:
   - Explore customer behavior using various visualizations.
   - Identify trends and patterns in the data.
   - Answer critical business questions, such as:
     - How do promotions affect sales?
     - What seasonal patterns exist in customer behavior?
     - How do competitor factors influence sales?

3. **Logging**:
   - Implement a logging system using Python's `logging` library to ensure reproducibility and traceability.

4. **Deliverables**:
   - A detailed Jupyter Notebook addressing all EDA objectives.
   - A slide deck (10-20 slides) summarizing key findings for the finance team.

---

## Data Description

The dataset includes the following fields:

- **Id**: Represents a (Store, Date) tuple within the test set.
- **Store**: Unique identifier for each store.
- **Sales**: Daily sales turnover (target variable).
- **Customers**: Number of customers per day.
- **Open**: Indicator for store operation (0 = closed, 1 = open).
- **StateHoliday**: Indicates state holidays (a = public holiday, b = Easter, c = Christmas, 0 = None).
- **SchoolHoliday**: Indicates if public schools were closed on a given day.
- **StoreType**: Categorizes stores into 4 types (a, b, c, d).
- **Assortment**: Assortment level (a = basic, b = extra, c = extended).
- **CompetitionDistance**: Distance to the nearest competitor store.
- **CompetitionOpenSince[Month/Year]**: Year and month the nearest competitor opened.
- **Promo**: Indicator for active promotions on a given day.
- **Promo2**: Ongoing promotion status (0 = no, 1 = yes).
- **Promo2Since[Year/Week]**: Start date of Promo2 participation.
- **PromoInterval**: Months when Promo2 starts anew.

---

## Key Questions

### Distribution and Correlation
- How are promotions distributed in the training and test datasets?
- What is the correlation between sales and the number of customers?

### Behavioral Trends
- How do sales behave before, during, and after holidays?
- Are there seasonal trends (e.g., Christmas, Easter) in purchasing patterns?

### Promotional Impact
- How do promotions affect sales and customer numbers?
- Could promotions be deployed more effectively? Which stores would benefit most?

### Competitive and Store Characteristics
- How does competitor proximity influence sales?
- Does the presence of new competitors impact existing stores?
- How does assortment type affect sales?

---

## Logging

For traceability and reproducibility, a logging system was implemented using Python's `logging` library. The following steps are logged:
- Data loading
- Data cleaning processes
- Summary of EDA findings

---

## Deliverables

1. **Jupyter Notebook**: A comprehensive notebook containing:
   - Data cleaning steps
   - Exploratory visualizations
   - Statistical summaries
   - Answers to key questions

2. **Slide Deck**:
   - A 10-20 slide presentation summarizing key findings.
   - Focus on actionable insights for the finance team.

---

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/rossmann-sales-forecasting.git
