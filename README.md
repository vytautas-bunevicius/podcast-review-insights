# Podcast Reviews Analysis

## Interactive Charts

To interactively explore the data with charts, you can view the Jupyter notebook associated with this project on my portfolio website:

[Interactive Jupyter Notebook on portfolio website](https://bunevicius.com/project-pages/podcasts.html)

See more visualizations in the Looker dashboard here:

[Looker Visuals](https://lookerstudio.google.com/reporting/2f3c7fbd-3ddb-4667-b545-834fbd7729e0)

## Setup Guide

If you wish to run the code locally, follow these steps:

- Clone the Repository

      git clone https://github.com/vytautas-bunevicius/podcasts-reviews.git

- Navigate to Repository Directory

      cd podcasts-reviews

- Install the required Python packages using the following command:

      pip install -r requirements.txt

- Launch Jupyter Notebook

      jupyter notebook

## Overview

This project involves the analysis of podcast review data using **Python** and **SQLite**. The dataset consists of 2 million reviews for 100,000 podcasts. We will conduct exploratory data analysis (EDA) to uncover insights into podcast popularity, review sentiments, category trends, and more. The analysis will utilize Python libraries such as **Pandas**, **Matplotlib**, and **Plotly** for data manipulation and visualization.

## Research Objectives

1. **Podcast Popularity and Ratings:** Explore the relationship between the number of reviews and the average rating of podcasts.
2. **Review Sentiment Analysis:** Identify common sentiments in review content and determine if there is a correlation between review length and rating.
3. **Category Trends:** Analyze the distribution of podcasts across categories and investigate how ratings vary by category.
4. **Trends Over Time:** Examine how the number of reviews and average ratings have changed over time.
5. **Author Analysis:** Investigate if there are authors who review more frequently than others and if they tend to review certain types of podcasts or give specific ratings.

## Hypotheses

a. **Null Hypothesis (H0)**: There is no correlation between the length of a review and the rating it gives.

b. **Alternative Hypothesis (H1)**: There is a correlation between the length of a review and the rating it gives.

## Exploratory Data Analysis Questions

1. What is the distribution of ratings across all reviews? Are there more positive or negative reviews?
2. Can common sentiments be identified in the review content?
3. What are the most common categories for podcasts, and how do ratings vary across categories?
4. How have the number of reviews and average ratings changed over time?
5. Are there authors who review more frequently than others, and do they tend to review certain types of podcasts or give specific ratings?

## Findings and Insights

### 1. Podcast Popularity and Ratings
- The number of reviews tends to positively correlate with the average rating of a podcast, indicating higher popularity for podcasts with more reviews.
- An ANOVA test was conducted to investigate the association between the length of a review and the rating it receives. The p-value obtained was significantly less than 0.05, leading to the rejection of the null hypothesis. This suggests a statistically significant association between review length and rating.

### 2. Review Sentiment Analysis
- In Negative Reviews, the most frequent words include `podcast`, `like`, `listen`, `show`, `one`, `it’s`, `listening`, `episode`, `get`, and `i’m`. These words, while not inherently negative, often appear in contexts expressing dissatisfaction (e.g., “I don’t get why…”).
- In contrast, Positive Reviews often include words like `podcast`, `love`, `great`, `listen`, `like`, `show`, `listening`, `one`, `really`, and `episode`. Words like `love` and `great` are typically associated with positive sentiments, suggesting enjoyment or appreciation of the podcast.

### 3. Category Trends
- The 'Society-Culture' category appears to have the highest number of reviews and generally positive ratings.
- Categories such as 'TV-Film' and 'Sports' show a broader range of ratings, with more varied audience reception.

### 4. Trends Over Time
- The number of reviews has increased over time, with peaks and troughs observed in certain periods.
- Average ratings have also fluctuated over time, with variations between months and years.

### 5. Author Analysis
- Some authors review more frequently than others, with certain authors consistently giving high ratings across different podcasts and categories.

## Future Improvements

1. **Sentiment Analysis Enhancement:** Implement advanced natural language processing (NLP) techniques to extract deeper insights from review content.
2. **Predictive Modeling:** Develop models to predict podcast popularity or user ratings based on various features.
