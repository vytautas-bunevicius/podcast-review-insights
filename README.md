# Podcast Reviews Analysis

## Table of Contents

- [Overview](#overview)
- [Dashboard](#dashboard)
- [Installation](#installation)
  - [Using uv (Recommended)](#using-uv-recommended)
  - [Using pip (Alternative)](#using-pip-alternative)
- [Data Analysis](#data-analysis)
  - [Research Objectives](#research-objectives)
  - [Hypotheses](#hypotheses)
  - [Exploratory Data Analysis Questions](#exploratory-data-analysis-questions)
- [Findings and Insights](#findings-and-insights)
- [Future Improvements](#future-improvements)
- [License](#license)

## Overview

Podcast Reviews Analysis is a comprehensive project that analyzes podcast review data using **Python** and **SQLite**. The dataset comprises two million reviews for one hundred thousand podcasts. We employ exploratory data analysis techniques to uncover insights into podcast popularity, review sentiment, category trends, temporal dynamics, and author behaviors. The project leverages libraries such as **Pandas**, **Matplotlib**, and **Plotly** for effective data manipulation and visualization.

## Dashboard

Explore the data interactively via our Looker Studio dashboard:

[Interactive Dashboard on Looker Studio](https://lookerstudio.google.com/reporting/2f3c7fbd-3ddb-4667-b545-834fbd7729e0)

For additional insights, visit our portfolio website to explore the Jupyter Notebook:

[Interactive Jupyter Notebook on Portfolio Website](https://bunevicius.com/project-pages/podcasts.html)

## Installation

### Using uv (Recommended)

1. **Install uv:**

   ```bash
   # On Unix/macOS
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # On Windows (PowerShell)
   irm https://astral.sh/uv/install.ps1 | iex
   ```

2. **Clone the Repository:**

   ```bash
   git clone https://github.com/vytautas-bunevicius/podcast-review-insights.git
   cd podcast-review-insights
   ```

3. **Create and Activate a Virtual Environment:**

   ```bash
   uv venv
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate     # On Windows
   ```

4. **Install Dependencies:**

   ```bash
   uv pip install -r requirements.txt
   ```

5. **Launch Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```

### Using pip (Alternative)

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/vytautas-bunevicius/podcast-review-insights.git
   cd podcast-review-insights
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Unix/macOS
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```

## Data Analysis

### Research Objectives

- **Podcast Popularity and Ratings:** Explore the relationship between the number of reviews and the average podcast rating.
- **Review Sentiment Analysis:** Investigate common sentiments in review content and assess the correlation between review length and rating.
- **Category Trends:** Examine the distribution of podcasts across various categories and analyze how ratings vary.
- **Trends Over Time:** Analyze changes in review counts and average ratings over different time periods.
- **Author Analysis:** Identify authors who review more frequently and explore any biases in their ratings.

### Hypotheses

- **Null Hypothesis (H0):** There is no statistically significant correlation between the length of a review and its rating.
- **Alternative Hypothesis (H1):** There exists a statistically significant correlation between review length and rating.

### Exploratory Data Analysis Questions

1. What is the distribution of ratings across all reviews? Do reviews tend to be more positive or negative?
2. Can prevalent sentiment patterns be identified in the review text?
3. Which podcast categories are most common, and how do ratings vary across these categories?
4. How have review counts and average ratings evolved over time?
5. Are there authors who consistently review more frequently, or who exhibit particular rating trends?

## Findings and Insights

1. **Podcast Popularity and Ratings:**  
   The analysis indicates a positive correlation between the number of reviews and the average rating, suggesting that podcasts with more reviews tend to have higher ratings.  
   An ANOVA test revealed that the length of a review is significantly associated with its rating (p-value < 0.05).

2. **Review Sentiment Analysis:**  
   - **Negative Reviews:** Common words include `podcast`, `like`, `listen`, `show`, `one`, `it's`, `listening`, `episode`, `get`, and `i'm`, which, despite not being inherently negative, often appear in critical contexts.  
   - **Positive Reviews:** Frequently used words include `podcast`, `love`, `great`, `listen`, `like`, `show`, `listening`, `one`, `really`, and `episode`, suggesting appreciation and enjoyment.

3. **Category Trends:**  
   The 'Society-Culture' category stands out with a high volume of reviews and generally positive ratings, while categories like 'TV-Film' and 'Sports' show more diverse audience reactions.

4. **Trends Over Time:**  
   The dataset reveals an overall increase in the number of reviews over time, with noticeable peaks and troughs. Average ratings have also fluctuated, reflecting changing listener sentiments.

5. **Author Analysis:**  
   Certain authors tend to review more frequently and consistently assign high ratings across different podcasts, indicating potential reviewer biases or specific audience engagement patterns.

## Future Improvements

- **Sentiment Analysis Enhancement:** Incorporate advanced natural language processing (NLP) techniques for deeper sentiment insights.
- **Predictive Modeling:** Develop models to forecast podcast popularity or predict user ratings based on various features.
- **Data Expansion:** Consider integrating additional datasets to enrich the analysis and validate findings further.

## License

This project is licensed under the Unlicense - see the [LICENSE](LICENSE) file for details.
