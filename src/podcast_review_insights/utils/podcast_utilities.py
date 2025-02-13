import sqlite3
import string
from collections import Counter
from functools import lru_cache
from multiprocessing import Pool
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langdetect import detect
from matplotlib.colors import LinearSegmentedColormap
from plotly.subplots import make_subplots
from scipy import stats
from textblob import TextBlob
from wordcloud import WordCloud


def load_data_from_sql(table_name: str, connection) -> pd.DataFrame:
    """
    Load data from a SQL table into a pandas DataFrame.

    Args:
        table_name (str): The name of the SQL table.
        connection: The SQL connection object.

    Returns:
        pd.DataFrame: The loaded data.
    """
    return pd.read_sql(f"SELECT * FROM {table_name}", connection)


def convert_object_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object data types of a DataFrame to more memory-efficient types.

    Parameters:
    df (pandas.DataFrame): The DataFrame to convert.

    Returns:
    pandas.DataFrame: The DataFrame with converted data types.
    """
    for col in df.select_dtypes(include=["object"]).columns:
        if col in ["created_at", "run_at"]:
            df[col] = pd.to_datetime(df[col])
        elif df[col].nunique() < 0.5 * df[col].size:
            df[col] = df[col].astype("category")
    return df


def convert_float_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert float64 data types of a DataFrame to float32.

    Parameters:
    df (pandas.DataFrame): The DataFrame to convert.

    Returns:
    pandas.DataFrame: The DataFrame with converted data types.
    """
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    return df


def convert_int_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert int64 data types of a DataFrame to int32.

    Parameters:
    df (pandas.DataFrame): The DataFrame to convert.

    Returns:
    pandas.DataFrame: The DataFrame with converted data types.
    """
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")
    return df


def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert data types of a DataFrame to more memory-efficient types.

    Parameters:
    df (pandas.DataFrame): The DataFrame to convert.

    Returns:
    pandas.DataFrame: The DataFrame with converted data types.
    """
    df = convert_object_dtypes(df)
    df = convert_float_dtypes(df)
    df = convert_int_dtypes(df)
    return df


def print_summary_statistics(df: pd.DataFrame) -> None:
    """
    Print summary statistics of a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.
    """
    print("\nSummary statistics:")
    print(df.describe())


def print_duplicates(df: pd.DataFrame) -> None:
    """
    Print the number of duplicates in a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.
    """
    print(f"\nNumber of duplicates: {df.duplicated().sum()}")


def print_missing_values(df: pd.DataFrame) -> None:
    """
    Print the number of missing values in a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.
    """
    print(f"\nNumber of missing values:\n{df.isnull().sum()}")


def print_first_rows(df: pd.DataFrame) -> None:
    """
    Print the first few rows of a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.
    """
    print(f"\nFirst few rows:\n{df.head()}\n")


def analyze_dataframe(df: pd.DataFrame, name: str) -> None:
    """
    Analyze a DataFrame.

    This function prints out basic statistics, the number of duplicates,
    the number of missing values, the first few rows of the DataFrame,
    and a list of all available columns.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.
    name (str): The name of the DataFrame.
    """
    df = convert_dtypes(df)
    print(f"{name} table:")
    print(f"\nColumns: {df.columns.tolist()}")
    print_summary_statistics(df)
    print_duplicates(df)
    print_missing_values(df)
    print_first_rows(df)


def convert_arabic_numerals_to_english(text: str) -> str:
    """
    Convert a string containing Arabic numerals into a string containing English numerals.

    Parameters:
    text (str): The string to convert.

    Returns:
    str: The converted string.
    """
    map_arabic_english = {
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
    }
    return "".join(map_arabic_english.get(i, i) for i in text)


def to_lowercase(df: pd.DataFrame, column_names: List[str]) -> pd.DataFrame:
    """
    Convert the specified columns in a DataFrame to lowercase.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        column_names (List[str]): The names of the columns to convert to lowercase.

    Returns:
        pd.DataFrame: The modified DataFrame with the specified columns converted to lowercase.
    """
    for column_name in column_names:
        df.loc[:, column_name] = df[column_name].str.lower()
    return df


def remove_punctuation(df: pd.DataFrame, column_names: List[str]) -> pd.DataFrame:
    """
    Remove punctuation from the specified columns in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to remove punctuation from.
        column_names (List[str]): The names of the columns to remove punctuation from.

    Returns:
        pd.DataFrame: The DataFrame with punctuation removed from the specified columns.
    """
    translator = str.maketrans("", "", string.punctuation)
    for column_name in column_names:
        df[column_name] = df[column_name].str.translate(translator)
    return df


def remove_stopwords(df: pd.DataFrame, column_names: List[str]) -> pd.DataFrame:
    """
    Removes stopwords from the specified columns in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to remove stopwords from.
        column_names (List[str]): The names of the columns to remove stopwords from.

    Returns:
        pd.DataFrame: The DataFrame with stopwords removed from the specified columns.
    """
    import nltk
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    stop = set(stopwords.words("english"))
    import re
    pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, stop)) + r')\b', flags=re.IGNORECASE)
    
    for column_name in column_names:
        df[column_name] = (
            df[column_name]
            .str.replace(pattern, '', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
        )
    return df


@lru_cache(maxsize=1024)
def is_text_english_cached(text: str) -> bool:
    """
    Cached helper function to check if a given text is English.
    """
    try:
        text = text.replace("\n", " ").encode("utf-8", "ignore").decode("utf-8")
        return detect(text) == "en"
    except Exception:
        return False


def is_english(df: pd.DataFrame, column_names: List[str]) -> pd.Series:
    """
    Check if the texts in the specified columns of a DataFrame are in English using langdetect.
    Optimized to work on individual columns rather than row-wise iteration.

    Args:
        df (pandas.DataFrame): The DataFrame to check.
        column_names (List[str]): The names of the columns to check for English content.

    Returns:
        pd.Series: A boolean Series indicating whether each row is English.
    """
    mask = pd.Series(True, index=df.index)
    for col in column_names:
        mask &= df[col].apply(lambda x: True if pd.isnull(x) else is_text_english_cached(x))
    return mask


def remove_non_english(df: pd.DataFrame, column_names: List[str]) -> pd.DataFrame:
    """
    Remove non-English rows from a DataFrame based on the content of specific columns.

    Args:
        df (pandas.DataFrame): The DataFrame to modify.
        column_names (List[str]): The names of the columns to check for English content.

    Returns:
        pd.DataFrame: The DataFrame with non-English rows removed.
    """
    english = is_english(df, column_names)
    return df[english]


def add_length_column(
    df: pd.DataFrame, column_name: str, new_column_name: str
) -> pd.DataFrame:
    """
    Adds a new column to the DataFrame with the length of the values in the specified column.
    Optimized to use vectorized string length calculation.

    Parameters:
        df (pandas.DataFrame): The DataFrame to modify.
        column_name (str): The name of the column for which to calculate the length.
        new_column_name (str): The name of the new column to add.

    Returns:
        pandas.DataFrame: The modified DataFrame with the new column added.
    """
    df[new_column_name] = df[column_name].str.len()  # Vectorized instead of apply(len)
    return df


def save_dataframe_to_sql(
    df: pd.DataFrame, conn: sqlite3.Connection, table_name: str
) -> None:
    """
    Saves a pandas DataFrame to a SQL table.

    Args:
        df (pandas.DataFrame): The DataFrame to be saved.
        conn (sqlite3.Connection): The SQLite database connection.
        table_name (str): The name of the table to save the DataFrame to.

    Returns:
        None
    """
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.commit()


def create_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    x_name: str,
    y_name: str,
    title: str,
    color: Union[str, Dict[str, str]],
    save_path: str = None,
) -> None:
    """
    Creates a bar chart using the specified dataframe and column names.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data.
    x (str): The name of the column to be used as the x-axis.
    y (str): The name of the column to be used as the y-axis.
    x_name (str): The name to be displayed on the x-axis.
    y_name (str): The name to be displayed on the y-axis.
    title (str): The title of the chart.
    color (str or dict): Either a single color for all bars or a dictionary mapping the x column values to colors.
    save_path (str): The path to save the chart as a PNG.

    Returns:
    None
    """
    if isinstance(color, dict):
        fig = px.bar(df, x=x, y=y, color=x, title=title, color_discrete_map=color)
    else:
        fig = px.bar(df, x=x, y=y, title=title, color_discrete_sequence=[color])

    fig.update_layout(xaxis_title=x_name, yaxis_title=y_name, title_x=0.5)
    if save_path:
        fig.write_image(save_path)
    else:
        fig.show()


def create_scatter_plot(df, x, y, x_name, y_name, title, color, trendline_color, save_path: str = None):
    """
    Creates a scatter plot with a trend line using the specified DataFrame and column names.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    x (str): The name of the column to be used as the x-axis.
    y (str): The name of the column to be used as the y-axis.
    x_name (str): The label for the x-axis.
    y_name (str): The label for the y-axis.
    title (str): The title of the scatter plot.
    color (str): The color of the scatter plot markers.
    trendline_color (str): The color of the trendline.
    save_path (str): The path to save the chart as a PNG.

    Returns:
    None
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df[x], y=df[y], mode="markers", name="Data", marker=dict(color=color)
        )
    )

    slope, intercept, _, _, _ = stats.linregress(df[x], df[y])
    x_range = np.linspace(df[x].min(), df[x].max(), 100)
    y_range = slope * x_range + intercept

    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_range,
            mode="lines",
            name="Trendline",
            line=dict(color=trendline_color),
        )
    )
    fig.update_layout(xaxis_title=x_name, yaxis_title=y_name, title=title, title_x=0.5)
    if save_path:
        fig.write_image(save_path)
    else:
        fig.show()


def plot_histogram(df, column, title, x_axis_title, y_axis_title, color, save_path: str = None):
    """
    Creates a histogram using the specified DataFrame and column name.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    column (str): The name of the column to be used.
    title (str): The title of the histogram.
    x_axis_title (str): The title of the x-axis.
    y_axis_title (str): The title of the y-axis.
    color (str): The color of the histogram bars.
    save_path (str): The path to save the chart as a PNG.

    Returns:
    None
    """
    total = len(df[column])

    counts, bins = np.histogram(df[column], bins=range(1, 7))

    percentages = np.round(counts / total * 100, decimals=2)

    percentages_text = [str(p) + "%" for p in percentages]

    fig = go.Figure(
        data=[
            go.Bar(
                x=bins[:-1],
                y=percentages,
                marker_color=color,
                text=percentages_text,
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title_text=title,
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        title_x=0.5,
    )
    if save_path:
        fig.write_image(save_path)
    else:
        fig.show()


def create_horizontal_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    x_name: str,
    y_name: str,
    title: str,
    color_list: List[str],
    save_path: str = None,
) -> None:
    """
    Creates a horizontal bar chart using the specified dataframe and column names.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data.
    x (str): The name of the column to be used as the x-axis.
    y (str): The name of the column to be used as the y-axis.
    x_name (str): The name to be displayed on the x-axis.
    y_name (str): The name to be displayed on the y-axis.
    title (str): The title of the chart.
    color_list (list): A list of colors for the bars.
    save_path (str): The path to save the chart as a PNG.

    Returns:
    None
    """
    df["rating"] = df["rating"].astype(str)

    sorted_ratings = sorted(df["rating"].unique())

    color_map = {
        rating: color_list[i % len(color_list)]
        for i, rating in enumerate(sorted_ratings)
    }

    fig = px.bar(
        df,
        y=y,
        x=x,
        orientation="h",
        title=title,
        color="rating",
        color_discrete_map=color_map,
    )

    fig.update_layout(xaxis_title=x_name, yaxis_title=y_name, title_x=0.5)
    if save_path:
        fig.write_image(save_path)
    else:
        fig.show()


def perform_sentiment_and_frequency_analysis(
    df: pd.DataFrame, text_column: str
) -> None:
    """Perform sentiment analysis and frequency analysis on reviews.

    Args:
        df (pandas.DataFrame): The DataFrame containing the reviews.
        text_column (str): The name of the column in the DataFrame that contains the text reviews.

    Raises:
        ValueError: If the specified text_column does not exist in the DataFrame.
    """

    if text_column not in df.columns:
        raise ValueError(f"'{text_column}' does not exist in the DataFrame.")

    df["sentiment"] = df[text_column].apply(lambda x: TextBlob(x).sentiment.polarity)
    df["sentiment_category"] = df["sentiment"].apply(
        lambda x: "positive" if x > 0 else "negative"
    )

    for category, group in df.groupby("sentiment_category"):
        words = " ".join(group[text_column]).split()
        common_words = Counter(words).most_common(10)
        print(f"Common words in {category} reviews:", common_words)
        generate_wordcloud(common_words)


cmap_colors: List[str] = ["#ffa600", "#ff8ca3", "#ff9061", "#ff9cd7", "#dcb0f2"]
cmap: LinearSegmentedColormap = LinearSegmentedColormap.from_list("mycmap", cmap_colors)


def generate_wordcloud(common_words: dict, save_path: str = None) -> None:
    """
    Generates a word cloud visualization based on the frequency of words in a given dictionary.

    Parameters:
    common_words (dict): A dictionary containing words as keys and their frequencies as values.
    save_path (str): The path to save the word cloud image.

    Returns:
    None
    """
    wordcloud = WordCloud(
        width=800, height=400, background_color="white", colormap=cmap
    ).generate_from_frequencies(dict(common_words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def create_xyz_scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    color_by: str,
    x_name: str,
    y_name: str,
    title: str,
    color_scale: List[str],
    save_path: str = None,
) -> None:
    """
    Creates a scatter plot with a color scale using the specified DataFrame and column names.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    x (str): The name of the column to be used as the x-axis.
    y (str): The name of the column to be used as the y-axis.
    color_by (str): The name of the column to be used for the color scale.
    x_name (str): The label for the x-axis.
    y_name (str): The label for the y-axis.
    title (str): The title of the scatter plot.
    color_scale (List[str]): The color scale to use for the markers.
    save_path (str): The path to save the chart as a PNG.

    Returns:
    None
    """
    fig = go.Figure()

    scatter = go.Scatter(
        x=df[x],
        y=df[y],
        mode="markers",
        marker=dict(
            color=df[color_by], colorscale=color_scale, colorbar=dict(title=color_by)
        ),
        hovertemplate="<b>Date</b>: %{x}<br>"
        + "<b>Number of Reviews</b>: %{y}<br>"
        + "<b>Average Rating</b>: %{marker.color:.2f}<br>",
        name="Data",
    )

    fig.add_trace(scatter)

    fig.update_layout(xaxis_title=x_name, yaxis_title=y_name, title=title, title_x=0.5)
    if save_path:
        fig.write_image(save_path)
    else:
        fig.show()


def plot_monthly_reviews_and_ratings(
    df: pd.DataFrame,
    x_col: str,
    bar_y_col: str,
    line_y_col: str,
    x_label: str,
    bar_label: str,
    line_label: str,
    bar_color: str,
    line_color: str,
    save_path: str = None,
) -> None:
    """
    Plots monthly reviews and ratings using a bar chart and a line chart.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data.
    x_col (str): The column name for the x-axis values.
    bar_y_col (str): The column name for the y-axis values of the bar chart.
    line_y_col (str): The column name for the y-axis values of the line chart.
    x_label (str): The label for the x-axis.
    bar_label (str): The label for the y-axis of the bar chart.
    line_label (str): The label for the y-axis of the line chart.
    bar_color (str): The color of the bars in the bar chart.
    line_color (str): The color of the line in the line chart.
    save_path (str): The path to save the chart as a PNG.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=df[x_col], y=df[bar_y_col], name=bar_label, marker_color=bar_color),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df[x_col], y=df[line_y_col], name=line_label, line_color=line_color
        ),
        secondary_y=True,
    )

    fig.update_xaxes(title_text=x_label)

    fig.update_yaxes(title_text=bar_label, secondary_y=False)
    fig.update_yaxes(title_text=line_label, secondary_y=True)

    if save_path:
        fig.write_image(save_path)
    else:
        fig.show()


def create_box_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    color: str,
    save_path: str = None,
) -> None:
    """
    Create a box plot using the given dataframe and column names.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data.
    x_col (str): The column name for the x-axis.
    y_col (str): The column name for the y-axis.
    title (str): The title of the plot.
    x_label (str): The label for the x-axis.
    y_label (str): The label for the y-axis.
    color (str): The color of the markers.
    save_path (str): The path to save the chart as a PNG.

    Returns:
    None
    """
    fig = px.box(df, x=x_col, y=y_col)
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        xaxis_title=x_label,
        yaxis_title=y_label,
    )
    fig.update_traces(marker_color=color)
    if save_path:
        fig.write_image(save_path)
    else:
        fig.show()


def process_chunk(chunk):
    """
    Process a DataFrame chunk by removing non-English rows using the 'remove_non_english' function.
    
    Parameters:
        chunk (pandas.DataFrame): A chunk of the DataFrame.

    Returns:
        pandas.DataFrame: The processed DataFrame chunk with non-English rows removed.
    """
    required_cols = [col for col in ["title", "content"] if col in chunk.columns]
    return remove_non_english(chunk, required_cols)


def parallel_process(df, n_processes=4):
    """
    Process the DataFrame in parallel to remove non-English rows.

    Parameters:
        df (pandas.DataFrame): The DataFrame to process.
        n_processes (int): The number of processes to use.

    Returns:
        pandas.DataFrame: The processed DataFrame with non-English rows removed.
    """
    chunks = np.array_split(df.index, n_processes)
    df_chunks = [df.loc[chunk] for chunk in chunks]
    with Pool(n_processes) as pool:
        results = pool.map(process_chunk, df_chunks)
    return pd.concat(results)
