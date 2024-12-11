import os
import sys
import pandas as pd
import chardet
import json
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import base64
from dotenv import load_dotenv
load_dotenv()

def load_dataset(file_path):
    """
    Load a dataset from a CSV file, attempting to detect encoding if necessary.
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            sys.exit("The dataset is empty.")
        return df
    except UnicodeDecodeError:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']
        df = pd.read_csv(file_path, encoding=encoding)
        if df.empty:
            sys.exit("The dataset is empty.")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)


def get_headers_as_json(df):
    headers = df.columns.tolist()
    headers_json = json.dumps({"headers": headers})
    return headers_json


def profile_dataset(df):
    summary = {
        "shape": df.shape,
        "null_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.apply(str).to_dict(),
        "numerical_summary": df.describe().to_dict(),
        "headers": df.columns.tolist(),
        "sample_data": df.head(3).to_dict(orient='records')
    }
    return summary


def generate_scatterplot(df, profile, api_key, output_dir):
    try:
        response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "Given the following dataset analysis, suggest two numeric columns from the dataset that would make an interesting scatterplot. Return only the column names, separated by a comma. No explanation needed. If there are not enough numeric columns, return an empty string."
                    },
                    {
                        "role": "user",
                        "content": json.dumps(profile)
                    }
                ],
                "temperature": 0
            }
        )
        response.raise_for_status()
        content = response.json()['choices'][0]['message']["content"]
        if not content.strip():
            print("No columns suggested for scatterplot.")
            return
        fields_for_scatterplot = content.strip().split(',')
    except Exception as e:
        print(f"API request failed: {e}")
        return

    if len(fields_for_scatterplot) != 2:
        print("Not enough columns returned for scatterplot.")
        return

    fields_for_scatterplot = [field.strip() for field in fields_for_scatterplot]
    x_column, y_column = fields_for_scatterplot

    if x_column not in df.columns or y_column not in df.columns:
        print("Columns not found in DataFrame.")
        return
    if not pd.api.types.is_numeric_dtype(df[x_column]) or not pd.api.types.is_numeric_dtype(df[y_column]):
        print("Selected columns are not numeric.")
        return

    df_cleaned = df.dropna(subset=[x_column, y_column])

    if df_cleaned.empty:
        print("No data available for the selected columns after dropping NaNs.")
        return

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_cleaned, x=x_column, y=y_column)
    plt.title(f'Scatterplot between {x_column} and {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)

    x_column_safe = x_column.replace(" ", "_")
    y_column_safe = y_column.replace(" ", "_")
    output_path = os.path.join(output_dir, f'{x_column_safe}_{y_column_safe}_scatterplot.png')

    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Scatterplot saved to {output_path}")


def generate_correlation_heatmap(df, output_dir):
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.shape[1] < 2:
        print("Not enough numeric columns for correlation heatmap.")
        return

    correlation_matrix = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')

    output_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Correlation heatmap saved to {output_path}")


def generate_cluster_data(df, profile, api_key, output_dir):
    try:
        response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "Given the following dataset summary, which numeric columns are suitable for clustering? Exclude any IDs or non-informative columns. Please provide a maximum of 5 columns. Return only the column names, separated by a comma. No explanation needed. If the dataset is not suitable for clustering, return an empty string."
                    },
                    {
                        "role": "user",
                        "content": json.dumps(profile)
                    }
                ],
                "temperature": 0
            }
        )
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        if not content.strip():
            print("No columns suggested for clustering.")
            return
        columns = [col.strip() for col in content.split(',') if col.strip()]
    except Exception as e:
        print(f"API request failed: {e}")
        return

    if len(columns) < 2:
        print("Not enough suitable columns for clustering.")
        return

    selected_columns = [col for col in columns if col in df.columns]
    if len(selected_columns) < 2:
        print("Selected columns are not present in the DataFrame.")
        return

    df_numeric = df[selected_columns].select_dtypes(include='number')
    df_numeric = df_numeric.dropna()
    if df_numeric.shape[0] < 3:
        print("Not enough data points after dropping NaNs for clustering.")
        return

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_numeric)

    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_numeric['Cluster'] = kmeans.fit_predict(data_scaled)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=df_numeric.iloc[:, 0],
        y=df_numeric.iloc[:, 1],
        hue='Cluster',
        data=df_numeric,
        palette='viridis',
        s=100,
        alpha=0.7
    )
    plt.title('KMeans Clustering')
    plt.xlabel(selected_columns[0])
    plt.ylabel(selected_columns[1])
    plt.legend(title='Cluster')

    output_path = os.path.join(output_dir, 'clustering_plot.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Clustering plot saved to {output_path}")


def get_plot_description(df, plot_type, columns):
    if plot_type == 'scatterplot':
        x_col, y_col = columns
        desc = f"A scatterplot of '{y_col}' versus '{x_col}' based on the dataset."
    elif plot_type == 'heatmap':
        desc = "A correlation heatmap showing correlations between numeric variables in the dataset."
    elif plot_type == 'clustering':
        desc = f"A scatterplot showing KMeans clustering results using variables '{columns[0]}' and '{columns[1]}'."
    else:
        desc = "Plot description not available."
    return desc


def get_plot_narrative(api_key, plot_description, headers_json, sample_data):
    endpoint = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are a data analyst who writes detailed and engaging narratives for data plots."
            },
            {
                "role": "user",
                "content": f"Create a detailed and engaging story based on the following plot description:\n\n{plot_description}\n\nInclude insights, trends, and any interesting findings, but do not reference any images directly. Here is some context about the dataset:\n\nHeaders: {headers_json}\n\nSample data: {sample_data}"
            }
        ],
        "temperature": 0.7,
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        story = result['choices'][0]['message']['content']
        return story
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Unexpected response format: {e}")
        return None


def process_plots_and_create_readme(dataset_file, api_key, headers_json, sample_data):
    output_dir = os.path.splitext(dataset_file)[0]
    if not os.path.exists(output_dir):
        print(f"Directory does not exist: {output_dir}")
        return

    plot_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]

    if not plot_files:
        print("No plot images found in the directory.")
        return

    readme_path = os.path.join(output_dir, "README.md")
    readme_content = "# Data Analysis Report\n\n"

    for plot_file in plot_files:
        plot_path = os.path.join(output_dir, plot_file)
        print(f"Processing {plot_path}...")

        if 'scatterplot' in plot_file:
            plot_type = 'scatterplot'
            base_name = os.path.splitext(plot_file)[0]
            parts = base_name.split('_')
            x_col = parts[0]
            y_col = parts[1]
            columns = (x_col, y_col)
        elif 'heatmap' in plot_file:
            plot_type = 'heatmap'
            columns = None
        elif 'clustering' in plot_file:
            plot_type = 'clustering'
            columns = df.columns[:2].tolist()
        else:
            plot_type = 'unknown'
            columns = None

        plot_desc = get_plot_description(df, plot_type, columns)

        story = get_plot_narrative(api_key, plot_desc, headers_json, sample_data)

        if story:
            readme_content += f"## {os.path.splitext(plot_file)[0]}\n\n"
            readme_content += f"![{plot_file}](./{plot_file})\n\n"
            readme_content += f"{story}\n\n"
        else:
            print(f"Failed to generate narrative for {plot_file}.")

    with open(readme_path, "w", encoding="utf-8") as readme_file:
        readme_file.write(readme_content)
    print(f"README.md created at {readme_path}")


def check_filepath():
    if len(sys.argv) < 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)
    dataset_file = sys.argv[1]
    return dataset_file


if __name__ == "__main__":
    try:
        api_key = os.environ["AIPROXY_TOKEN"]
    except KeyError:
        raise ValueError("AIPROXY_TOKEN environment variable not set.")

    dataset_file = check_filepath()
    df = load_dataset(dataset_file)

    headers_json = get_headers_as_json(df)
    sample_data = df.sample(n=min(5, len(df))).to_dict(orient='records')

    profile = profile_dataset(df)

    output_dir = os.path.splitext(dataset_file)[0]
    os.makedirs(output_dir, exist_ok=True)

    generate_scatterplot(df, profile, api_key, output_dir)
    generate_correlation_heatmap(df, output_dir)
    generate_cluster_data(df, profile, api_key, output_dir)

    process_plots_and_create_readme(dataset_file, api_key, headers_json, sample_data)