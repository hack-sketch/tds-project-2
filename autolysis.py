# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "seaborn", "matplotlib", "requests", "chardet", "scikit-learn", "python-dotenv", "uv"]
# ///


import os
import sys
import glob  
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

def check_filepath():
    if len(sys.argv) < 2:
        print("Usage: python autolysis.py <dataset1.csv> [dataset2.csv ...]")
        # If no arguments, look for CSV files in the current directory
        csv_files = glob.glob('*.csv')
        if not csv_files:
            print("No CSV files found in the current directory.")
            sys.exit(1)
        return csv_files
    return sys.argv[1:]

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


def generate_scatterplot(df, profile, api_key, output_dirs):
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
    
    # Save plot to all output directories
    for output_dir in output_dirs:
        output_path = os.path.join(output_dir, f'{x_column_safe}_{y_column_safe}_scatterplot.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Scatterplot saved to {output_path}")
    plt.close()


def generate_correlation_heatmap(df, output_dirs):
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.shape[1] < 2:
        print("Not enough numeric columns for correlation heatmap.")
        return

    correlation_matrix = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')

    # Save plot to all output directories
    for output_dir in output_dirs:
        output_path = os.path.join(output_dir, 'correlation_heatmap.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Correlation heatmap saved to {output_path}")
    plt.close()


def generate_cluster_data(df, profile, api_key, output_dirs):
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

    # Save plot to all output directories
    for output_dir in output_dirs:
        output_path = os.path.join(output_dir, 'clustering_plot.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Clustering plot saved to {output_path}")
    plt.close()


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


def process_plots_and_create_readme(dataset_file, api_key, headers_json, sample_data, df, output_dirs):
    dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]
    
    # Use the plots from the first output directory
    plot_dir = output_dirs[0]
    plot_files = [f for f in os.listdir(plot_dir) if f.endswith('.png')]

    if not plot_files:
        print(f"No plot images found in {plot_dir}.")
        return

    readme_content = f"# Data Analysis Report for {dataset_name}\n\n"

    for plot_file in plot_files:
        plot_path = os.path.join(plot_dir, plot_file)
        print(f"Processing {plot_path}...")

        if 'scatterplot' in plot_file:
            plot_type = 'scatterplot'
            base_name = os.path.splitext(plot_file)[0]
            parts = base_name.split('_')
            x_col = parts[0]
            y_col = parts[1] if len(parts) > 1 else ''
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
            readme_content += f"![{plot_file}]({plot_file})\n\n"
            readme_content += f"{story}\n\n"
        else:
            print(f"Failed to generate narrative for {plot_file}.")

    # Write README to all output directories
    for output_dir in output_dirs:
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as readme_file:
            readme_file.write(readme_content)
        print(f"README.md created at {readme_path}")


if __name__ == "__main__":
    try:
        api_key = os.environ.get("AIPROXY_TOKEN")
        if not api_key:
            raise ValueError("AIPROXY_TOKEN environment variable not set.")

        # Set the base directory where files should be saved
        base_dir = os.path.expanduser("~/.local/share/tds-sep-24-project-2/hack-sketch-tds-project-2/eval")
        os.makedirs(base_dir, exist_ok=True)

        # Handle case when no arguments are provided
        csv_files = sys.argv[1:] if len(sys.argv) > 1 else glob.glob(os.path.join(base_dir, '*.csv'))

        # Process all provided CSV files
        for dataset_file in csv_files:
            if not dataset_file.endswith('.csv'):
                print(f"Skipping {dataset_file} - not a CSV file")
                continue
                
            print(f"\nProcessing {dataset_file}")
            try:
                df = load_dataset(dataset_file)
                headers_json = get_headers_as_json(df)
                sample_data = df.sample(n=min(5, len(df))).to_dict(orient='records')
                profile = profile_dataset(df)

                # Create output directories in both locations
                dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]
                output_dir_base = os.path.join(base_dir, dataset_name) + ".csv"
                output_dir_current = os.path.join(os.getcwd(), dataset_name)
                os.makedirs(output_dir_base, exist_ok=True)
                os.makedirs(output_dir_current, exist_ok=True)

                output_dirs = [output_dir_base, output_dir_current]

                generate_scatterplot(df, profile, api_key, output_dirs)
                generate_correlation_heatmap(df, output_dirs)
                generate_cluster_data(df, profile, api_key, output_dirs)

                process_plots_and_create_readme(dataset_file, api_key, headers_json, sample_data, df, output_dirs)
                print(f"Completed processing {dataset_file}")
            except Exception as e:
                print(f"Error processing {dataset_file}: {e}")
                continue

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
