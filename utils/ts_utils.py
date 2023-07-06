import numpy as np
import json
import stumpy
import os
import cv2
import pandas as pd
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import subprocess
from typing import List, Tuple, Optional, Dict, Any, Union
from pandas import DataFrame, Series
import matplotlib.pyplot as plt


def plot_query_and_results_multi(query: List[Series], results: List[List[Series]], topic_names: List[str],
                                 intervals: List[Tuple[int, int]]) -> None:
    """
    Plots the query and most similar subsequences for one or more topics.

    Args:
        query (List[Series]): List of queries, one per topic.
        results (List[List[Series]]): List of result lists, one per topic.
        topic_names (List[str]): List of topic names.
        intervals (List[Tuple[int, int]]): List of intervals for the most similar sequences.
    """

    if len(topic_names) == 1:
        # Handle single topic case
        fig, ax = plt.subplots(figsize=(18.5, 10.5))
        ax.plot(query[0].index, query[0], label='Query', linewidth=2, color='black')
        for j, (result, interval) in enumerate(zip(results[0], intervals)):
            result.index = range(query[0].index[0], query[0].index[0] + len(result))
            ax.plot(result.index, result, label=f'Sequence {interval}', linewidth=2)
        ax.set_title(f'{topic_names[0]} with Query and Most Similar Sequences')
        ax.set(xlabel='Time', ylabel=topic_names[0])
        ax.legend()
    else:
        fig, axs = plt.subplots(len(topic_names), figsize=(18.5, 10.5 * len(topic_names)))
        for i, (topic, query_topic, results_topic) in enumerate(zip(topic_names, query, results)):
            axs[i].plot(query_topic.index, query_topic, label='Query', linewidth=2, color='black')
            for j, (result, interval) in enumerate(zip(results_topic, intervals)):
                result.index = range(query_topic.index[0], query_topic.index[0] + len(result))
                axs[i].plot(result.index, result, label=f'Sequence {interval}', linewidth=2)
            axs[i].set_title(f'{topic} with Query and Most Similar Sequences')
            axs[i].set(xlabel='Time', ylabel=topic)
            axs[i].legend()

    plt.tight_layout()
    plt.show()


def plot_sequence_on_full_data_multi(df: DataFrame, queries: List[Series], results: List[List[Series]],
                                     topic_names: List[str]) -> None:
    """
    Plots the full time series, the query, and the most similar subsequences for one or more topics.

    Args:
        df (DataFrame): DataFrame of full time series.
        queries (List[Series]): List of queries, one per topic.
        results (List[List[Series]]): List of result lists, one per topic.
        topic_names (List[str]): List of topic names.
    """

    if len(topic_names) == 1:
        topic_name = topic_names[0]
        fig, ax = plt.subplots()
        ax.plot(df[topic_name], label='Full Time Series')
        ax.plot(queries[0].index, queries[0], label='Query', linewidth=2)
        for j, result in enumerate(results[0], start=1):
            ax.plot(result.index, result, label=f'Most Similar Sequence {j}', linewidth=2)
        ax.set_title(f'Time Series with Query and Most Similar Sequences Highlighted - {topic_name}')
        ax.set(xlabel='Time', ylabel=topic_name)
        ax.legend()
    else:
        fig, axs = plt.subplots(len(topic_names))
        fig.set_size_inches(18.5, 10.5 * len(topic_names))
        for i, topic_name in enumerate(topic_names):
            ax = axs[i]
            ax.plot(df[topic_name], label='Full Time Series')
            ax.plot(queries[i].index, queries[i], label='Query', linewidth=2)
            for j, result in enumerate(results[i], start=1):
                ax.plot(result.index, result, label=f'Most Similar Sequence {j}', linewidth=2)
            ax.set_title(f'Time Series with Query and Most Similar Sequences Highlighted - {topic_name}')
            ax.set(xlabel='Time', ylabel=topic_name)
            ax.legend()
    plt.tight_layout()
    plt.show()


def get_similar_intervals_multi(filename: str, start_index: int, end_index: int, topic_names: List[str], n: int) -> Tuple[List[List[pd.Series]], List[Tuple[int, int]]]:
    """
    Finds the n most similar intervals (subsequences) to a given query interval for one or more topics.

    Args:
        filename (str): Path to the CSV file.
        start_index (int): Start index of the query interval.
        end_index (int): End index of the query interval.
        topic_names (List[str]): List of topic names.
        n (int): Number of similar subsequences to find.

    Returns:
        Tuple[List[List[pd.Series]], List[Tuple[int, int]]]: Tuple containing a list of lists of similar subsequences
        (one list per topic) and a list of tuples representing the intervals of the similar subsequences.
    """

    df = pd.read_csv(filename)
    queries = [df.loc[start_index:end_index, topic] for topic in topic_names]
    ts_datas = [df[topic] for topic in topic_names]
    queries = [query.astype(float) for query in queries]
    ts_datas = [ts_data.astype(float) for ts_data in ts_datas]

    query_nps = [query.to_numpy() for query in queries]
    ts_data_nps = [ts_data.to_numpy() for ts_data in ts_datas]

    matches = [stumpy.match(query_np, ts_data_np, max_matches=n+1) for query_np, ts_data_np in zip(query_nps, ts_data_nps)]

    combined_matches = np.concatenate(matches)
    combined_matches = combined_matches[combined_matches[:, 0].argsort()]

    def check_overlap(new_interval: Tuple[int, int], intervals: List[Tuple[int, int]]) -> bool:
        new_start, new_end = new_interval
        for start, end in intervals:
            if not (new_end < start or new_start > end):
                return True
        return False

    selected_intervals = []
    for _, idx in combined_matches:
        interval = (idx, idx + len(query_nps[0]) - 1)
        if not check_overlap(interval, selected_intervals):
            if interval != (start_index, end_index):
                selected_intervals.append(interval)
                if len(selected_intervals) == n:
                    break

    return [[df.loc[start:end, topic] for start, end in selected_intervals] for topic in topic_names], selected_intervals


def draw_trajectory(gt_df: pd.DataFrame, start_time: float, end_time: float, current_time: float,
                    img_size: Tuple[int, int],
                    start_time_query_seconds: Optional[float] = None,
                    end_time_query_seconds: Optional[float] = None) -> np.ndarray:
    """
    Draw a trajectory path based on time-stamped positions in a DataFrame.

    Args:
        gt_df (pd.DataFrame): Ground truth DataFrame containing columns 'pose.position.x', 'pose.position.y', 'Time'.
        start_time (float): Start time for the interval of interest.
        end_time (float): End time for the interval of interest.
        current_time (float): Current time for the position of interest.
        img_size (Tuple[int, int]): Size of the output image as (height, width).
        start_time_query_seconds (Optional[float], default=None): Start time for the query interval.
        end_time_query_seconds (Optional[float], default=None): End time for the query interval.

    Returns:
        np.ndarray: Image with trajectory drawn.
    """

    img = np.ones(img_size + (3,), np.uint8) * 0

    df_filtered = gt_df[['pose.position.x', 'pose.position.y', 'Time']].copy()

    positions_x = df_filtered['pose.position.x'].values
    positions_y = df_filtered['pose.position.y'].values
    times = df_filtered['Time'].values

    cv2.polylines(img, np.int32([np.column_stack((positions_x, positions_y))]),
                  isClosed=False, color=(0, 255, 0), thickness=2)

    df_interval = df_filtered[(times >= start_time) & (times <= end_time)]

    if start_time_query_seconds is not None and end_time_query_seconds is not None:
        df_interval_query = df_filtered[(times >= start_time_query_seconds) & (times <= end_time_query_seconds)]
        cv2.polylines(img, np.int32([np.column_stack((df_interval_query['pose.position.x'],
                                                      df_interval_query['pose.position.y']))]),
                      isClosed=False, color=(255, 255, 0), thickness=3)

    cv2.polylines(img, np.int32([np.column_stack((df_interval['pose.position.x'],
                                                  df_interval['pose.position.y']))]),
                  isClosed=False, color=(0, 0, 255), thickness=2)

    current_position = df_filtered[(times >= current_time - 0.01) & (times <= current_time + 0.2)]

    if len(current_position) > 0:
        cv2.circle(img, (current_position['pose.position.x'].values[0],
                         current_position['pose.position.y'].values[0]),
                   8, (255, 0, 0), -1)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def create_interactive_plot(csv_file: str, column_names: List[str]) -> Tuple[dash.Dash, List[int]]:
    """
    Create an interactive plot using Dash and Plotly.

    Args:
        csv_file (str): Path to the CSV file.
        column_names (List[str]): List of column names to plot.

    Returns:
        Tuple[dash.Dash, List[int]]: Dash application instance and a list of selected indices.
    """
    df = pd.read_csv(csv_file)

    fig = go.Figure()
    for column_name in column_names:
        fig.add_trace(go.Scatter(x=df.index, y=df[column_name], mode='lines', name=column_name))

    app = dash.Dash(__name__)

    app.layout = html.Div([
        dcc.Graph(id='plot', figure=fig),
        html.Div(id='selected-window')
    ])

    selected_indices = []

    @app.callback(
        dash.dependencies.Output('selected-window', 'children'),
        [dash.dependencies.Input('plot', 'relayoutData')]
    )
    def update_selected_window(relayout_data):
        if relayout_data is not None and 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
            first_index = int(relayout_data['xaxis.range[0]'])
            last_index = int(relayout_data['xaxis.range[1]'])
            selected_indices.clear()
            selected_indices.extend([first_index, last_index])
            return f'Selected Window: {first_index} - {last_index}'

    app.run_server(mode='inline')

    return app, selected_indices


def preprocess_ground_truth(filename: str, first_timestamp: float, img_size: Tuple[int, int]) -> pd.DataFrame:
    """
    Preprocesses ground truth data by normalizing and scaling positional data, and adjusting time values.

    Args:
        filename (str): Path to the CSV file containing the data.
        first_timestamp (float): Timestamp to adjust the time data by.
        img_size (Tuple[int, int]): Tuple specifying the size of the image (width, height).

    Returns:
        pd.DataFrame: The preprocessed data in a DataFrame.
    """
    df = pd.read_csv(filename)
    df['Time'] -= first_timestamp

    # Normalize positions to the range [0, 1]
    for axis in ['x', 'y']:
        df[f'pose.position.{axis}'] = (df[f'pose.position.{axis}'] - df[f'pose.position.{axis}'].min()) / (
            df[f'pose.position.{axis}'].max() - df[f'pose.position.{axis}'].min())

    # Scale positions to image size and convert to integer
    df['pose.position.x'] = (df['pose.position.x'] * (img_size[0] - 1)).astype(int)
    df['pose.position.y'] = (df['pose.position.y'] * (img_size[1] - 1)).astype(int)

    # Flip Y coordinates to match Cartesian coordinate system
    df['pose.position.y'] = img_size[1] - df['pose.position.y']

    return df


def show_imgs_based_on_timerange(start_time_seconds: float, end_time_seconds: float, img_folder: str,
                                 gt_df: pd.DataFrame, img_manifest: Dict[str, Any],
                                 cached_images: Dict[str, np.ndarray]) -> None:
    """
    Visualizes images based on a specified time range.

    Args:
        start_time_seconds (float): Start time in seconds.
        end_time_seconds (float): End time in seconds.
        img_folder (str): Path to the folder containing images.
        gt_df (pd.DataFrame): Ground truth DataFrame.
        img_manifest (Dict[str, Any]): Image manifest as a dictionary.
        cached_images (Dict[str, np.ndarray]): Cached images as a dictionary mapping image paths to image arrays.

    Returns:
        None
    """
    first_key = list(img_manifest["images"].keys())[0]
    first_rosbag_timestamp = img_manifest["images"][first_key]["rosbag_timestamp"]

    for k in img_manifest["images"].keys():
        entry = img_manifest["images"][k]
        timestamp_s = (entry["rosbag_timestamp"] - first_rosbag_timestamp) / 1_000_000_000
        img_path = os.path.join(img_folder, k)

        if start_time_seconds <= timestamp_s <= end_time_seconds:
            img_map = draw_trajectory(gt_df, start_time_seconds, end_time_seconds, timestamp_s, (500, 500))
            img = cached_images[img_path]

            # Pad images to be the same size
            height_diff = img.shape[0] - img_map.shape[0]
            width_diff = img.shape[1] - img_map.shape[1]

            if height_diff != 0:
                border_width = (0, abs(height_diff))
                border_value = [0, 0, 0] if height_diff > 0 else [255, 255, 255]
                img_map = cv2.copyMakeBorder(img_map, 0, border_width[1], 0, 0, cv2.BORDER_CONSTANT, value=border_value)
                img = cv2.copyMakeBorder(img, 0, border_width[0], 0, 0, cv2.BORDER_CONSTANT, value=border_value)

            if width_diff != 0:
                border_width = (0, abs(width_diff))
                border_value = [0, 0, 0] if width_diff > 0 else [255, 255, 255]
                img_map = cv2.copyMakeBorder(img_map, 0, 0, 0, border_width[1], cv2.BORDER_CONSTANT, value=border_value)
                img = cv2.copyMakeBorder(img, 0, 0, 0, border_width[0], cv2.BORDER_CONSTANT, value=border_value)

            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # Concatenate the images along the second axis
            concatenated_img = np.concatenate((img, img_map), axis=1)
            cv2.imshow("img", concatenated_img)
            cv2.waitKey(1)


def show_video_based_on_timerange(
    start_time_seconds: float,
    end_time_seconds: float,
    img_folder: str,
    gt_df: 'pandas.DataFrame',
    img_manifest: Dict[str, Union[Dict[str, str], Dict[str, int]]],
    output_path: str,
    start_time_query_seconds: Optional[float] = None,
    end_time_query_seconds: Optional[float] = None) -> str:
    """Creates a video from images based on given time range.

    Args:
        start_time_seconds (float): The start time in seconds.
        end_time_seconds (float): The end time in seconds.
        img_folder (str): The folder where the images are stored.
        gt_df (pandas.DataFrame): Ground truth data frame.
        img_manifest (Dict[str, Union[Dict[str, str], Dict[str, int]]]):
            The image manifest containing image keys and metadata.
        output_path (str): The output path where the video should be saved.
        start_time_query_seconds (Optional[float], optional):
            Query start time in seconds. Defaults to None.
        end_time_query_seconds (Optional[float], optional):
            Query end time in seconds. Defaults to None.

    Returns:
        str: Path of the output video in .webm format.
    """
    first_key = list(img_manifest["images"].keys())[0]
    first_rosbag_timestamp = img_manifest["images"][first_key]["rosbag_timestamp"]

    output_video_path = os.path.join(output_path, "output.mp4")
    output_video_path_webm = os.path.join(output_path, f"{str(start_time_seconds)}_{str(end_time_seconds)}_output.webm")

    # Video properties
    fps = 30  # Frames per second
    img_size = (500, 500)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (1280, 500))

    for k in img_manifest["images"].keys():
        entry = img_manifest["images"][k]
        timestamp_s = (entry["rosbag_timestamp"] - first_rosbag_timestamp) / 1_000_000_000

        img_path = os.path.join(img_folder, k)

        if timestamp_s >= start_time_seconds and timestamp_s <= end_time_seconds:
            img_map = draw_trajectory(gt_df,
                                      start_time_seconds,
                                      end_time_seconds,
                                      timestamp_s,
                                      img_size,
                                      start_time_query_seconds,
                                      end_time_query_seconds)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Ensure the images are the same height
            height_diff = img.shape[0] - img_map.shape[0]
            width_diff = img.shape[1] - img_map.shape[1]

            # If one image is smaller, pad it with a white border to match the other image's size
            if height_diff > 0:  # img_map is smaller in height
                img_map = cv2.copyMakeBorder(img_map, 0, height_diff, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            elif height_diff < 0:  # img is smaller in height
                img = cv2.copyMakeBorder(img, 0, -height_diff, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            if width_diff > 0:  # img_map is smaller in width
                img_map = cv2.copyMakeBorder(img_map, 0, 0, 0, width_diff, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            elif width_diff < 0:  # img is smaller in width
                img = cv2.copyMakeBorder(img, 0, 0, 0, -width_diff, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # Concatenate the images along the second axis
            concatenated_img = np.concatenate((img, img_map), axis=1)

            # Write frame to video
            out.write(concatenated_img)

    # Release the video writer
    out.release()

    # Use ffmpeg to convert the video to WebM format with VP9 codec
    subprocess.run(
       ['ffmpeg', '-y', '-i', output_video_path, '-c:v', 'libvpx-vp9', '-c:a', 'libvorbis', '-b:v', '1M', '-b:a', '192k',
        output_video_path_webm], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return output_video_path_webm