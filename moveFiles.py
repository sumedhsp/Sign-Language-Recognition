import os
import json
import re
import shutil

def move_videos_with_youtube_urls(json_file_path, video_folder_path, destination_folder_path):
    """
    Moves video files with YouTube URLs from the source folder to a destination folder.

    Parameters:
    - json_file_path (str): Path to the JSON file containing video metadata.
    - video_folder_path (str): Path to the folder containing videos.
    - destination_folder_path (str): Path to the folder where videos will be moved.

    Returns:
    - None
    """
    try:
        # Ensure destination folder exists
        os.makedirs(destination_folder_path, exist_ok=True)

        # Load the JSON metadata
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

        # Regular expression to identify YouTube URLs
        youtube_regex = re.compile(r'(https?://)?(www\.)?(youtube\.com|youtu\.be)/')

        # Collect video IDs from YouTube URLs in the JSON
        youtube_video_ids = set()
        for entry in data:
            for instance in entry['instances']:
                url = instance.get('url', '')
                if youtube_regex.search(url):  # Check if URL is a YouTube link
                    video_id = instance.get('video_id')
                    if video_id:
                        youtube_video_ids.add(video_id)

        # Check and move corresponding files
        for video_id in youtube_video_ids:
            video_file = os.path.join(video_folder_path, f"{video_id}.mp4")  # Assuming .mp4 extension
            destination_file = os.path.join(destination_folder_path, f"{video_id}.mp4")
            if os.path.exists(video_file):
                print(f"Moving file: {video_file} -> {destination_file}")
                shutil.move(video_file, destination_file)
            else:
                print(f"File not found, skipping: {video_file}")

        print("File moving process completed.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Input paths
    json_file_path = ""
    video_folder_path = ""
    destination_folder_path = ""

    # Run the moving process
    move_videos_with_youtube_urls(json_file_path, video_folder_path, destination_folder_path)
