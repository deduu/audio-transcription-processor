# YouTube Audio Extractor and Transcriber using Whisper

This script allows you to:

- Specify multiple YouTube URLs
- Set time ranges for audio extraction from YouTube videos
- Automatically download the specified audio segments
- Transcribe the audio using OpenAI's Whisper model
- Generate a text file with the audio file paths and their transcriptions

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Output](#output)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Prerequisites

- Python 3.7 or higher
- pip package manager

## Installation

1. Clone the repository or download the script:

```bash
git clone https://github.com/deduu/audio-transcription-processor
cd audio-transcription-processor
```

2. Install required Python packages:

```bash
pip install yt-dlp openai-whisper
```

Note:
- `yt-dlp` is used for downloading audio from YouTube
- `openai-whisper` is used for transcribing audio files

3. Ensure FFmpeg is installed on your system:

**Windows:**
- Download FFmpeg from ffmpeg.org
- Add FFmpeg to your system's PATH environment variable

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg
```

## Usage

### 1. Configure URLs and Time Ranges

Edit the `main()` function in the script to specify the YouTube URLs and the time ranges you want to extract and transcribe.

Example configuration:

```python
def main():
    # User specifies the list of URLs and corresponding time ranges
    url_info_list: List[Dict[str, Any]] = [
        {
            'url': 'https://www.youtube.com/watch?v=VIDEO_ID_1',
            'time_ranges': [
                {'start_time': '00:00:10', 'end_time': '00:00:20', 'id': 1},
                {'start_time': '00:01:00', 'end_time': '00:01:10', 'id': 2},
                # Add more time ranges as needed
            ]
        },
        {
            'url': 'https://www.youtube.com/watch?v=VIDEO_ID_2',
            'time_ranges': [
                {'start_time': '00:02:00', 'end_time': '00:02:10', 'id': 3},
                {'start_time': '00:03:00', 'end_time': '00:03:10', 'id': 4},
                # Add more time ranges as needed
            ]
        },
        # Add more URLs and time ranges as needed
    ]
    # Rest of the code...
```

- Replace `'VIDEO_ID_1'` and `'VIDEO_ID_2'` with the actual YouTube video IDs
- Adjust the `start_time` and `end_time` to the desired segments
- The `id` field is used to name the output files uniquely

### 2. Run the Script

Execute the script using Python:

```bash
python your_script_name.py
```

Replace `your_script_name.py` with the actual name of the script file.

## Output

### Extracted Audio Files

- Saved in the `extracted_audio` directory
- Filenames are based on the provided `id` (e.g., `1.wav`, `2.wav`)

### Transcriptions

- Compiled into `transcriptions/list.txt`
- Each line has the format: `relative/path/to/audio.wav|transcription`

Example `list.txt` content:
```
../extracted_audio/1.wav|This is the transcribed text from audio file 1.
../extracted_audio/2.wav|This is the transcribed text from audio file 2.
```

## Customization

### Choose Whisper Model Size

The Whisper model comes in various sizes. Larger models provide better accuracy but require more computational resources.

Modify the line in `transcribe_audio_files` function:

```python
model = whisper.load_model("base")
```

Available models:
- `'tiny'`
- `'base'`
- `'small'`
- `'medium'`
- `'large'`

Model Characteristics:

| Model Size | Parameters | Relative Speed | Memory Requirement |
|-----------|------------|---------------|-------------------|
| tiny      | 39 M       | ~32x          | ~1 GB             |
| base      | 74 M       | ~16x          | ~1 GB             |
| small     | 244 M      | ~6x           | ~2 GB             |
| medium    | 769 M      | ~2x           | ~5 GB             |
| large     | 1550 M     | ~1x           | ~10 GB            |

Choose a model that fits your accuracy needs and hardware capabilities.

### Adjust Output Directories

By default, the script uses:
- `extracted_audio` for downloaded audio files
- `transcriptions` for the output text file

You can change these by modifying:

```python
audio_output_dir = "extracted_audio"
transcription_output_dir = "transcriptions"
```

Ensure the directories exist or let the script create them by including:

```python
os.makedirs(audio_output_dir, exist_ok=True)
os.makedirs(transcription_output_dir, exist_ok=True)
```

## Troubleshooting

### FFmpeg Not Found

- Ensure FFmpeg is installed and added to your system's PATH
- Verify by running `ffmpeg -version` in your terminal

### yt-dlp Errors

- Update yt-dlp to the latest version:
  ```bash
  pip install -U yt-dlp
  ```
- Check network connectivity and ensure YouTube is accessible

### Whisper Errors

- Ensure you have sufficient memory for the chosen model size
- For GPU acceleration, install PyTorch with CUDA support:
  ```bash
  pip install torch --extra-index-url https://download.pytorch.org/whl/cu117
  ```

### Permission Issues

- Run the script with appropriate permissions
- Check read/write permissions for the output directories

## License

This project is licensed under the Apache License Version 2.0 License. See the LICENSE file for details.

**Disclaimer:** This script is intended for educational and personal use. Respect YouTube's Terms of Service and copyright laws when downloading and using content.