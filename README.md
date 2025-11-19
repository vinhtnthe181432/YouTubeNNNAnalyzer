# YouTubeNNNAnalyzer
Simple python program to detect current active YouTube video url and perform AI classification and detection to provide 3 level warning for NSFW content

- 0 -> Safe
- 1 -> Moderate Risk (Sexy, lewd content)
- 2 -> High Risk (nudity / pornographic content)

It works by extracting frames from the YouTube video (1 FPS), then applying:

1. GantMan/nsfw_model - general 5-class classifier
(drawings, hentai, neutral, porn, sexy)

2. notAI-tech/NudeNet - region-based nudity detector
(breasts exposed, genitalia exposed, armpits, feet, etc.)

Finally, results are aggregated into a final warning level.

# Directory Structure
```
YouTubeNNNAnalyzer/
│── nnn.py                   # YoutubeNNNAnalyzer class
│── main.py                  # Example usage with active browser URL
│── requirements.txt
│── README.md
│── nsfw_model/              # Pretrained and modified MobileNetV2 GantMan/nsfw_model
│── temp_frames/             # Temporary extraction directory (auto-cleaned)
```

# Installation
Just clone the repo, install requirements and go
```bash
git clone https://github.com/vinhtnthe181432/YouTubeNNNAnalyzer.git
cd YouTubeNNNAnalyzer
pip install -r requirements.txt
```

# Usage Example
Just ensure your active browser tab is a YouTube video, run main.py and wait for output of warning level, or ```YoutubeNNNAnalyzer()analyze_active_youtube_video(your_youtube_video_url)``` can be called to return warning level

```python
from nnn import YoutubeNNNAnalyzer

analyzer = YoutubeNNNAnalyzer()
warning = analyzer.analyze_active_youtube_video("https://youtu.be/dQw4w9WgXcQ")

print("Warning Level:", warning)
```
