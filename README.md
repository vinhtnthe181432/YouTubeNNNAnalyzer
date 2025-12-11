# YouTubeNNNAnalyzer
Simple python program to detect current active YouTube video url and perform AI classification and detection to provide 3 level warning for NSFW content

- 0 -> Safe
- 1 -> Moderate Risk (Sexy, lewd content)
- 2 -> High Risk (nudity / pornographic content)

It works by extracting frames from the YouTube video (1 FPS if video length less than 300 seconds, and 300 frames max if longer), then applying:

1. [GantMan/nsfw_model](https://github.com/GantMan/nsfw_model) - general 5-class classifier
(drawings, hentai, neutral, porn, sexy)

2. [notAI-tech/NudeNet](https://github.com/notAI-tech/NudeNet ) - region-based nudity detector
(breasts exposed, genitalia exposed, armpits, feet, etc.)

Finally, results are aggregated into a final warning level.

# Directory Structure
```
YouTubeNNNAnalyzer/
├─ nsfw_model/              # Pretrained and modified MobileNetV2 GantMan/nsfw_model
├─ README.md
├─ main.py                  # Example usage with active browser URL
├─ nnn.py                   # YoutubeNNNAnalyzer class
├─ requirements.txt
└─ temp_frames/             # Temporary extraction directory (auto-cleaned)
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
