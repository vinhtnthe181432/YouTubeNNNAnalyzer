import os
import shutil
import re
import cv2
import yt_dlp
from urllib.parse import urlparse, parse_qs

from nsfw_model.nsfw_detector import predict
from nudenet import NudeDetector


class YoutubeNNNAnalyzer:
    '''
    Analyze the currently active YouTube tab and classify the content into three level of warning:
        - 0 = no warning
        - 1 = moderate warning
        - 2 = high warning
    '''

    def __init__(self, model_path='YouTubeNNNAnalyzer/nsfw_model/mobilenet_v2_140_224/saved_model.h5'):
        self.nsfw_model = predict.load_model(model_path)
        self.nude_detector = NudeDetector()

        self.high_warning_labels = {
            'FEMALE_GENITALIA_EXPOSED',
            'MALE_GENITALIA_EXPOSED',
            'ANUS_EXPOSED',
            'FEMALE_BREAST_EXPOSED',
            'BUTTOCKS_EXPOSED'
        }

        self.moderate_warning_labels = {
            'FEMALE_GENITALIA_COVERED',
            'FEET_EXPOSED',
            'ARMPITS_EXPOSED',
            'BELLY_EXPOSED',
            'FEMALE_BREAST_COVERED',
            'BUTTOCKS_COVERED'
        }

    # Step 1: Download frames from YouTube video (1 FPS)
    @staticmethod
    def extract_frames(video_id, output_dir, target_fps=1):
        youtube_url = f'https://youtube.com/watch?v={video_id}'

        ydl_opts = {
            'url': youtube_url,
            'format': 'bestvideo[ext=mp4][height<360]/bestvideo[ext=mp4]/bestvideo',
            'quiet': True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                video_url = info.get('url', info.get('formats', [{}])[0].get('url'))
        except Exception:
            return None

        if not video_url:
            return None

        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            return None

        os.makedirs(output_dir, exist_ok=True)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_step = int(fps / target_fps)

        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % frame_step == 0:
                frame_path = os.path.join(output_dir, f'frame_{count}.jpg')
                cv2.imwrite(frame_path, frame)

            count += 1

        cap.release()
        return output_dir

    # Step 2: GantMan/nsfw_model general classification (drawings/hentai/neutral/porn/sexy)
    def classify_frames(self, directory):
        result = predict.classify(self.nsfw_model, directory)

        for frame_path, info in result.items():
            frame_class = max(info, key=info.get)

            # Keep hentai/porn/sexy frames for further analyzation
            if frame_class not in ['hentai', 'porn', 'sexy']:
                os.remove(frame_path)
                continue

        return result

    # Step 3: notAI-tech/NudeNet area detection
    def detect_areas(self, directory):
        frames = [os.path.join(directory, f) for f in os.listdir(directory)]
        return self.nude_detector.detect_batch(frames)

    # Step 4: Handle warning logic -> output 0/1/2
    def compute_warning_level(self, nsfw_result, region_result):
        logged_frames = len(nsfw_result)
        if logged_frames == 0:
            return 0

        moderate_count = 0
        high_count = 0

        for detection in region_result:
            labels = {d['class'] for d in detection}

            if labels & self.high_warning_labels:
                high_count += 1
            elif labels & self.moderate_warning_labels:
                moderate_count += 1

        # Ratio threshold
        if high_count >= logged_frames * 0.01:
            return 2
        if moderate_count >= logged_frames * 0.01:
            return 1

        return 0

    # Helper function to robustly extract youtube video ID
    def extract_youtube_video_id(self, url: str):
        if not url:
            return None

        parsed = urlparse(url)
        domain = parsed.netloc

        # youtube.com/watch?v={ID}
        if 'youtube.com' in domain:
            query = parse_qs(parsed.query)
            if 'v' in query:
                return query['v'][0]

            # youtube.com/shorts/{ID}
            shorts_match = re.match(r'^/shorts/([^/?]+)', parsed.path)
            if shorts_match:
                return shorts_match.group(1)

        # youtu.be/{ID}
        if 'youtu.be' in domain:
            match = re.match(r'^/([^/?]+)', parsed.path)
            if match:
                return match.group(1)

        return None

    # Analyze currently active YouTube video
    def analyze_active_youtube_video(self,  youtube_url: str):
        video_id =self.extract_youtube_video_id(youtube_url)
        if not video_id:
            return 0

        output_dir = f'YouTubeNNNAnalyzer/temp_frames/{video_id}'

        # Step 1: extract frames
        out = self.extract_frames(video_id, output_dir, target_fps=1)
        if not out:
            return 0

        # Step 2: General classification
        nsfw_result = self.classify_frames(output_dir)

        # Step 3: Areas detection
        region_result = self.detect_areas(output_dir)

        # Step 4: Compute warning level
        level = self.compute_warning_level(nsfw_result, region_result)

        # Cleanup
        shutil.rmtree(output_dir)

        return level