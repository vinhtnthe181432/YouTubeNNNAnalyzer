from nnn import YoutubeNNNAnalyzer
import uiautomation as auto
from urllib.parse import urlparse, parse_qs

# Detect active URL of foreground browser
def get_active_url():
    try:
        browser_window = auto.WindowControl(searchDepth=1, ClassName='Chrome_WidgetWin_1')
        address_bar = browser_window.EditControl()
        url = address_bar.GetValuePattern().Value

        if '://' not in url:
            url = 'http://' + url

        return {
            'domain': urlparse(url).netloc,
            'url': url,
            'title': browser_window.Name
        }
    except Exception:
        return None

active_url = get_active_url()

if active_url:
    url = active_url['url']
    domain = active_url['domain']

    if any(d in domain for d in ['youtube.com', 'youtu.be']):
        analyzer = YoutubeNNNAnalyzer()
        warning_level = analyzer.analyze_active_youtube_video(url)
        
        if warning_level == 2:
            print('High Warning Video')
        elif warning_level == 1:
            print('Moderate Warning Video')
        else:
            print('Normal Video')