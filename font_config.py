import os
import matplotlib
from matplotlib.font_manager import FontProperties

BASE_DIR = os.path.dirname(__file__)
FONT_DIR = os.path.join(BASE_DIR, "fonts")
FONT_PATH = os.path.join(FONT_DIR, "NotoSansDevanagari-Regular.ttf")

if not os.path.exists(FONT_PATH):
    FONT_PATH = matplotlib.get_data_path() + "/fonts/ttf/DejaVuSans.ttf"

try:
    from matplotlib import font_manager as fm
    fm.fontManager.addfont(FONT_PATH)
    emoji_font_prop = FontProperties(fname=FONT_PATH)
except Exception:
    emoji_font_prop = FontProperties()
