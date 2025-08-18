import os
import matplotlib
from matplotlib.font_manager import FontProperties

BASE_DIR = os.path.dirname(__file__)
FONT_DIR = os.path.join(BASE_DIR, "fonts")

# Paths to fonts
DEVANAGARI_FONT = os.path.join(FONT_DIR, "NotoSansDevanagari-Regular.ttf")
EMOJI_FONT = os.path.join(FONT_DIR, "NotoEmoji-Regular.ttf")  # or Symbola.ttf

# ✅ Fallback if fonts are missing
DEFAULT_FONT = os.path.join(matplotlib.get_data_path(), "fonts/ttf/DejaVuSans.ttf")

# ---- Devanagari text font ----
if os.path.exists(DEVANAGARI_FONT):
    from matplotlib import font_manager as fm
    fm.fontManager.addfont(DEVANAGARI_FONT)
    devanagari_font_prop = FontProperties(fname=DEVANAGARI_FONT)
else:
    devanagari_font_prop = FontProperties(fname=DEFAULT_FONT)

# ---- Emoji font ----
if os.path.exists(EMOJI_FONT):
    from matplotlib import font_manager as fm
    fm.fontManager.addfont(EMOJI_FONT)
    emoji_font_prop = FontProperties(fname=EMOJI_FONT)
else:
    emoji_font_prop = FontProperties(fname=DEFAULT_FONT)
