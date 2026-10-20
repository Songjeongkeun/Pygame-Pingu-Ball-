# Approved ice UI artwork

The four screen references were supplied and approved by the user. These
runtime PNGs are extracted by `tools/prepare_ui_assets.py`; the game needs
only Pygame, not Pillow. All hit regions use the 1600×832 logical canvas.

`pause.png` uses the approved transparent image with the bottom
“ESC 또는 P로 돌아가기” line removed using the built-in imagegen editor.
Edit request: remove that line and both short dashes, fill with the surrounding
icy texture, and preserve all other lettering, buttons, penguin and transparency.

Stage title, timer, fish counters, grade, score, record status and exceptional
next-button labels are rendered from live state. The supplied gameplay
screenshot's map and footer are not part of the runtime artwork.

The current view reserves y=144…816 for the map. `expand_playfield` gives the
world the same width as this view, shifting terrain, devices (including patrol
paths), collectibles, spawn and goal together. The added side areas are air.
Source document coordinates and native preview images are retained.
