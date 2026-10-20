# Approved ice UI artwork

The four screen references were supplied and approved by the user. These
runtime PNGs are extracted by `tools/prepare_ui_assets.py`; the game needs
only Pygame, not Pillow. All hit regions use the 1600×832 logical canvas.

`tutorial_menu.png` and `regular_menu.png` are the two supplied stage-list
screens (1737×906), scaled once to the logical canvas. Their ten tutorial
cards, twelve regular cards, back buttons and sound buttons use matching
hit regions. Regular completion counts, unlocked play buttons and best
records come from the save file; mute labels follow the current audio state.

Three local patches were edited with the built-in imagegen tool, then cropped
to the affected UI regions so all other supplied artwork stays intact:
`menu_tutorial_count_removed.png`, `menu_regular_count_removed.png`, and
`regular_first_button_removed.png`. The edit prompts remove the two count
pills (10/12 stages) and the baked-in STAGE 1 play button, reconstructing the
icy panel and thumbnail behind them while preserving the rest of each screen.
All unlocked regular cards now draw the same cached play button at the same
relative card position and size, including STAGE 1.

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
