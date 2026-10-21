# Approved ice UI artwork

The four screen references were supplied and approved by the user. These
runtime PNGs are extracted by `tools/prepare_ui_assets.py`; the game needs
only Pygame, not Pillow. All hit regions use the 1600×832 logical canvas.

`hud_stage.png` and `hud_timer.png` use the approved seamless ice design.
Built-in imagegen edit prompts: remove the flat inner text patches, restore
continuous illuminated faceted ice, and preserve the snowy frame, flag or
stopwatch, empty live-text areas and transparent exterior. The approved
comparison is `output/hud-natural-preview/comparison.png` (local review output).
`regular_counter_blank.png` similarly removes the baked completion count from
the original list panel while preserving its rim and subtitle. Live stage,
timer and count lettering is antialiased and centered by visible glyph bounds;
no solid rectangles are drawn beneath it. `prepare_ui_assets.py` preserves
these separately maintained HUD panels when preparing the other controls.

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
Regular card clicks compress and darken the play button, then ease it back
over 140 ms before starting the selected stage. Repeated clicks during that
effect are ignored; leaving the menu or losing focus cancels the pending start.

`menu_sound_face.png` is cropped from a built-in imagegen edit of the
supplied mode-selection screen. Prompt: remove the painted sound wording or
flat label patch, restore a seamless glossy blue ice button face, and preserve
the snowy outer frame, speaker icon, position and all other artwork. Runtime
text is drawn directly on that face in both states, with no inset rectangle;
the cached Korean bold lettering has a thin antialiased navy edge.
The lettering is cropped to its visible alpha bounds before placement, so
font ascent/descent whitespace cannot shift either state below center.
`title_sound_face.png` is a transparent cutout of the same snowy blue button,
edited with built-in imagegen: remove only exterior scenery while preserving
the frame, speaker icon, proportions and blank face. The title now uses this
ice artwork and the shared sound labels instead of a plain white UI button.
Both stage lists reuse that exact transparent sprite at 240×78 logical pixels,
with the same bottom-right position, text field and hit region. Their formerly
different sound artwork is cleared with `tutorial_menu_sound_cleared.png` and
`regular_menu_sound_cleared.png`, cropped from built-in imagegen edits. Prompt:
remove only the entire old sound control and restore the snow/lake behind it,
preserving all cards, characters, lettering and other scenery.

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
