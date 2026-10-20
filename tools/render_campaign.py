"""Export actual engine renderings of the replacement twelve-stage campaign."""
from pathlib import Path
import os
import sys
import tempfile
import time
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pygame
from pingu_ball.app import App
from pingu_ball.config import REGULAR_STAGE_COUNT


def save_preview(image, destination):
    # Readers always see a complete PNG, even if another app is open.
    temporary = destination.with_suffix('.render.png')
    pygame.image.save(image,str(temporary))
    for attempt in range(30):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == 29: raise
            time.sleep(.02)


def main():
    folder = ROOT / 'output/stage-import/runtime'
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        app = App(Path(tmp) / 'progress.json', headless=True)
        try:
            app.progress.unlocked = REGULAR_STAGE_COUNT
            app.show_normal_menu(); app.render()
            pygame.image.save(app.canvas, str(folder / 'menu.png'))
            for n in range(1, REGULAR_STAGE_COUNT + 1):
                app.start(n); app.render()
                pygame.image.save(app.canvas, str(folder / f'play_{n:02d}.png'))
                preview = pygame.transform.smoothscale(app.background, app.map_canvas.get_size())
                preview.blit(app.map_canvas, (0, 0))
                save_preview(preview, ROOT / app.world.level.preview)
                print(n, len(app.world.platforms), len(app.world.level.specials), flush=True)
        finally: pygame.quit()


if __name__ == '__main__': main()
