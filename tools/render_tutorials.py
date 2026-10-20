"""Render the current game maps, independently of the document's concept art."""
from pathlib import Path
import sys
import tempfile
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pygame
from pingu_ball.app import App


def main():
    folder=ROOT/'output/tutorial-validation';folder.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        app=App(Path(tmp)/'save.json',headless=True)
        try:
            app.show_lab_menu();app.render()
            pygame.image.save(app.canvas,str(folder/'menu.png'))
            for n in range(1,11):
                app.start_special(n);app.render()
                pygame.image.save(app.canvas,str(folder/f'stage_{n:02d}.png'))
        finally: pygame.quit()


if __name__=='__main__':main()
