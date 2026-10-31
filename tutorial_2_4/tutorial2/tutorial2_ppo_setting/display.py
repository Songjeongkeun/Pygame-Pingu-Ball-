class LiveDisplay:
    def __init__(self, env, fps=0):
        import pygame
        from pingu_ball.app import App
        self.pg = pygame
        self.app = App()
        self.app.start_special(2)
        self.app.world = env.world
        self.app.scene = 'play'
        self.app.sound_on = False
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 22)
        self.fps = max(0, fps)
        self.stop = False
        self.paused = False

    def events(self):
        p = self.pg
        for evt in p.event.get():
            if evt.type == p.QUIT or evt.type == p.KEYDOWN and evt.key == p.K_ESCAPE:
                self.stop = True
            elif evt.type == p.KEYDOWN and evt.key == p.K_SPACE:
                self.paused = not self.paused
        while self.paused and not self.stop:
            p.time.wait(35)
            for evt in p.event.get():
                if evt.type == p.QUIT or (evt.type == p.KEYDOWN and evt.key == p.K_ESCAPE):
                    self.stop = True
                    self.paused = False
                elif evt.type == p.KEYDOWN and evt.key == p.K_SPACE:
                    self.paused = False
        return not self.stop

    def draw(self, algo, episode, step, action, reward, info):
        from .environment import ACTIONS
        self.app.render()
        p = self.pg
        direction, frames = ACTIONS[action]
        panel = p.Surface((710, 76), p.SRCALPHA)
        panel.fill((8, 21, 35, 225))
        label = {-1:'LEFT', 0:'WAIT', 1:'RIGHT'}[direction]
        lines = (
            f'{algo.upper()} | tutorial 2 | ep={episode} | decisions={step} | {label} {frames}f',
            f'reward={reward:+.2f} total={info["reward_total"]:+.2f} spring={info["spring1"]}/{info["spring2"]}/{info["spring3"]} decks={info["milestone"]}/3',
            'SPACE=pause/resume  ESC=save & stop',
        )
        for i, line in enumerate(lines):
            panel.blit(self.font.render(line, True, (235, 250, 255)), (10, 5+i*23))
        self.app.window.blit(panel, (8, self.app.window.get_height()-86))
        p.display.flip()
        if self.fps:
            self.clock.tick(self.fps)

    def close(self):
        self.pg.quit()
