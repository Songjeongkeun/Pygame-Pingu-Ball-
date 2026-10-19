from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Physics:
    width: int = 1600
    height: int = 832
    tile: int = 64
    radius: float = 14.0
    bounce_height: float = 96.0
    pad_height: float = 144.0
    flight_time: float = 0.9
    horizontal_range: float = 192.0
    acceleration: float = 9600.0
    deceleration: float = 14400.0
    hz: int = 120
    break_delay: float = 0.5

    @property
    def gravity(self):
        return 8 * self.bounce_height / self.flight_time**2

    @property
    def bounce_speed(self):
        return math.sqrt(2 * self.gravity * self.bounce_height)

    @property
    def pad_speed(self):
        return math.sqrt(2 * self.gravity * self.pad_height)

    @property
    def max_speed(self):
        return self.horizontal_range / self.flight_time

    @property
    def platform_speed(self):
        return self.max_speed / 3 * 3

    @property
    def launch_speed(self):
        return self.max_speed * 3

    @property
    def dt(self):
        return 1 / self.hz


PHYSICS = Physics()
