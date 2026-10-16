from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Physics:
    width: int = 1200
    height: int = 600
    tile: float = 20.0
    radius: float = 4.5
    bounce_height: float = 60.0
    pad_height: float = 90.0
    flight_time: float = 0.8
    horizontal_range: float = 240.0
    acceleration: float = 3000.0
    deceleration: float = 4500.0
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
        return self.max_speed / 3

    @property
    def dt(self):
        return 1 / self.hz


PHYSICS = Physics()
