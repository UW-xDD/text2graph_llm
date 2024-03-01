from dataclasses import dataclass


@dataclass
class Point:
    lat: float
    lon: float

    def __post_init__(self):
        if not -180 <= self.lon <= 180:
            raise ValueError("{self.lon} is not a valid lon")
        if not -90 <= self.lat <= 90:
            raise ValueError("{self.lat} is not a valid lat")
