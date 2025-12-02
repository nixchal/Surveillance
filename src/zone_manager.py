"""Virtual zone management for the surveillance system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple


Point = Tuple[int, int]


@dataclass(slots=True)
class Zone:
    name: str
    points: List[Point]
    zone_type: str = "restricted"
    enabled: bool = True


class ZoneManager:
    def __init__(self) -> None:
        self._zones: Dict[str, Zone] = {}

    def add_zone(self, zone: Zone) -> None:
        self._zones[zone.name] = zone

    def remove_zone(self, zone_name: str) -> None:
        self._zones.pop(zone_name, None)

    def toggle_zone(self, zone_name: str, enabled: bool) -> None:
        if zone_name in self._zones:
            self._zones[zone_name].enabled = enabled

    def zones(self) -> Iterable[Zone]:
        return list(self._zones.values())

    def active_zones(self) -> List[Zone]:
        return [zone for zone in self._zones.values() if zone.enabled]


__all__ = ["Zone", "ZoneManager", "Point"]

