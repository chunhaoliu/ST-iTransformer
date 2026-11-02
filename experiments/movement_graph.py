"""Directed movement-level graph from corridor adjacency and turning geometry.

Two observed intersections are corridor neighbours when they share a named
street and no other observed intersection on that street lies between them.
A movement token at an upstream intersection feeds the downstream approach
that its discharge reaches: for a downstream intersection located east of
the upstream one, EBT, NBR and SBL discharge eastbound and feed the
downstream EBL/EBT/EBR approach.  Directions are the nominal compass labels
used by the TMF channels.  Milton's arterial grid is rotated by roughly 45
degrees from true north: nominal east points to true north-east and nominal
north to true north-west.  A corridor whose true bearing (modulo 180) lies in
[0, 90) degrees is therefore a nominal east-west street, and the remaining
corridors are nominal north-south streets.  The orientation is then checked
on training rows only, using same-interval correlation of first differences; no
validation or test label is read.

This is a geometric corridor graph, not a verified GIS road network: links
through unobserved intersections and driveways are not represented.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

CHANNELS = ("EBL", "EBR", "EBT", "NBL", "NBR", "NBT", "SBL", "SBR", "SBT", "WBL", "WBR", "WBT")
# Movements that discharge toward each compass direction.
DISCHARGE = {
    "E": ("EBT", "NBR", "SBL"),
    "W": ("WBT", "SBR", "NBL"),
    "N": ("NBT", "WBR", "EBL"),
    "S": ("SBT", "EBR", "WBL"),
}
# The approach a vehicle travelling in that direction joins downstream.
APPROACH = {d: tuple(f"{d}B{m}" for m in "LRT") for d in "EWNS"}
_SUFFIX = re.compile(
    r"\s+(north|south|east|west|n|s|e|w)$", flags=re.IGNORECASE)


def street_key(name: str) -> str:
    """Normalise a street label: 'Main Street E' and 'Main Street' match."""
    text = re.sub(r"\s+", " ", name.strip())
    previous = None
    while previous != text:
        previous = text
        text = _SUFFIX.sub("", text)
    return text.lower()


def split_streets(name: str) -> list[str]:
    parts = re.split(r"\s+and\s+", name, flags=re.IGNORECASE)
    streets = []
    for part in parts:
        streets.extend(p for p in re.split(r"\s*/\s*", part) if p.strip())
    return [street_key(p) for p in streets]


def corridor_pairs(names, coords):
    """Consecutive observed intersections along each shared street."""
    streets = [split_streets(name) for name in names]
    by_street: dict[str, list[int]] = {}
    for index, keys in enumerate(streets):
        for key in set(keys):
            by_street.setdefault(key, []).append(index)
    pairs = []
    for key, members in sorted(by_street.items()):
        if len(members) < 2:
            continue
        points = coords[members]
        centred = points - points.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        order = np.argsort(centred @ vt[0])
        ordered = [members[i] for i in order]
        pairs.extend((key, a, b) for a, b in zip(ordered[:-1], ordered[1:]))
    return pairs


def _local_xy(coords):
    """Approximate local metres: x east, y north, adequate within one town."""
    lat0 = np.deg2rad(coords[:, 0].mean())
    y = (coords[:, 0] - coords[:, 0].mean()) * 111_320.0
    x = (coords[:, 1] - coords[:, 1].mean()) * 111_320.0 * np.cos(lat0)
    return np.stack((x, y), axis=1)


def _axis(xy, members):
    """'EW' when the corridor's true bearing modulo 180 lies in [0, 90)."""
    points = xy[members]
    centred = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    east, north = vt[0]
    bearing = np.degrees(np.arctan2(east, north)) % 180.0
    return "EW" if bearing < 90.0 else "NS"


def build_movement_graph(names, coords):
    """Return the [N*C, N*C] boolean graph, query rows attending key columns.

    Every token attends all twelve movements of its own intersection.  A
    downstream approach movement also attends the upstream movements whose
    discharge enters that approach.
    """
    names = [str(n) for n in names]
    coords = np.asarray(coords, dtype=np.float64)  # [N, 2] = (lat, lon)
    n_nodes, n_channels = len(names), len(CHANNELS)
    channel = {c: i for i, c in enumerate(CHANNELS)}
    graph = np.zeros((n_nodes * n_channels, n_nodes * n_channels), dtype=bool)
    for node in range(n_nodes):
        block = slice(node * n_channels, (node + 1) * n_channels)
        graph[block, block] = True
    links = []
    pairs = corridor_pairs(names, coords)
    corridors: dict[str, set[int]] = {}
    for key, a, b in pairs:
        corridors.setdefault(key, set()).update((a, b))
    xy = _local_xy(coords)
    axes = {key: _axis(xy, sorted(members)) for key, members in corridors.items()}
    for key, a, b in pairs:
        axis = axes[key]
        for up, down in ((a, b), (b, a)):
            dx, dy = xy[down] - xy[up]
            if axis == "EW":
                direction = "E" if dx + dy > 0 else "W"
            else:
                direction = "N" if dy - dx > 0 else "S"
            for src in DISCHARGE[direction]:
                for dst in APPROACH[direction]:
                    graph[down * n_channels + channel[dst], up * n_channels + channel[src]] = True
            links.append(dict(corridor=key, upstream=names[up], downstream=names[down],
                              direction=direction))
    return graph, links, axes


def orientation_check(train_raw, links, names, lag=0):
    """Training-only check on first differences, which remove the shared daily
    profile.  Adjacent signals are a few hundred metres apart, so a vehicle
    discharged toward a neighbour usually reaches it within the same
    five-minute interval: the neighbour's receiving approach should co-move
    with that discharge more than the approach facing the opposite way."""
    index = {n: i for i, n in enumerate(names)}
    channel = {c: i for i, c in enumerate(CHANNELS)}
    opposite = dict(E="W", W="E", N="S", S="N")

    def corr(u, v):
        u, v = u - u.mean(), v - v.mean()
        denominator = np.sqrt((u * u).sum() * (v * v).sum())
        return float((u * v).sum() / denominator) if denominator > 0 else float("nan")

    rows = []
    for link in links:
        up, down, d = index[link["upstream"]], index[link["downstream"]], link["direction"]
        source = np.diff(train_raw[:, up, [channel[c] for c in DISCHARGE[d]]].sum(axis=1))
        matched = np.diff(train_raw[:, down, [channel[c] for c in APPROACH[d]]].sum(axis=1))
        reverse = np.diff(train_raw[:, down, [channel[c] for c in APPROACH[opposite[d]]]].sum(axis=1))
        if lag:
            source, matched, reverse = source[:-lag], matched[lag:], reverse[lag:]
        rows.append(dict(link, matched_corr=corr(source, matched),
                         opposite_corr=corr(source, reverse)))
    valid = [r for r in rows if np.isfinite(r["matched_corr"]) and np.isfinite(r["opposite_corr"])]
    agreeing = sum(r["matched_corr"] > r["opposite_corr"] for r in valid)
    return dict(links=len(rows), testable=len(valid), agreeing=int(agreeing), rows=rows)


def coordinates_for(names, metadata_path):
    records = {r["name"]: r for r in json.loads(Path(metadata_path).read_text(encoding="utf-8"))}
    missing = [n for n in names if str(n) not in records]
    if missing:
        raise ValueError(f"Missing coordinates for {missing}")
    return np.asarray([(records[str(n)]["lat"], records[str(n)]["lon"]) for n in names])
