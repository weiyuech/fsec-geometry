"""Export the measured contour behind the browser explorer. Run from any directory."""
import hashlib
import json
from pathlib import Path
import sys

import contourpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fsa.fit import fit_four_quadrants, fit_quadrant, quadrant

EXAMPLE = ROOT / 'examples/Tense__French.npz'
OUTPUT = ROOT / 'docs/contour-data.js'
# Each tuple is in file order (tense, French); the public figure plots French horizontally.
QUADRANTS = {'topRight': (1, 1), 'topLeft': (1, -1), 'bottomLeft': (-1, -1), 'bottomRight': (-1, 1)}


def first_crossing(points, direction):
    """Intersect a positive ray from the origin with each segment; return the nearest intersection."""
    normal = np.array([-direction[1], direction[0]])
    side = points @ normal
    hits = []
    for i in range(len(points) - 1):
        if abs(side[i]) < 1e-12:
            hits.append(points[i] @ direction)
        if side[i] * side[i+1] < 0:
            u = side[i] / (side[i] - side[i+1])
            hit = points[i] + u * (points[i+1] - points[i])
            hits.append(hit @ direction)
    if abs(side[-1]) < 1e-12:
        hits.append(points[-1] @ direction)
    return min(r for r in hits if r > 0)


def display_points(tense_french, quadrant_fit, tense_sign, french_sign):
    """Map raw angular coordinates to signed, normalized x=French and y=tense."""
    raw_radius = np.linalg.norm(tense_french, axis=1)
    scale = np.pi / 180 * np.sinc(raw_radius / 180)
    x = french_sign * tense_french[:, 1] * scale / np.sin(np.deg2rad(quadrant_fit['t2']))
    y = tense_sign * tense_french[:, 0] * scale / np.sin(np.deg2rad(quadrant_fit['t1']))
    return np.column_stack((x, y))


def export():
    with np.load(EXAMPLE, allow_pickle=False) as data:
        angles = data['angles_deg']
        n_anchors = data['l2'].shape[0]
        grid = np.median(data['l2'], axis=0)
    fits = fit_four_quadrants(angles, grid)
    pp = fit_quadrant(*quadrant(angles, grid, 1, 1))
    result = dict(
        model='Gemma-2-9B', anchors=n_anchors, threshold=fits['thr'],
        source='examples/Tense__French.npz', sourceSHA256=hashlib.sha256(EXAMPLE.read_bytes()).hexdigest(),
        axes={
            'x': dict(name='French', positive='Toward French', negative='Opposite the French direction',
                      positiveInterceptDeg=fits['++']['t2'], negativeInterceptDeg=fits['+-']['t2']),
            'y': dict(name='Tense', positive='Toward future tense', negative='Toward past tense',
                      positiveInterceptDeg=fits['++']['t1'], negativeInterceptDeg=fits['-+']['t1'])
        },
        originalQuadrant=dict(name='topRight', p=pp['p'], threshold=pp['thr']),
        quadrants={}, raySamples=[]
    )
    arcs = {}
    raw_lines = {}
    for name, (tense, french) in QUADRANTS.items():
        signs = ''.join('+' if sign > 0 else '-' for sign in (tense, french))
        q = fits[signs]
        a, g = quadrant(angles, grid, tense, french)
        lines = contourpy.contour_generator(a, a, g.T).lines(fits['thr'])
        if len(lines) != 1:
            raise ValueError(f'Expected one connected contour in {name}, got {len(lines)}')
        raw_lines[name] = lines[0]
        # Each point is a threshold interpolation on the measured grid. Keep axis intercepts too.
        points = display_points(lines[0], q, tense, french)
        arcs[name] = points
        if not np.allclose(points[[0,-1]], [[french,0],[0,tense]], atol=1e-6, rtol=0):
            raise ValueError(f'Unexpected axis crossings in {name}')
        diagonal = np.array([french, tense]) / np.sqrt(2)
        radius = first_crossing(points, diagonal)
        result['quadrants'][name] = dict(fileSigns=signs, p=q['p'], delta=q['delta'],
                                          points=points.round(9).tolist(), diagonalRadius=radius,
                                          diagonalPoint=(radius*diagonal).tolist())
    # Validate the plotted polyline against an independent raw-angular-space first-crossing calculation.
    for deg in range(361):
        rad = np.deg2rad(deg)
        direction = np.array([np.cos(rad), np.sin(rad)])
        direction[np.abs(direction) < 1e-12] = 0
        french, tense = (1 if value >= 0 else -1 for value in direction)
        name = next(key for key, signs in QUADRANTS.items() if signs == (tense, french))
        radius = first_crossing(arcs[name], direction)
        q = fits[result['quadrants'][name]['fileSigns']]
        raw_ray = np.abs(direction[::-1]) * np.sin(np.deg2rad([q['t1'],q['t2']]))
        raw_ray /= np.linalg.norm(raw_ray)
        raw_crossing = raw_ray * first_crossing(raw_lines[name], raw_ray)
        independently_mapped = display_points(raw_crossing[None,:], q, tense, french)[0]
        if np.linalg.norm(radius * direction - independently_mapped) > 0.001:
            raise ValueError(f'Contour interpolation disagreement at {deg} degrees')
        result['raySamples'].append(dict(angleDeg=deg, radius=round(radius,9),
                                         x=round(radius*direction[0],9), y=round(radius*direction[1],9)))
    return result


if __name__ == '__main__':
    result = export()
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text('// Generated by scripts/export_explorer.py from the measured response grid.\n'
                      + 'window.FSEC_CONTOUR = ' + json.dumps(result, separators=(',', ':'), allow_nan=False)
                      + ';\n', encoding='utf-8')
