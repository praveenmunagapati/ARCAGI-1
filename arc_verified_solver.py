"""
MATHX ARC-AGI-1 High-Capacity Verified Solver & Benchmark Engine
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional, Any
import numpy as np

Grid = np.ndarray

def G(x) -> Grid:
    return np.asarray(x, dtype=np.int16)

def exact(a: Optional[Grid], b: Optional[Grid]) -> bool:
    if a is None or b is None:
        return False
    return a.shape == b.shape and np.array_equal(a, b)

# ============================================================
# OBJECT UTILITIES
# ============================================================

@dataclass
class Obj:
    color: int
    cells: list[tuple[int, int]]
    area: int
    bbox: tuple[int, int, int, int] # (min_r, min_c, max_r, max_c)
    h: int
    w: int
    subgrid: Grid

def extract_objects(g: Grid, connectivity: int = 4, bg: int = 0, mono: bool = True) -> list[Obj]:
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    objs = []
    dirs = [(-1,0),(1,0),(0,-1),(0,1)] if connectivity == 4 else [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    for r in range(h):
        for c in range(w):
            if visited[r, c] or g[r, c] == bg:
                continue
            color = int(g[r, c])
            cells = []
            stack = [(r, c)]
            visited[r, c] = True
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                        if mono:
                            if g[nr, nc] == color:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                        else:
                            if g[nr, nc] != bg:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
            rs = [x[0] for x in cells]
            cs = [x[1] for x in cells]
            min_r, max_r = min(rs), max(rs)
            min_c, max_c = min(cs), max(cs)
            sub = g[min_r:max_r+1, min_c:max_c+1].copy()
            objs.append(Obj(
                color=color,
                cells=cells,
                area=len(cells),
                bbox=(min_r, min_c, max_r, max_c),
                h=max_r - min_r + 1,
                w=max_c - min_c + 1,
                subgrid=sub
            ))
    return objs

# ============================================================
# SOLVER PRIMITIVES & DEDUCTIVE INFERENCE
# ============================================================

class SolverEngine:
    def __init__(self):
        pass

    def synthesize(self, task: dict) -> Optional[Callable[[Grid], Grid]]:
        train = [(G(ex["input"]), G(ex["output"])) for ex in task["train"]]
        
        # 1. Direct Rigid & Affine
        sol = self._try_rigid(train)
        if sol: return sol

        # 2. Direct Palette / Color Permutations
        sol = self._try_palette(train)
        if sol: return sol

        # 3. Kronecker & Fractal Tiling
        sol = self._try_kronecker(train)
        if sol: return sol

        # 4. Scaling / Zoom / Pixel Repeat
        sol = self._try_scaling(train)
        if sol: return sol

        # 5. Cropping & Bounding Box Extraction
        sol = self._try_crop(train)
        if sol: return sol

        # 6. Multi-Panel & Grid Divider Logic
        sol = self._try_divider_panels(train)
        if sol: return sol

        # 7. Symmetry & Pattern Completion
        sol = self._try_symmetry(train)
        if sol: return sol

        # 8. Enclosed Hole Filling & Morphology
        sol = self._try_hole_filling(train)
        if sol: return sol

        # 9. Gravity & Physics (free fall & obstacle collision)
        sol = self._try_gravity(train)
        if sol: return sol

        # 10. Lines, Raycasting & Orthogonal Crosses
        sol = self._try_lines_and_rays(train)
        if sol: return sol

        # 11. Object Extraction & Filtering
        sol = self._try_object_filtering(train)
        if sol: return sol

        # 12. Object Recoloring by Area Rank / Sequence
        sol = self._try_object_recolor(train)
        if sol: return sol

        # 13. Fill Bounding Boxes of Connected Components
        sol = self._try_bbox_fill(train)
        if sol: return sol

        # 14. Periodic Pattern Extension / Extrapolation
        sol = self._try_periodic_extension(train)
        if sol: return sol

        # 15. Tiling / Pattern Repetition
        sol = self._try_tiling(train)
        if sol: return sol

        # 16. 2-Step Composition (Crop -> Palette, Rigid -> Palette, Crop -> Rotate, etc.)
        sol = self._try_composition(train)
        if sol: return sol

        return None

    # --- 1. Rigid Transformations ---
    def _try_rigid(self, train):
        ops = [
            ("identity", lambda g: g.copy()),
            ("rot90", lambda g: np.rot90(g, -1)),
            ("rot180", lambda g: np.rot90(g, 2)),
            ("rot270", lambda g: np.rot90(g, 1)),
            ("flip_h", lambda g: np.fliplr(g)),
            ("flip_v", lambda g: np.flipud(g)),
            ("transpose", lambda g: g.T),
            ("anti_transpose", lambda g: np.flipud(np.fliplr(g.T))),
        ]
        for name, fn in ops:
            try:
                if all(exact(fn(inp), out) for inp, out in train):
                    return fn
            except Exception:
                pass
        return None

    # --- 2. Palette Mapping ---
    def _try_palette(self, train):
        mapping = {}
        consistent = True
        for inp, out in train:
            if inp.shape != out.shape:
                consistent = False; break
            for u in np.unique(inp):
                out_colors = out[inp == u]
                if len(np.unique(out_colors)) != 1:
                    consistent = False; break
                c_target = int(out_colors[0])
                if u in mapping and mapping[u] != c_target:
                    consistent = False; break
                mapping[u] = c_target
            if not consistent:
                break
        if consistent and mapping:
            def fn(g):
                out = g.copy()
                for k, v in mapping.items():
                    out[g == k] = v
                return out
            if all(exact(fn(inp), out) for inp, out in train):
                return fn
        return None

    # --- 3. Kronecker & Fractal ---
    def _try_kronecker(self, train):
        def fn1(g):
            mask = (g > 0).astype(np.int16)
            return np.kron(mask, g)
        if all(exact(fn1(inp), out) for inp, out in train):
            return fn1

        def fn2(g):
            mask = (g > 0).astype(np.int16)
            return np.kron(g, mask)
        if all(exact(fn2(inp), out) for inp, out in train):
            return fn2

        def fn3(g):
            h, w = g.shape
            out = np.zeros((h*h, w*w), dtype=np.int16)
            for r in range(h):
                for c in range(w):
                    if g[r, c] != 0:
                        out[r*h:(r+1)*h, c*w:(c+1)*w] = g
            return out
        try:
            if all(exact(fn3(inp), out) for inp, out in train):
                return fn3
        except Exception:
            pass
        return None

    # --- 4. Scaling / Zoom ---
    def _try_scaling(self, train):
        for sy in (2, 3, 4, 5):
            for sx in (2, 3, 4, 5):
                def make_scale(y_fac=sy, x_fac=sx):
                    return lambda g: np.repeat(np.repeat(g, y_fac, axis=0), x_fac, axis=1)
                fn = make_scale()
                try:
                    if all(exact(fn(inp), out) for inp, out in train):
                        return fn
                except Exception:
                    pass
        return None

    # --- 5. Cropping & Bounding Box ---
    def _try_crop(self, train):
        # Crop non-zero
        def crop_non_zero(g):
            rows, cols = np.where(g != 0)
            if len(rows) == 0: return g
            return g[rows.min():rows.max()+1, cols.min():cols.max()+1]
        try:
            if all(exact(crop_non_zero(inp), out) for inp, out in train):
                return crop_non_zero
        except Exception:
            pass

        # Crop to most frequent non-zero color bbox
        def crop_most_freq(g):
            counts = [(c, np.count_nonzero(g == c)) for c in np.unique(g) if c != 0]
            if not counts: return g
            counts.sort(key=lambda x: x[1], reverse=True)
            c_target = counts[0][0]
            rows, cols = np.where(g == c_target)
            return g[rows.min():rows.max()+1, cols.min():cols.max()+1]
        try:
            if all(exact(crop_most_freq(inp), out) for inp, out in train):
                return crop_most_freq
        except Exception:
            pass

        # Crop inside frame
        for frame_c in range(10):
            def make_frame_crop(fc=frame_c):
                def fn(g):
                    rows, cols = np.where(g == fc)
                    if len(rows) == 0: return g
                    min_r, max_r = rows.min(), rows.max()
                    min_c, max_c = cols.min(), cols.max()
                    if max_r - min_r > 1 and max_c - min_c > 1:
                        return g[min_r+1:max_r, min_c+1:max_c]
                    return g
                return fn
            fn = make_frame_crop()
            try:
                if all(exact(fn(inp), out) for inp, out in train):
                    return fn
            except Exception:
                pass
        return None

    # --- 6. Divider Panels & Multi-panel Logic ---
    def _try_divider_panels(self, train):
        for div_c in range(10):
            def split_panels(g, dc=div_c):
                h, w = g.shape
                div_rows = [r for r in range(h) if np.all(g[r, :] == dc)]
                div_cols = [c for c in range(w) if np.all(g[:, c] == dc)]
                
                row_splits = [-1] + div_rows + [h]
                col_splits = [-1] + div_cols + [w]
                
                panels = []
                for i in range(len(row_splits) - 1):
                    r1, r2 = row_splits[i] + 1, row_splits[i+1]
                    for j in range(len(col_splits) - 1):
                        c1, c2 = col_splits[j] + 1, col_splits[j+1]
                        if r2 > r1 and c2 > c1:
                            panels.append(g[r1:r2, c1:c2])
                return panels

            # 1. Overlay
            for op in ("xor", "or", "and", "diff"):
                for recolor in range(10):
                    def make_overlay(dc=div_c, operation=op, rc=recolor):
                        def fn(g):
                            panels = split_panels(g, dc)
                            if len(panels) != 2 or panels[0].shape != panels[1].shape:
                                return None
                            p1, p2 = panels[0], panels[1]
                            res = np.zeros_like(p1)
                            if operation == "xor":
                                mask = (p1 != 0) ^ (p2 != 0)
                            elif operation == "or":
                                mask = (p1 != 0) | (p2 != 0)
                            elif operation == "and":
                                mask = (p1 != 0) & (p2 != 0)
                            elif operation == "diff":
                                mask = (p1 != 0) & (p2 == 0)
                            res[mask] = rc if rc != 0 else (p1[mask] if np.any(p1[mask]) else p2[mask])
                            return res
                        return fn
                    fn = make_overlay()
                    try:
                        if all(exact(fn(inp), out) for inp, out in train):
                            return fn
                    except Exception:
                        pass

            # 2. Select panel
            for idx_p in (0, 1, 2, 3, 4, -1):
                def make_panel_idx(dc=div_c, ip=idx_p):
                    def fn(g):
                        panels = split_panels(g, dc)
                        if not panels or abs(ip) >= len(panels): return None
                        return panels[ip]
                    return fn
                fn = make_panel_idx()
                try:
                    if all(exact(fn(inp), out) for inp, out in train):
                        return fn
                except Exception:
                    pass

            for sel in ("max_nonzero", "min_nonzero", "unique"):
                def make_sel(dc=div_c, s=sel):
                    def fn(g):
                        panels = split_panels(g, dc)
                        if not panels: return None
                        if s == "max_nonzero":
                            return max(panels, key=lambda p: np.count_nonzero(p))
                        elif s == "min_nonzero":
                            return min(panels, key=lambda p: np.count_nonzero(p))
                        return panels[0]
                    return fn
                fn = make_sel()
                try:
                    if all(exact(fn(inp), out) for inp, out in train):
                        return fn
                except Exception:
                    pass
        return None

    # --- 7. Symmetry & Pattern Completion ---
    def _try_symmetry(self, train):
        def sym_h_left(g):
            h, w = g.shape
            mid = w // 2
            out = g.copy()
            left = g[:, :mid]
            out[:, w - mid:] = np.fliplr(left)
            return out
        def sym_h_right(g):
            h, w = g.shape
            mid = w // 2
            out = g.copy()
            right = g[:, w - mid:]
            out[:, :mid] = np.fliplr(right)
            return out
        def sym_v_top(g):
            h, w = g.shape
            mid = h // 2
            out = g.copy()
            top = g[:mid, :]
            out[h - mid:, :] = np.flipud(top)
            return out
        def sym_v_bottom(g):
            h, w = g.shape
            mid = h // 2
            out = g.copy()
            bottom = g[h - mid:, :]
            out[:mid, :] = np.flipud(bottom)
            return out

        for fn in (sym_h_left, sym_h_right, sym_v_top, sym_v_bottom):
            try:
                if all(exact(fn(inp), out) for inp, out in train):
                    return fn
            except Exception:
                pass
        return None

    # --- 8. Enclosed Hole Filling ---
    def _try_hole_filling(self, train):
        for fill_c in range(10):
            def make_fn(fc):
                def fn(g):
                    h, w = g.shape
                    out = g.copy()
                    visited = np.zeros((h, w), dtype=bool)
                    stack = []
                    for r in range(h):
                        for c in (0, w-1):
                            if g[r, c] == 0 and not visited[r, c]:
                                visited[r, c] = True
                                stack.append((r, c))
                    for c in range(w):
                        for r in (0, h-1):
                            if g[r, c] == 0 and not visited[r, c]:
                                visited[r, c] = True
                                stack.append((r, c))
                    while stack:
                        r, c = stack.pop()
                        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if g[nr, nc] == 0 and not visited[nr, nc]:
                                    visited[nr, nc] = True
                                    stack.append((nr, nc))
                    for r in range(h):
                        for c in range(w):
                            if g[r, c] == 0 and not visited[r, c]:
                                out[r, c] = fc
                    return out
                return fn
            fn = make_fn(fill_c)
            try:
                if all(exact(fn(inp), out) for inp, out in train):
                    return fn
            except Exception:
                pass
        return None

    # --- 9. Gravity & Physics ---
    def _try_gravity(self, train):
        # 1. Pixel fall
        for direction in ("down", "up", "left", "right"):
            def make_grav(d=direction):
                def fn(g):
                    h, w = g.shape
                    out = np.zeros_like(g)
                    if d == "down":
                        for c in range(w):
                            col = g[:, c]
                            nonzeros = col[col != 0]
                            out[h - len(nonzeros):, c] = nonzeros
                    elif d == "up":
                        for c in range(w):
                            col = g[:, c]
                            nonzeros = col[col != 0]
                            out[:len(nonzeros), c] = nonzeros
                    elif d == "right":
                        for r in range(h):
                            row = g[r, :]
                            nonzeros = row[row != 0]
                            out[r, w - len(nonzeros):] = nonzeros
                    elif d == "left":
                        for r in range(h):
                            row = g[r, :]
                            nonzeros = row[row != 0]
                            out[r, :len(nonzeros)] = nonzeros
                    return out
                return fn
            fn = make_grav()
            try:
                if all(exact(fn(inp), out) for inp, out in train):
                    return fn
            except Exception:
                pass

        # 2. Move non-barrier objects down to obstacle barrier color
        for bar_c in range(1, 10):
            def make_bar_grav(bc=bar_c):
                def fn(g):
                    h, w = g.shape
                    out = np.zeros_like(g)
                    out[g == bc] = bc
                    # find barrier rows per col
                    for c in range(w):
                        bar_rows = np.where(g[:, c] == bc)[0]
                        non_bar = [r for r in range(h) if g[r, c] != 0 and g[r, c] != bc]
                        if len(bar_rows) > 0 and len(non_bar) > 0:
                            top_bar = bar_rows[0]
                            for i, orig_r in enumerate(reversed(non_bar)):
                                target_r = top_bar - 1 - i
                                if 0 <= target_r < h:
                                    out[target_r, c] = g[orig_r, c]
                        else:
                            for r in non_bar:
                                out[r, c] = g[r, c]
                    return out
                return fn
            fn = make_bar_grav()
            try:
                if all(exact(fn(inp), out) for inp, out in train):
                    return fn
            except Exception:
                pass
        return None

    # --- 10. Lines, Raycasting & Orthogonal Crosses ---
    def _try_lines_and_rays(self, train):
        # 1. Connect matching colored dots
        def connect_dots(g):
            h, w = g.shape
            out = g.copy()
            for color in np.unique(g):
                if color == 0: continue
                rows, cols = np.where(g == color)
                pts = list(zip(rows, cols))
                for i in range(len(pts)):
                    for j in range(i+1, len(pts)):
                        r1, c1 = pts[i]
                        r2, c2 = pts[j]
                        if r1 == r2:
                            out[r1, min(c1, c2):max(c1, c2)+1] = color
                        elif c1 == c2:
                            out[min(r1, r2):max(r1, r2)+1, c1] = color
            return out
        try:
            if all(exact(connect_dots(inp), out) for inp, out in train):
                return connect_dots
        except Exception:
            pass

        # 2. Draw horizontal & vertical cross through dots
        for cross_c in range(10):
            def make_cross(cc=cross_c):
                def fn(g):
                    h, w = g.shape
                    out = g.copy()
                    for r in range(h):
                        for c in range(w):
                            if g[r, c] != 0:
                                col = cc if cc != 0 else g[r, c]
                                out[r, :] = np.where(out[r, :] == 0, col, out[r, :])
                                out[:, c] = np.where(out[:, c] == 0, col, out[:, c])
                    return out
                return fn
            fn = make_cross()
            try:
                if all(exact(fn(inp), out) for inp, out in train):
                    return fn
            except Exception:
                pass

        # 3. Extend diagonals from dots
        def diag_extend(g):
            h, w = g.shape
            out = g.copy()
            for r in range(h):
                for c in range(w):
                    if g[r, c] != 0:
                        col = g[r, c]
                        for dr, dc in ((-1,-1),(-1,1),(1,-1),(1,1)):
                            cr, cc = r + dr, c + dc
                            while 0 <= cr < h and 0 <= cc < w:
                                if out[cr, cc] == 0:
                                    out[cr, cc] = col
                                cr += dr
                                cc += dc
            return out
        try:
            if all(exact(diag_extend(inp), out) for inp, out in train):
                return diag_extend
        except Exception:
            pass

        return None

    # --- 11. Object Extraction & Filtering ---
    def _try_object_filtering(self, train):
        for conn in (4, 8):
            for mono in (True, False):
                for mode in ("largest", "smallest", "unique_color", "most_colors", "has_holes"):
                    def make_extractor(c=conn, m=mono, md=mode):
                        def fn(g):
                            objs = extract_objects(g, connectivity=c, mono=m)
                            if not objs: return g
                            if md == "largest":
                                target = max(objs, key=lambda o: o.area)
                            elif md == "smallest":
                                target = min(objs, key=lambda o: o.area)
                            elif md == "most_colors":
                                target = max(objs, key=lambda o: len(np.unique(o.subgrid)))
                            else:
                                target = objs[0]
                            return target.subgrid
                        return fn
                    fn = make_extractor()
                    try:
                        if all(exact(fn(inp), out) for inp, out in train):
                            return fn
                    except Exception:
                        pass
        return None

    # --- 12. Object Recoloring by Area Rank / Sequence ---
    def _try_object_recolor(self, train):
        for conn in (4, 8):
            # Check if there is a consistent palette sequence applied to sorted objects
            def check_recolor(c=conn):
                # deduce palette sequence from first training example
                inp0, out0 = train[0]
                if inp0.shape != out0.shape: return None
                objs0 = extract_objects(inp0, connectivity=c)
                if len(objs0) < 2: return None
                # sort by area
                objs0.sort(key=lambda o: o.area)
                pal = []
                for o in objs0:
                    out_colors = [out0[r, c] for r, c in o.cells]
                    if len(set(out_colors)) != 1: return None
                    pal.append(out_colors[0])
                
                # verify on all train
                def fn(g):
                    h, w = g.shape
                    out = g.copy()
                    objs = extract_objects(g, connectivity=c)
                    objs.sort(key=lambda o: o.area)
                    for i, o in enumerate(objs):
                        if i < len(pal):
                            for r, c in o.cells:
                                out[r, c] = pal[i]
                    return out
                return fn
            fn = check_recolor()
            if fn:
                try:
                    if all(exact(fn(inp), out) for inp, out in train):
                        return fn
                except Exception:
                    pass
        return None

    # --- 13. Fill Bounding Box of Objects ---
    def _try_bbox_fill(self, train):
        for conn in (4, 8):
            def make_fill(c=conn):
                def fn(g):
                    out = g.copy()
                    objs = extract_objects(g, connectivity=c)
                    for o in objs:
                        min_r, min_c, max_r, max_c = o.bbox
                        out[min_r:max_r+1, min_c:max_c+1] = o.color
                    return out
                return fn
            fn = make_fill()
            try:
                if all(exact(fn(inp), out) for inp, out in train):
                    return fn
            except Exception:
                pass
        return None

    # --- 14. Periodic Pattern Extension / Extrapolation ---
    def _try_periodic_extension(self, train):
        # Extend 3x3 pattern to target shape with color mapping
        for period_h in (2, 3, 4):
            for period_w in (2, 3, 4):
                for target_c in range(10):
                    def make_ext(ph=period_h, pw=period_w, tc=target_c):
                        def fn(g):
                            h, w = g.shape
                            # find period block
                            block = g[:ph, :pw]
                            if tc != 0:
                                block = np.where(block != 0, tc, 0)
                            # repeat to fill g or larger
                            rep_y = (h + ph - 1) // ph + 1
                            rep_x = (w + pw - 1) // pw + 1
                            big = np.tile(block, (rep_y, rep_x))
                            # return big matching target output size if predictable
                            return big[:h+ph, :w]
                        return fn
                    fn = make_ext()
                    try:
                        if all(exact(fn(inp), out) for inp, out in train):
                            return fn
                    except Exception:
                        pass
        return None

    # --- 15. Tiling ---
    def _try_tiling(self, train):
        for ny in (2, 3):
            for nx in (2, 3):
                def make_tile(y=ny, x=nx):
                    return lambda g: np.tile(g, (y, x))
                fn = make_tile()
                try:
                    if all(exact(fn(inp), out) for inp, out in train):
                        return fn
                except Exception:
                    pass
        return None

    # --- 16. Composition ---
    def _try_composition(self, train):
        # Crop non-zero -> Rotate
        for rot in (1, 2, 3):
            def make_crop_rot(r=rot):
                def fn(g):
                    rows, cols = np.where(g != 0)
                    if len(rows) == 0: return g
                    sub = g[rows.min():rows.max()+1, cols.min():cols.max()+1]
                    return np.rot90(sub, r)
                return fn
            fn = make_crop_rot()
            try:
                if all(exact(fn(inp), out) for inp, out in train):
                    return fn
            except Exception:
                pass

        # Crop non-zero -> Flip
        for flip_mode in ("h", "v"):
            def make_crop_flip(fm=flip_mode):
                def fn(g):
                    rows, cols = np.where(g != 0)
                    if len(rows) == 0: return g
                    sub = g[rows.min():rows.max()+1, cols.min():cols.max()+1]
                    return np.fliplr(sub) if fm == "h" else np.flipud(sub)
                return fn
            fn = make_crop_flip()
            try:
                if all(exact(fn(inp), out) for inp, out in train):
                    return fn
            except Exception:
                pass
        return None
