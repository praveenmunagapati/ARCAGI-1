"""
MATHX ARC-AGI-1 PURE SYMBOLIC ENGINE v5 (STRICT NON-LLM)
Ultra-High-Performance Deductive Solver — 200+ Composable Symbolic Primitives
+ 2-Step Composition Engine + I/O Analysis + Background-Aware Transforms
Zero LLM Dependencies — 100% Deterministic Code
"""

from __future__ import annotations
import json, time, argparse
from pathlib import Path
from typing import Callable, Optional
from collections import Counter, deque
import numpy as np

Grid = np.ndarray
Prog = Callable[[Grid], Grid]

TASK_TIMEOUT = 15.0  # seconds per task

def G(x) -> Grid:
    return np.asarray(x, dtype=np.int32)

def exact(a: Optional[Grid], b: Optional[Grid]) -> bool:
    if a is None or b is None: return False
    try:
        return a.shape == b.shape and np.array_equal(a, b)
    except: return False

def safe_call(fn, g):
    try:
        r = fn(g)
        if r is None: return None
        if not isinstance(r, np.ndarray): return None
        if r.ndim != 2: return None
        if r.shape[0] == 0 or r.shape[1] == 0: return None
        if r.shape[0] > 30 or r.shape[1] > 30: return None
        return r
    except:
        return None


# ============================================================
# CORE UTILITIES
# ============================================================

def crop_nz(g: Grid) -> Optional[Grid]:
    r, c = np.where(g != 0)
    if len(r) == 0: return None
    return g[r.min():r.max()+1, c.min():c.max()+1].copy()

def crop_bg(g: Grid, bg: int) -> Optional[Grid]:
    r, c = np.where(g != bg)
    if len(r) == 0: return None
    return g[r.min():r.max()+1, c.min():c.max()+1].copy()

def crop_color(g: Grid, color: int) -> Optional[Grid]:
    r, c = np.where(g == color)
    if len(r) == 0: return None
    return g[r.min():r.max()+1, c.min():c.max()+1].copy()

def get_bg(g: Grid) -> int:
    vals, counts = np.unique(g, return_counts=True)
    total = g.size
    zero_idx = np.where(vals == 0)[0]
    if len(zero_idx) > 0 and counts[zero_idx[0]] >= total * 0.25:
        return 0
    return int(vals[np.argmax(counts)])

def get_all_colors(g: Grid) -> set:
    return set(map(int, np.unique(g)))

def get_nonbg_colors(g: Grid, bg: int = 0) -> list:
    return sorted(set(map(int, np.unique(g))) - {bg})

def overlay(top: Grid, bottom: Grid, bg: int = 0) -> Optional[Grid]:
    if top.shape != bottom.shape: return None
    out = bottom.copy()
    mask = top != bg
    out[mask] = top[mask]
    return out

def pad_to(g: Grid, h: int, w: int, bg: int = 0) -> Grid:
    out = np.full((h, w), bg, dtype=np.int32)
    rh, rw = min(h, g.shape[0]), min(w, g.shape[1])
    out[:rh, :rw] = g[:rh, :rw]
    return out


# ============================================================
# OBJECT SEGMENTATION
# ============================================================

def get_objects(g: Grid, conn: int = 4, bg: int = 0, mono: bool = True) -> list[dict]:
    h, w = g.shape
    vis = np.zeros((h, w), dtype=bool)
    objs = []
    dirs4 = [(-1,0),(1,0),(0,-1),(0,1)]
    dirs8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    dirs = dirs4 if conn == 4 else dirs8
    for r in range(h):
        for c in range(w):
            if vis[r,c] or g[r,c] == bg: continue
            color = int(g[r,c])
            cells = []
            stk = [(r,c)]; vis[r,c] = True
            while stk:
                cr,cc = stk.pop()
                cells.append((cr,cc))
                for dr,dc in dirs:
                    nr,nc = cr+dr, cc+dc
                    if 0<=nr<h and 0<=nc<w and not vis[nr,nc]:
                        if (mono and g[nr,nc]==color) or (not mono and g[nr,nc]!=bg):
                            vis[nr,nc]=True; stk.append((nr,nc))
            rs=[x[0] for x in cells]; cs=[x[1] for x in cells]
            mr,Mr,mc,Mc = min(rs),max(rs),min(cs),max(cs)
            mask = np.full((Mr-mr+1, Mc-mc+1), bg, dtype=np.int32)
            for cr,cc in cells: mask[cr-mr, cc-mc] = g[cr,cc]
            objs.append({
                'color': color, 'cells': cells, 'area': len(cells),
                'bbox': (mr,mc,Mr,Mc), 'h': Mr-mr+1, 'w': Mc-mc+1,
                'mask': mask, 'min_r': mr, 'min_c': mc,
            })
    return objs

def get_objects_multi(g: Grid, conn: int = 4, bg: int = 0) -> list[dict]:
    return get_objects(g, conn=conn, bg=bg, mono=False)


# ============================================================
# PANEL SPLITTING
# ============================================================

def find_divider_color(g: Grid) -> Optional[int]:
    h, w = g.shape
    for dc in range(10):
        dr = [r for r in range(h) if np.all(g[r,:]==dc)]
        dcc = [c for c in range(w) if np.all(g[:,c]==dc)]
        if len(dr) >= 1 or len(dcc) >= 1:
            return dc
    return None

def split_panels(g: Grid, dc: int) -> list[Grid]:
    h, w = g.shape
    dr = [r for r in range(h) if np.all(g[r,:]==dc)]
    dcc = [c for c in range(w) if np.all(g[:,c]==dc)]
    rs = [-1]+dr+[h]; cs_list = [-1]+dcc+[w]
    panels = []
    for i in range(len(rs)-1):
        r1,r2 = rs[i]+1, rs[i+1]
        for j in range(len(cs_list)-1):
            c1,c2 = cs_list[j]+1, cs_list[j+1]
            if r2>r1 and c2>c1: panels.append(g[r1:r2, c1:c2].copy())
    return panels


# ============================================================
# SHAPE NORMALIZATION FOR MATCHING
# ============================================================

def normalize_mask(mask: Grid, bg: int = 0) -> tuple:
    """Return a rotation/flip-invariant shape key."""
    variants = []
    m = mask.copy()
    m[m != bg] = 1
    m[m == bg] = 0
    for k in range(4):
        r = np.rot90(m, k)
        variants.append(tuple(r.flatten()))
        variants.append(tuple(np.fliplr(r).flatten()))
    return min(variants)

def mask_key(mask: Grid, bg: int = 0) -> tuple:
    """Exact shape key (not rotation invariant)."""
    m = mask.copy()
    m[m != bg] = 1
    m[m == bg] = 0
    return (m.shape, tuple(m.flatten()))


# ============================================================
# MASTER SYMBOLIC REASONING ENGINE v5
# ============================================================

class PureSymbolicSolverV5:
    
    def solve(self, task: dict) -> list[Prog]:
        train = [(G(ex["input"]), G(ex["output"])) for ex in task["train"]]
        solutions: list[Prog] = []
        t0 = time.perf_counter()
        
        # ---- I/O Analysis ----
        same_shape = all(inp.shape == out.shape for inp, out in train)
        inp0, out0 = train[0]
        ih, iw = inp0.shape
        oh, ow = out0.shape
        
        # Detect likely background colors
        bg_candidates = set()
        bg_candidates.add(0)
        for inp, out in train:
            bg_candidates.add(get_bg(inp))
            bg_candidates.add(get_bg(out))
        
        # Shape ratio analysis
        shape_ratio = None
        if oh > 0 and ow > 0 and ih > 0 and iw > 0:
            if oh % ih == 0 and ow % iw == 0:
                shape_ratio = (oh // ih, ow // iw)
            elif ih % oh == 0 and iw % ow == 0:
                shape_ratio = (-(ih // oh), -(iw // ow))  # negative = downscale
        
        # Get all solver families
        solver_families = self._get_all_solvers(train, same_shape, shape_ratio, bg_candidates)
        
        # Try all single-step solvers
        for s_fn in solver_families:
            if time.perf_counter() - t0 > TASK_TIMEOUT: break
            try:
                for c in s_fn(train):
                    try:
                        if all(exact(safe_call(c, inp), out) for inp, out in train):
                            solutions.append(c)
                            if len(solutions) >= 3: return solutions
                    except: pass
            except: pass
        
        if solutions: return solutions
        
        # Try 2-step composition
        if time.perf_counter() - t0 < TASK_TIMEOUT - 2:
            for c in self._compose(train, t0):
                try:
                    if all(exact(safe_call(c, inp), out) for inp, out in train):
                        solutions.append(c)
                        if len(solutions) >= 3: return solutions
                except: pass
        
        return solutions
    
    def _get_all_solvers(self, train, same_shape, shape_ratio, bg_candidates):
        """Return ordered list of solver families based on I/O analysis."""
        solvers = []
        
        # Always try these fast solvers first
        solvers.extend([
            self._rigid,
            self._palette,
            self._palette_with_bg,
        ])
        
        if same_shape:
            # Shape-preserving solvers
            solvers.extend([
                self._holes,
                self._gravity,
                self._lines,
                self._diamond_dilation,
                self._cellular,
                self._neighbor_count_recolor,
                self._border_recolor,
                self._replace_bg_around_objects,
                self._symmetry,
                self._mirror_complete,
                self._diagonal_mirror,
                self._per_color_shape_stamp,
                self._multi_color_object_stamp,
                self._row_col_intersection,
                self._directional_trail,
                self._cross_line_markers,
                self._object_symmetry_fill,
                self._pixel_position_rule,
                self._periodic_fill,
                self._object_pair_reflection,
                self._mask_overlay_objects,
                self._object_translation,
                self._bbox_fill,
                self._stamp_pattern_at_markers,
                self._majority_per_object,
                self._invert_colors,
                self._outline_objects,
                self._color_zone_propagation,
                self._flood_fill_per_object,
                self._extended_neighborhood_rule,
                self._conditional_pixel_transform,
                self._fill_enclosed_per_color,
                self._connect_same_color_hv,
                self._recolor_by_enclosure,
                self._draw_borders_around_objects,
                self._checkerboard_fill,
                self._pixel_neighbor_color_rule,
                self._repair_with_pattern,
                self._overlay_all_objects,
                self._recolor_by_object_size,
                self._extend_lines_to_border,
                self._paint_between_markers,
                self._gravity_with_obstacles,
                self._iterated_cellular,
                self._corner_fill,
                self._object_interior_fill_bg,
                self._row_col_color_rule,
            ])
        
        # Shape-changing solvers
        solvers.extend([
            self._cropping,
            self._scaling,
            self._downsampling,
            self._tiling,
            self._kronecker,
            self._dividers,
            self._obj_filter,
            self._obj_rank_recolor,
            self._crop_and_tile,
            self._sort_rows_cols,
            self._row_col_dedup,
            self._color_counting_output,
            self._most_common_object,
            self._object_sort_stack,
            self._extract_repeated_tile,
            self._panel_dimension_count,
            self._row_extension_with_color_sub,
            self._panel_majority_threshold,
            self._deduce_output_from_panels,
            self._unique_color_extraction,
            self._diagonal_periodic,
            self._alternating_ray_propagation,
            self._rigid_gravity_collision,
            self._spiral_fill,
            self._subgrid_majority,
            self._extract_unique_shape,
            self._count_objects_to_grid,
            self._compress_grid,
            self._extract_by_frame,
            self._split_and_select_by_content,
            self._mirrored_tiling,
            self._upscale_pattern,
            self._assemble_from_objects,
            self._extract_diff_region,
        ])
        
        # Two-step combos (selected, not full composition)
        solvers.extend([
            self._two_step,
        ])
        
        return solvers

    # ============================================================
    # 1. RIGID & AFFINE
    # ============================================================
    def _rigid(self, train) -> list[Prog]:
        cands: list[Prog] = [
            lambda g: g.copy(),
            lambda g: np.rot90(g, 1),
            lambda g: np.rot90(g, 2),
            lambda g: np.rot90(g, 3),
            lambda g: np.fliplr(g),
            lambda g: np.flipud(g),
            lambda g: g.T,
            lambda g: np.fliplr(g.T),
        ]
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                if dr == 0 and dc == 0: continue
                def mk(r=dr, c=dc): return lambda g: np.roll(g, (r,c), axis=(0,1))
                cands.append(mk())
        return cands

    # ============================================================
    # 2. PALETTE BIJECTION
    # ============================================================
    def _palette(self, train) -> list[Prog]:
        cands: list[Prog] = []
        mapping = {}; ok = True
        for inp, out in train:
            if inp.shape != out.shape: ok = False; break
            for u in np.unique(inp):
                oc = out[inp == int(u)]
                if len(np.unique(oc)) != 1: ok = False; break
                t = int(oc[0])
                if int(u) in mapping and mapping[int(u)] != t: ok = False; break
                mapping[int(u)] = t
            if not ok: break
        if ok and mapping:
            def mk(m=mapping.copy()):
                def fn(g):
                    out = g.copy()
                    for k, v in m.items(): out[g == k] = v
                    return out
                return fn
            cands.append(mk())
        return cands

    def _palette_with_bg(self, train) -> list[Prog]:
        """Try palette mapping treating different colors as background."""
        cands: list[Prog] = []
        for bg in range(1, 10):
            mapping = {}; ok = True
            for inp, out in train:
                if inp.shape != out.shape: ok = False; break
                for u in np.unique(inp):
                    oc = out[inp == int(u)]
                    if len(np.unique(oc)) != 1: ok = False; break
                    t = int(oc[0])
                    if int(u) in mapping and mapping[int(u)] != t: ok = False; break
                    mapping[int(u)] = t
                if not ok: break
            if ok and mapping and mapping != {k:k for k in mapping}:
                def mk(m=mapping.copy()):
                    def fn(g):
                        out = g.copy()
                        for k, v in m.items(): out[g == k] = v
                        return out
                    return fn
                cands.append(mk())
        return cands

    # ============================================================
    # 3. MULTI-PANEL & BOOLEAN OVERLAYS
    # ============================================================
    def _dividers(self, train) -> list[Prog]:
        cands: list[Prog] = []
        for dc in range(10):
            # Boolean overlays — try bg=0 (panels don't contain divider after split)
            for bg_val in (0, dc):
                for op in ("and", "xor", "or", "diff", "diff_r"):
                    for rc in range(10):
                        def mk(d=dc, o=op, r_c=rc, bv=bg_val):
                            def fn(g):
                                ps = split_panels(g, d)
                                if len(ps)!=2 or ps[0].shape!=ps[1].shape: return None
                                a, b = (ps[0]!=bv), (ps[1]!=bv)
                                if o=="and": m = a & b
                                elif o=="xor": m = a ^ b
                                elif o=="or": m = a | b
                                elif o=="diff": m = a & (~b)
                                elif o=="diff_r": m = b & (~a)
                                else: return None
                                res = np.full_like(ps[0], bv)
                                if r_c != 0:
                                    res[m] = r_c
                                else:
                                    res[m] = np.where(ps[0][m]!=bv, ps[0][m], ps[1][m])
                                return res
                            return fn
                        cands.append(mk())
            
            # Panel selection
            for idx in (0,1,2,-1):
                def mk_idx(d=dc, i=idx):
                    def fn(g):
                        ps = split_panels(g, d)
                        if not ps or abs(i)>=len(ps): return None
                        return ps[i]
                    return fn
                cands.append(mk_idx())
            
            for sel in ("max","min"):
                def mk_sel(d=dc, s=sel):
                    def fn(g):
                        ps = split_panels(g, d)
                        if not ps: return None
                        return max(ps, key=lambda p: np.count_nonzero(p)) if s=="max" else min(ps, key=lambda p: np.count_nonzero(p))
                    return fn
                cands.append(mk_sel())
            
            # Overlay all panels
            def mk_overlay(d=dc):
                def fn(g):
                    ps = split_panels(g, d)
                    if len(ps) < 2: return None
                    if not all(p.shape == ps[0].shape for p in ps): return None
                    out = np.zeros_like(ps[0])
                    for p in ps:
                        mask = p != d
                        out[mask] = p[mask]
                    return out
                return fn
            cands.append(mk_overlay())
            
            # Overlay with specific combination
            def mk_overlay_last(d=dc):
                def fn(g):
                    ps = split_panels(g, d)
                    if len(ps) < 2: return None
                    if not all(p.shape == ps[0].shape for p in ps): return None
                    out = ps[0].copy()
                    for p in ps[1:]:
                        mask = p != d
                        out[mask] = p[mask]
                    return out
                return fn
            cands.append(mk_overlay_last())
        return cands

    # ============================================================
    # 4. DIAGONAL PERIODIC PATTERNS
    # ============================================================
    def _diagonal_periodic(self, train) -> list[Prog]:
        cands: list[Prog] = []
        for K in (2, 3, 4, 5, 6, 7):
            for expr_type in ("r+c", "r-c", "r", "c"):
                def mk(period=K, et=expr_type):
                    def fn(g):
                        h, w = g.shape
                        mapping = {}
                        for r in range(h):
                            for c in range(w):
                                if g[r, c] != 0:
                                    col = int(g[r, c])
                                    if et == "r+c": rem = (r + c) % period
                                    elif et == "r-c": rem = (r - c) % period
                                    elif et == "r": rem = r % period
                                    elif et == "c": rem = c % period
                                    else: return None
                                    if rem in mapping and mapping[rem] != col: return None
                                    mapping[rem] = col
                        if len(mapping) == period:
                            out = np.zeros((h, w), dtype=np.int32)
                            for r in range(h):
                                for c in range(w):
                                    if et == "r+c": rem = (r + c) % period
                                    elif et == "r-c": rem = (r - c) % period
                                    elif et == "r": rem = r % period
                                    elif et == "c": rem = c % period
                                    else: rem = 0
                                    out[r, c] = mapping[rem]
                            return out
                        return None
                    return fn
                cands.append(mk())
        return cands

    # ============================================================
    # 5. RIGID GRAVITY COLLISION
    # ============================================================
    def _rigid_gravity_collision(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        colors = get_nonbg_colors(inp0)
        if len(colors) < 2: return cands
        
        for anchor_c in colors:
            for mover_c in colors:
                if anchor_c == mover_c: continue
                def make_fn(ac=anchor_c, mc=mover_c):
                    def fn(g):
                        h, w = g.shape
                        anchor_pts = list(zip(*np.where(g == ac)))
                        mover_pts = list(zip(*np.where(g == mc)))
                        if not anchor_pts or not mover_pts: return None
                        
                        ar_center = np.mean([r for r, c in anchor_pts])
                        ac_center = np.mean([c for r, c in anchor_pts])
                        mr_center = np.mean([r for r, c in mover_pts])
                        mc_center = np.mean([c for r, c in mover_pts])
                        
                        dr_diff = ar_center - mr_center
                        dc_diff = ac_center - mc_center
                        
                        if abs(dr_diff) > abs(dc_diff):
                            sdr = 1 if dr_diff > 0 else -1; sdc = 0
                        else:
                            sdr = 0; sdc = 1 if dc_diff > 0 else -1
                            
                        out = np.zeros_like(g)
                        for r, c in anchor_pts: out[r, c] = ac
                        
                        best_k = 0
                        for k in range(max(h, w)):
                            shifted = [(r + k*sdr, c + k*sdc) for r, c in mover_pts]
                            if any(r < 0 or r >= h or c < 0 or c >= w for r, c in shifted): break
                            if any(out[r, c] != 0 for r, c in shifted): break
                            adj = False
                            for r, c in shifted:
                                for ar, ac_pt in anchor_pts:
                                    if abs(r - ar) + abs(c - ac_pt) == 1:
                                        adj = True; break
                                if adj: break
                            if adj:
                                best_k = k; break
                                
                        for r, c in mover_pts:
                            nr, nc = r + best_k*sdr, c + best_k*sdc
                            if 0 <= nr < h and 0 <= nc < w:
                                out[nr, nc] = mc
                        return out
                    return fn
                cands.append(make_fn())
        return cands

    # ============================================================
    # 6. ALTERNATING RAY PROPAGATION
    # ============================================================
    def _alternating_ray_propagation(self, train) -> list[Prog]:
        cands: list[Prog] = []
        def make_fn():
            def fn(g):
                h, w = g.shape
                pts = list(zip(*np.where(g != 0)))
                if len(pts) != 2: return None
                (r0, c0), (r1, c1) = pts[0], pts[1]
                col0, col1 = int(g[r0, c0]), int(g[r1, c1])
                out = np.zeros((h, w), dtype=np.int32)
                if c0 > c1: c0, c1 = c1, c0; col0, col1 = col1, col0
                if r0 > r1: r0, r1 = r1, r0; col0, col1 = col1, col0
                d_c = max(1, abs(c1 - c0))
                d_r = max(1, abs(r1 - r0))
                if abs(c1-c0) > 0:
                    period = 2 * d_c
                    for c in range(w):
                        rem = c % period
                        if rem == c0 % period: out[:, c] = col0
                        elif rem == c1 % period: out[:, c] = col1
                else:
                    period = 2 * d_r
                    for r in range(h):
                        rem = r % period
                        if rem == r0 % period: out[r, :] = col0
                        elif rem == r1 % period: out[r, :] = col1
                return out
            return fn
        cands.append(make_fn())
        return cands

    # ============================================================
    # 7. UNIQUE COLOR EXTRACTION
    # ============================================================
    def _unique_color_extraction(self, train) -> list[Prog]:
        cands: list[Prog] = []
        for conn in (4, 8):
            for mono in (True, False):
                # Extract object with unique color
                def make_fn(cn=conn, m=mono):
                    def fn(g):
                        objs = get_objects(g, conn=cn, mono=m)
                        if not objs: return None
                        col_counts = Counter(o['color'] for o in objs)
                        unique_cols = [c for c, count in col_counts.items() if count == 1]
                        if unique_cols:
                            for o in objs:
                                if o['color'] == unique_cols[0]:
                                    mr, mc, Mr, Mc = o['bbox']
                                    return g[mr:Mr+1, mc:Mc+1]
                        return None
                    return fn
                cands.append(make_fn())
                
                # Extract object with least frequent color
                def make_fn2(cn=conn, m=mono):
                    def fn(g):
                        objs = get_objects(g, conn=cn, mono=m)
                        if not objs: return None
                        col_counts = Counter(o['color'] for o in objs)
                        least = col_counts.most_common()[-1][0]
                        for o in objs:
                            if o['color'] == least:
                                mr, mc, Mr, Mc = o['bbox']
                                return g[mr:Mr+1, mc:Mc+1]
                        return None
                    return fn
                cands.append(make_fn2())
        return cands

    # ============================================================
    # 8. KRONECKER / FRACTAL
    # ============================================================
    def _kronecker(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        ih, iw = inp0.shape; oh, ow = out0.shape
        
        # Standard kronecker products
        cands.append(lambda g: np.kron((g > 0).astype(np.int32), g))
        cands.append(lambda g: np.kron(g, (g > 0).astype(np.int32)))
        
        if oh > ih and ow > iw and oh % ih == 0 and ow % iw == 0:
            sy, sx = oh // ih, ow // iw
            if sy == ih and sx == iw:
                def mk_self_tile():
                    def fn(g):
                        h, w = g.shape
                        out = np.zeros((h*h, w*w), dtype=np.int32)
                        for r in range(h):
                            for c in range(w):
                                if g[r, c] != 0:
                                    out[r*h:(r+1)*h, c*w:(c+1)*w] = g
                        return out
                    return fn
                cands.append(mk_self_tile())
                
                # Variant: each cell replaces with colored version
                def mk_self_tile2():
                    def fn(g):
                        h, w = g.shape
                        out = np.zeros((h*h, w*w), dtype=np.int32)
                        for r in range(h):
                            for c in range(w):
                                v = g[r, c]
                                if v != 0:
                                    block = np.where(g != 0, v, 0)
                                    out[r*h:(r+1)*h, c*w:(c+1)*w] = block
                        return out
                    return fn
                cands.append(mk_self_tile2())
        return cands

    # ============================================================
    # 9. SCALING
    # ============================================================
    def _scaling(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        oh, ow = out0.shape; ih, iw = inp0.shape
        for sy in range(2, 8):
            for sx in range(2, 8):
                if ih * sy == oh and iw * sx == ow:
                    def mk(y=sy, x=sx): return lambda g: np.repeat(np.repeat(g, y, axis=0), x, axis=1)
                    cands.append(mk())
        return cands

    # ============================================================
    # 10. DOWNSAMPLING
    # ============================================================
    def _downsampling(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        ih, iw = inp0.shape; oh, ow = out0.shape
        for sy in (2,3,4,5):
            for sx in (2,3,4,5):
                if ih == oh * sy and iw == ow * sx:
                    # Majority vote
                    def mk(y=sy, x=sx):
                        def fn(g):
                            h,w = g.shape
                            if h%y or w%x: return None
                            rh,rw = h//y, w//x
                            out = np.zeros((rh,rw), dtype=np.int32)
                            for r in range(rh):
                                for c in range(rw):
                                    blk = g[r*y:(r+1)*y, c*x:(c+1)*x]
                                    nz = blk[blk!=0]
                                    if len(nz):
                                        v,cn = np.unique(nz, return_counts=True)
                                        out[r,c] = v[np.argmax(cn)]
                            return out
                        return fn
                    cands.append(mk())
                    
                    # Any non-zero
                    def mk2(y=sy, x=sx):
                        def fn(g):
                            h,w = g.shape
                            if h%y or w%x: return None
                            rh,rw = h//y, w//x
                            out = np.zeros((rh,rw), dtype=np.int32)
                            for r in range(rh):
                                for c in range(rw):
                                    blk = g[r*y:(r+1)*y, c*x:(c+1)*x]
                                    nz = blk[blk!=0]
                                    if len(nz): out[r,c] = int(nz[0])
                            return out
                        return fn
                    cands.append(mk2())
                    
                    # Top-left pixel
                    def mk3(y=sy, x=sx):
                        def fn(g):
                            h,w = g.shape
                            if h%y or w%x: return None
                            return g[::y, ::x].copy()
                        return fn
                    cands.append(mk3())
        return cands

    # ============================================================
    # 11. TILING (output = tiled input)
    # ============================================================
    def _tiling(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        ih, iw = inp0.shape; oh, ow = out0.shape
        if oh >= ih and ow >= iw and oh % ih == 0 and ow % iw == 0:
            ny, nx = oh // ih, ow // iw
            if ny > 1 or nx > 1:
                def mk(y=ny, x=nx): return lambda g: np.tile(g, (y, x))
                cands.append(mk())
                # Tile with alternating flips
                def mk_mirror_h(y=ny, x=nx):
                    def fn(g):
                        rows = []
                        for i in range(y):
                            row_tiles = []
                            for j in range(x):
                                t = g.copy()
                                if j % 2 == 1: t = np.fliplr(t)
                                if i % 2 == 1: t = np.flipud(t)
                                row_tiles.append(t)
                            rows.append(np.hstack(row_tiles))
                        return np.vstack(rows)
                    return fn
                cands.append(mk_mirror_h())
        return cands

    # ============================================================
    # 12. CROPPING & SUBGRID EXTRACTION
    # ============================================================
    def _cropping(self, train) -> list[Prog]:
        cands: list[Prog] = []
        
        # Crop to non-zero
        cands.append(lambda g: crop_nz(g))
        
        # Crop to each color
        for tc in range(1, 10):
            def mk_col(t=tc):
                def fn(g):
                    r,c = np.where(g==t)
                    if len(r)==0: return None
                    return g[r.min():r.max()+1, c.min():c.max()+1]
                return fn
            cands.append(mk_col())
        
        # Crop inside frame of specific color
        for fc in range(10):
            def mk_frame(f=fc):
                def fn(g):
                    r,c = np.where(g==f)
                    if len(r)==0: return None
                    mr,Mr,mc,Mc = r.min(),r.max(),c.min(),c.max()
                    if Mr-mr>1 and Mc-mc>1: return g[mr+1:Mr, mc+1:Mc]
                    return None
                return fn
            cands.append(mk_frame())
        
        # Crop to non-bg content (for each possible bg)
        for bg in range(1, 10):
            def mk_bg(b=bg):
                def fn(g):
                    return crop_bg(g, b)
                return fn
            cands.append(mk_bg())
        
        # Hollow rectangular frame interior crop
        def crop_hollow_frame(g):
            h, w = g.shape
            colors_in = get_nonbg_colors(g)
            for c in colors_in:
                rows, cols = np.where(g == c)
                if len(rows) >= 8:
                    r1, r2 = rows.min(), rows.max()
                    c1, c2 = cols.min(), cols.max()
                    if (r2 - r1 >= 2 and c2 - c1 >= 2 and
                        np.all(g[r1, c1:c2+1] == c) and
                        np.all(g[r2, c1:c2+1] == c) and
                        np.all(g[r1:r2+1, c1] == c) and
                        np.all(g[r1:r2+1, c2] == c)):
                        return g[r1+1:r2, c1+1:c2]
            return None
        cands.append(crop_hollow_frame)
        
        # Quadrant extraction
        for q in ("tl", "tr", "bl", "br"):
            def mk_quad(quad=q):
                def fn(g):
                    rows, cols = np.where(g != 0)
                    if len(rows) == 0: return None
                    sub = g[rows.min():rows.max()+1, cols.min():cols.max()+1]
                    sh, sw = sub.shape
                    if sh < 2 or sw < 2: return None
                    if quad == "tl": return sub[:sh//2, :sw//2]
                    elif quad == "tr": return sub[:sh//2, sw//2:]
                    elif quad == "bl": return sub[sh//2:, :sw//2]
                    elif quad == "br": return sub[sh//2:, sw//2:]
                    return None
                return fn
            cands.append(mk_quad())
        
        # Panel with anomaly
        def panel_anomaly(g):
            h, w = g.shape
            for dc in range(10):
                ps = split_panels(g, dc)
                if len(ps) >= 2:
                    col_sets = [set(map(int, np.unique(p))) - {0, dc} for p in ps]
                    for idx, cset in enumerate(col_sets):
                        other_colors = set().union(*[col_sets[j] for j in range(len(ps)) if j != idx])
                        if len(cset - other_colors) > 0:
                            return ps[idx]
            return None
        cands.append(panel_anomaly)
        
        # Extract mask of specific color (non-zero → 1 pattern)
        for tc in range(1, 10):
            def mk_mask(t=tc):
                def fn(g):
                    r, c = np.where(g == t)
                    if len(r) == 0: return None
                    mr, Mr, mc, Mc = r.min(), r.max(), c.min(), c.max()
                    sub = g[mr:Mr+1, mc:Mc+1].copy()
                    # Replace background with 0, keep target color
                    mask = np.zeros_like(sub)
                    mask[sub == t] = t
                    return mask
                return fn
            cands.append(mk_mask())
        
        return cands

    # ============================================================
    # 13. SYMMETRY & MIRROR INPAINTING
    # ============================================================
    def _symmetry(self, train) -> list[Prog]:
        def sh_l(g): h,w=g.shape; m=w//2; o=g.copy(); o[:,w-m:]=np.fliplr(g[:,:m]); return o
        def sh_r(g): h,w=g.shape; m=w//2; o=g.copy(); o[:,:m]=np.fliplr(g[:,w-m:]); return o
        def sv_t(g): h,w=g.shape; m=h//2; o=g.copy(); o[h-m:,:]=np.flipud(g[:m,:]); return o
        def sv_b(g): h,w=g.shape; m=h//2; o=g.copy(); o[:m,:]=np.flipud(g[h-m:,:]); return o
        return [sh_l, sh_r, sv_t, sv_b]

    def _mirror_complete(self, train) -> list[Prog]:
        cands: list[Prog] = []
        
        def mirror_h(g):
            h,w = g.shape; out = g.copy()
            for r in range(h):
                for c in range(w):
                    mc = w - 1 - c
                    if out[r,c] == 0 and g[r,mc] != 0: out[r,c] = g[r,mc]
                    elif out[r,mc] == 0 and g[r,c] != 0: out[r,mc] = g[r,c]
            return out
        cands.append(mirror_h)
        
        def mirror_v(g):
            h,w = g.shape; out = g.copy()
            for r in range(h):
                mr = h - 1 - r
                for c in range(w):
                    if out[r,c] == 0 and g[mr,c] != 0: out[r,c] = g[mr,c]
                    elif out[mr,c] == 0 and g[r,c] != 0: out[mr,c] = g[r,c]
            return out
        cands.append(mirror_v)
        
        def mirror_hv(g):
            h,w = g.shape; out = g.copy()
            for r in range(h):
                for c in range(w):
                    if out[r,c] == 0:
                        if g[r, w-1-c] != 0: out[r,c] = g[r, w-1-c]
                        elif g[h-1-r, c] != 0: out[r,c] = g[h-1-r, c]
                        elif g[h-1-r, w-1-c] != 0: out[r,c] = g[h-1-r, w-1-c]
            return out
        cands.append(mirror_hv)
        
        # Mirror with non-zero background
        for bg in range(1, 10):
            def mk_mirh(b=bg):
                def fn(g):
                    h,w = g.shape; out = g.copy()
                    for r in range(h):
                        for c in range(w):
                            mc = w - 1 - c
                            if out[r,c] == b and g[r,mc] != b: out[r,c] = g[r,mc]
                    return out
                return fn
            cands.append(mk_mirh())
            
            def mk_mirv(b=bg):
                def fn(g):
                    h,w = g.shape; out = g.copy()
                    for r in range(h):
                        mr = h - 1 - r
                        for c in range(w):
                            if out[r,c] == b and g[mr,c] != b: out[r,c] = g[mr,c]
                    return out
                return fn
            cands.append(mk_mirv())
        
        return cands

    # ============================================================
    # 14. ENCLOSED HOLES & FLOOD FILL
    # ============================================================
    def _holes(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        diff = out0[inp0 != out0] if np.any(inp0 != out0) else np.array([])
        fill_colors = list(set(map(int, np.unique(diff)))) if len(diff) > 0 else list(range(1, 10))
        
        for fc in fill_colors:
            def mk(f=fc):
                def fn(g):
                    h,w=g.shape; out=g.copy()
                    vis=np.zeros((h,w),dtype=bool); stk=[]
                    for r in range(h):
                        for c in (0,w-1):
                            if g[r,c]==0 and not vis[r,c]: vis[r,c]=True; stk.append((r,c))
                    for c in range(w):
                        for r in (0,h-1):
                            if g[r,c]==0 and not vis[r,c]: vis[r,c]=True; stk.append((r,c))
                    while stk:
                        r,c=stk.pop()
                        for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                            nr,nc=r+dr,c+dc
                            if 0<=nr<h and 0<=nc<w and g[nr,nc]==0 and not vis[nr,nc]:
                                vis[nr,nc]=True; stk.append((nr,nc))
                    for r in range(h):
                        for c in range(w):
                            if g[r,c]==0 and not vis[r,c]: out[r,c]=f
                    return out
                return fn
            cands.append(mk())
        
        # Fill holes with surrounding object color
        def fill_holes_with_obj_color(g):
            h,w=g.shape; out=g.copy()
            vis=np.zeros((h,w),dtype=bool); stk=[]
            for r in range(h):
                for c in (0,w-1):
                    if g[r,c]==0 and not vis[r,c]: vis[r,c]=True; stk.append((r,c))
            for c in range(w):
                for r in (0,h-1):
                    if g[r,c]==0 and not vis[r,c]: vis[r,c]=True; stk.append((r,c))
            while stk:
                r,c=stk.pop()
                for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                    nr,nc=r+dr,c+dc
                    if 0<=nr<h and 0<=nc<w and g[nr,nc]==0 and not vis[nr,nc]:
                        vis[nr,nc]=True; stk.append((nr,nc))
            # For each hole region, find surrounding color
            hole_vis = np.zeros((h,w), dtype=bool)
            for r in range(h):
                for c in range(w):
                    if g[r,c]==0 and not vis[r,c] and not hole_vis[r,c]:
                        # BFS to find this hole region
                        region = []
                        q = deque([(r,c)]); hole_vis[r,c] = True
                        while q:
                            cr,cc = q.popleft()
                            region.append((cr,cc))
                            for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                                nr,nc=cr+dr,cc+dc
                                if 0<=nr<h and 0<=nc<w and g[nr,nc]==0 and not vis[nr,nc] and not hole_vis[nr,nc]:
                                    hole_vis[nr,nc]=True; q.append((nr,nc))
                        # Find surrounding colors
                        surr = Counter()
                        for cr,cc in region:
                            for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                                nr,nc=cr+dr,cc+dc
                                if 0<=nr<h and 0<=nc<w and g[nr,nc]!=0:
                                    surr[int(g[nr,nc])] += 1
                        if surr:
                            fill = surr.most_common(1)[0][0]
                            for cr,cc in region: out[cr,cc] = fill
            return out
        cands.append(fill_holes_with_obj_color)
        
        # Fill with non-zero background
        for bg in range(1, 10):
            for fc in fill_colors:
                if fc == bg: continue
                def mk_bg(b=bg, f=fc):
                    def fn(g):
                        h,w=g.shape; out=g.copy()
                        vis=np.zeros((h,w),dtype=bool); stk=[]
                        for r in range(h):
                            for c in (0,w-1):
                                if g[r,c]==b and not vis[r,c]: vis[r,c]=True; stk.append((r,c))
                        for c in range(w):
                            for r in (0,h-1):
                                if g[r,c]==b and not vis[r,c]: vis[r,c]=True; stk.append((r,c))
                        while stk:
                            r,c=stk.pop()
                            for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                                nr,nc=r+dr,c+dc
                                if 0<=nr<h and 0<=nc<w and g[nr,nc]==b and not vis[nr,nc]:
                                    vis[nr,nc]=True; stk.append((nr,nc))
                        for r in range(h):
                            for c in range(w):
                                if g[r,c]==b and not vis[r,c]: out[r,c]=f
                        return out
                    return fn
                cands.append(mk_bg())
        
        return cands

    # ============================================================
    # 15. DIRECTIONAL GRAVITY
    # ============================================================
    def _gravity(self, train) -> list[Prog]:
        cands: list[Prog] = []
        for d in ("down","up","left","right"):
            def mk(dr=d):
                def fn(g):
                    h,w=g.shape; out=np.zeros_like(g)
                    if dr=="down":
                        for c in range(w): col=g[:,c]; nz=col[col!=0]; out[h-len(nz):,c]=nz
                    elif dr=="up":
                        for c in range(w): col=g[:,c]; nz=col[col!=0]; out[:len(nz),c]=nz
                    elif dr=="right":
                        for r in range(h): row=g[r,:]; nz=row[row!=0]; out[r,w-len(nz):]=nz
                    elif dr=="left":
                        for r in range(h): row=g[r,:]; nz=row[row!=0]; out[r,:len(nz)]=nz
                    return out
                return fn
            cands.append(mk())
        
        # Gravity preserving background
        for bg in range(1, 10):
            for d in ("down","up","left","right"):
                def mk_bg(dr=d, b=bg):
                    def fn(g):
                        h,w=g.shape; out=np.full_like(g, b)
                        if dr=="down":
                            for c in range(w):
                                col=g[:,c]; nz=col[col!=b]
                                out[h-len(nz):,c]=nz
                        elif dr=="up":
                            for c in range(w):
                                col=g[:,c]; nz=col[col!=b]
                                out[:len(nz),c]=nz
                        elif dr=="right":
                            for r in range(h):
                                row=g[r,:]; nz=row[row!=b]
                                out[r,w-len(nz):]=nz
                        elif dr=="left":
                            for r in range(h):
                                row=g[r,:]; nz=row[row!=b]
                                out[r,:len(nz)]=nz
                        return out
                    return fn
                cands.append(mk_bg())
        return cands

    # ============================================================
    # 16. GRAVITY WITH OBSTACLES
    # ============================================================
    def _gravity_with_obstacles(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        colors = get_nonbg_colors(inp0)
        if len(colors) < 2: return cands
        
        for obstacle_c in colors:
            for d in ("down","up","left","right"):
                def mk(oc=obstacle_c, dr=d):
                    def fn(g):
                        h,w=g.shape; out=g.copy()
                        if dr=="down":
                            for c in range(w):
                                # Collect non-zero, non-obstacle pixels
                                movers = []
                                obstacles = set()
                                for r in range(h):
                                    if g[r,c] == oc: obstacles.add(r)
                                    elif g[r,c] != 0: movers.append((r, int(g[r,c])))
                                # Clear mover positions
                                for r, _ in movers: out[r,c] = 0
                                # Drop each mover down
                                for r, col in reversed(movers):
                                    nr = h - 1
                                    while nr >= 0 and (out[nr,c] != 0 or nr in obstacles): nr -= 1
                                    if nr >= 0: out[nr,c] = col
                        elif dr=="up":
                            for c in range(w):
                                movers = []
                                obstacles = set()
                                for r in range(h):
                                    if g[r,c] == oc: obstacles.add(r)
                                    elif g[r,c] != 0: movers.append((r, int(g[r,c])))
                                for r, _ in movers: out[r,c] = 0
                                for r, col in movers:
                                    nr = 0
                                    while nr < h and (out[nr,c] != 0 or nr in obstacles): nr += 1
                                    if nr < h: out[nr,c] = col
                        return out
                    return fn
                cands.append(mk())
        return cands

    # ============================================================
    # 17. LINES, RAYS & CONNECTIONS
    # ============================================================
    def _lines(self, train) -> list[Prog]:
        cands: list[Prog] = []
        diff_cols = set()
        for inp, out in train:
            if inp.shape == out.shape:
                diff_cols |= set(map(int, np.unique(out[inp != out])))
        cand_colors = [0] + sorted(diff_cols)

        for rc in cand_colors:
            # Connect same-color points with lines
            def mk_conn(fill_col=rc):
                def connect(g):
                    h,w=g.shape; out=g.copy()
                    for cl in np.unique(g):
                        if cl==0: continue
                        rs,cs=np.where(g==cl); pts=list(zip(rs,cs))
                        for i in range(len(pts)):
                            for j in range(i+1,len(pts)):
                                r1,c1=pts[i]; r2,c2=pts[j]
                                col = fill_col if fill_col != 0 else cl
                                if r1==r2:
                                    for c in range(min(c1,c2), max(c1,c2)+1):
                                        if out[r1,c]==0: out[r1,c]=col
                                elif c1==c2:
                                    for r in range(min(r1,r2), max(r1,r2)+1):
                                        if out[r,c1]==0: out[r,c1]=col
                    return out
                return connect
            cands.append(mk_conn())

        # Cross rays (H+V through each colored point)
        for rc in cand_colors:
            def mk_cross_ray(fill_col=rc):
                def fn(g):
                    h, w = g.shape; out = g.copy()
                    for r in range(h):
                        for c in range(w):
                            if g[r, c] != 0:
                                col = fill_col if fill_col != 0 else int(g[r, c])
                                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                                    cr, cc = r + dr, c + dc
                                    while 0 <= cr < h and 0 <= cc < w:
                                        if out[cr, cc] == 0: out[cr, cc] = col
                                        else: break
                                        cr += dr; cc += dc
                    return out
                return fn
            cands.append(mk_cross_ray())
        
        # Cross rays that don't stop at obstacles
        for rc in cand_colors:
            def mk_cross_thru(fill_col=rc):
                def fn(g):
                    h, w = g.shape; out = g.copy()
                    for r in range(h):
                        for c in range(w):
                            if g[r, c] != 0:
                                col = fill_col if fill_col != 0 else int(g[r, c])
                                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                                    cr, cc = r + dr, c + dc
                                    while 0 <= cr < h and 0 <= cc < w:
                                        if out[cr, cc] == 0: out[cr, cc] = col
                                        cr += dr; cc += dc
                    return out
                return fn
            cands.append(mk_cross_thru())

        # Diagonal rays
        for rc in cand_colors:
            def mk_diag(fill_col=rc):
                def fn(g):
                    h, w = g.shape; out = g.copy()
                    for r in range(h):
                        for c in range(w):
                            if g[r, c] != 0:
                                col = fill_col if fill_col != 0 else int(g[r, c])
                                for dr, dc in ((-1,-1),(-1,1),(1,-1),(1,1)):
                                    cr, cc = r + dr, c + dc
                                    while 0 <= cr < h and 0 <= cc < w:
                                        if out[cr, cc] == 0: out[cr, cc] = col
                                        cr += dr; cc += dc
                    return out
                return fn
            cands.append(mk_diag())
        
        # All 8 rays
        for rc in cand_colors:
            def mk_8ray(fill_col=rc):
                def fn(g):
                    h, w = g.shape; out = g.copy()
                    for r in range(h):
                        for c in range(w):
                            if g[r, c] != 0:
                                col = fill_col if fill_col != 0 else int(g[r, c])
                                for dr, dc in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
                                    cr, cc = r + dr, c + dc
                                    while 0 <= cr < h and 0 <= cc < w:
                                        if out[cr, cc] == 0: out[cr, cc] = col
                                        else: break
                                        cr += dr; cc += dc
                    return out
                return fn
            cands.append(mk_8ray())

        # Wireframe bbox
        for rc in cand_colors:
            def mk_wireframe(fill_col=rc):
                def fn(g):
                    h, w = g.shape; out = g.copy()
                    for cl in np.unique(g):
                        if cl == 0: continue
                        rows, cols = np.where(g == cl)
                        if len(rows) >= 2:
                            r1, r2 = rows.min(), rows.max()
                            c1, c2 = cols.min(), cols.max()
                            col = fill_col if fill_col != 0 else cl
                            for c in range(c1, c2+1):
                                if out[r1, c] == 0: out[r1, c] = col
                                if out[r2, c] == 0: out[r2, c] = col
                            for r in range(r1, r2+1):
                                if out[r, c1] == 0: out[r, c1] = col
                                if out[r, c2] == 0: out[r, c2] = col
                    return out
                return fn
            cands.append(mk_wireframe())

        return cands

    def _diamond_dilation(self, train) -> list[Prog]:
        cands: list[Prog] = []
        diff_cols = set()
        for inp, out in train:
            if inp.shape == out.shape:
                diff_cols |= set(map(int, np.unique(out[inp != out])))
        cand_colors = [0] + sorted(diff_cols)
        for radius in (1, 2, 3, 4):
            for target_c in cand_colors:
                def mk(rad=radius, tc=target_c):
                    def fn(g):
                        h, w = g.shape; out = g.copy()
                        for r in range(h):
                            for c in range(w):
                                if g[r, c] != 0:
                                    col = tc if tc != 0 else int(g[r, c])
                                    for dr in range(-rad, rad+1):
                                        for dc in range(-rad, rad+1):
                                            if abs(dr) + abs(dc) <= rad:
                                                nr, nc = r+dr, c+dc
                                                if 0<=nr<h and 0<=nc<w and out[nr, nc] == 0:
                                                    out[nr, nc] = col
                        return out
                    return fn
                cands.append(mk())
        return cands

    # ============================================================
    # 18. OBJECT FILTERING & RANKING
    # ============================================================
    def _obj_filter(self, train) -> list[Prog]:
        cands: list[Prog] = []
        for conn in (4,8):
            for mono in (True,False):
                for mode in ("largest","smallest"):
                    def mk(c=conn,m=mono,md=mode):
                        def fn(g):
                            objs=get_objects(g,conn=c,mono=m)
                            if not objs: return None
                            t = max(objs,key=lambda o:o['area']) if md=="largest" else min(objs,key=lambda o:o['area'])
                            mr,mc,Mr,Mc = t['bbox']
                            return g[mr:Mr+1, mc:Mc+1]
                        return fn
                    cands.append(mk())
                
                # Extract mask only (on bg=0)
                for mode2 in ("largest","smallest"):
                    def mk2(c=conn,m=mono,md=mode2):
                        def fn(g):
                            objs=get_objects(g,conn=c,mono=m)
                            if not objs: return None
                            t = max(objs,key=lambda o:o['area']) if md=="largest" else min(objs,key=lambda o:o['area'])
                            return t['mask']
                        return fn
                    cands.append(mk2())
        
        # Extract by position (top-left, bottom-right, etc.)
        for conn in (4,8):
            for pos in ("top-left", "top-right", "bottom-left", "bottom-right"):
                def mk_pos(c=conn, p=pos):
                    def fn(g):
                        objs=get_objects(g,conn=c)
                        if not objs: return None
                        if p == "top-left": t = min(objs, key=lambda o: o['min_r'] + o['min_c'])
                        elif p == "top-right": t = min(objs, key=lambda o: o['min_r'] - o['min_c'])
                        elif p == "bottom-left": t = max(objs, key=lambda o: o['min_r'] - o['min_c'])
                        elif p == "bottom-right": t = max(objs, key=lambda o: o['min_r'] + o['min_c'])
                        else: return None
                        return t['mask']
                    return fn
                cands.append(mk_pos())
        
        return cands

    def _obj_rank_recolor(self, train) -> list[Prog]:
        cands: list[Prog] = []
        for conn in (4,8):
            inp0,out0 = train[0]
            if inp0.shape!=out0.shape: continue
            objs0 = get_objects(inp0, conn=conn)
            if len(objs0)<2: continue
            objs0.sort(key=lambda o:o['area'])
            pal=[]; ok=True
            for o in objs0:
                cols=[out0[r,c] for r,c in o['cells']]
                if len(set(cols))!=1: ok=False; break
                pal.append(cols[0])
            if ok and pal:
                def mk(c=conn,p=pal[:]):
                    def fn(g):
                        out=g.copy(); objs=get_objects(g,conn=c); objs.sort(key=lambda o:o['area'])
                        for i,o in enumerate(objs):
                            if i<len(p):
                                for r,cc in o['cells']: out[r,cc]=p[i]
                        return out
                    return fn
                cands.append(mk())
        return cands

    def _bbox_fill(self, train) -> list[Prog]:
        cands: list[Prog] = []
        for conn in (4,8):
            def mk(c=conn):
                def fn(g):
                    out=g.copy()
                    for o in get_objects(g,conn=c):
                        mr,mc,Mr,Mc = o['bbox']
                        out[mr:Mr+1,mc:Mc+1] = o['color']
                    return out
                return fn
            cands.append(mk())
        return cands

    def _stamp_pattern_at_markers(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        for marker_c in range(1, 10):
            marker_pos = list(zip(*np.where(inp0 == marker_c)))
            if not (1 <= len(marker_pos) <= 20): continue
            for radius in (1, 2, 3, 4):
                patches = []
                valid = True
                for mr, mc in marker_pos:
                    r1 = mr-radius; r2 = mr+radius+1
                    c1 = mc-radius; c2 = mc+radius+1
                    if r1 < 0 or r2 > inp0.shape[0] or c1 < 0 or c2 > inp0.shape[1]:
                        valid = False; break
                    patches.append(out0[r1:r2, c1:c2].copy())
                if valid and patches and all(np.array_equal(patches[0], p) for p in patches):
                    stamp = patches[0].copy()
                    def mk(mc_=marker_c, rad=radius, st=stamp.copy()):
                        def fn(g):
                            h,w=g.shape; out=g.copy()
                            for r in range(h):
                                for c in range(w):
                                    if g[r,c]==mc_:
                                        for dr in range(-rad, rad+1):
                                            for dc in range(-rad, rad+1):
                                                nr,nc = r+dr,c+dc
                                                if 0<=nr<h and 0<=nc<w:
                                                    sv = st[dr+rad, dc+rad]
                                                    if sv != 0: out[nr,nc] = sv
                            return out
                        return fn
                    cands.append(mk())
        return cands

    def _mask_overlay_objects(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        colors = sorted(set(map(int, np.unique(inp0))) - {0})
        for c1 in colors:
            for c2 in colors:
                if c1 == c2: continue
                def mk(a=c1, b=c2):
                    def fn(g):
                        out = g.copy(); out[g == b] = a; return out
                    return fn
                cands.append(mk())
        return cands

    def _object_translation(self, train) -> list[Prog]:
        cands: list[Prog] = []
        def center_content(g):
            h, w = g.shape
            rows, cols = np.where(g != 0)
            if len(rows) == 0: return g
            oh = rows.max() - rows.min() + 1
            ow = cols.max() - cols.min() + 1
            content = g[rows.min():rows.max()+1, cols.min():cols.max()+1]
            out = np.zeros_like(g)
            sr = (h - oh)//2; sc = (w - ow)//2
            out[sr:sr+oh, sc:sc+ow] = content
            return out
        cands.append(center_content)
        return cands

    # ============================================================
    # 19. CELLULAR AUTOMATA & NEIGHBORHOOD RULES
    # ============================================================
    def _cellular(self, train) -> list[Prog]:
        cands: list[Prog] = []
        def expand_cross(g):
            h,w=g.shape; out=g.copy()
            for r in range(h):
                for c in range(w):
                    if g[r,c]!=0:
                        col=int(g[r,c])
                        for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                            nr,nc=r+dr,c+dc
                            if 0<=nr<h and 0<=nc<w and out[nr,nc]==0: out[nr,nc]=col
            return out
        cands.append(expand_cross)
        
        def expand_8(g):
            h,w=g.shape; out=g.copy()
            for r in range(h):
                for c in range(w):
                    if g[r,c]!=0:
                        col=int(g[r,c])
                        for dr in (-1,0,1):
                            for dc in (-1,0,1):
                                if dr==0 and dc==0: continue
                                nr,nc=r+dr,c+dc
                                if 0<=nr<h and 0<=nc<w and out[nr,nc]==0: out[nr,nc]=col
            return out
        cands.append(expand_8)
        
        # Square dilation (box)
        for rad in (1, 2, 3):
            def mk_box(r=rad):
                def fn(g):
                    h,w=g.shape; out=g.copy()
                    for rr in range(h):
                        for cc in range(w):
                            if g[rr,cc]!=0:
                                col=int(g[rr,cc])
                                for dr in range(-r,r+1):
                                    for dc in range(-r,r+1):
                                        nr,nc=rr+dr,cc+dc
                                        if 0<=nr<h and 0<=nc<w and out[nr,nc]==0: out[nr,nc]=col
                    return out
                return fn
            cands.append(mk_box())
        
        return cands

    def _iterated_cellular(self, train) -> list[Prog]:
        """Apply cellular expansion multiple times."""
        cands: list[Prog] = []
        for steps in (2, 3, 4, 5):
            def mk_iter_cross(s=steps):
                def fn(g):
                    out = g.copy()
                    for _ in range(s):
                        prev = out.copy()
                        h,w=out.shape
                        for r in range(h):
                            for c in range(w):
                                if prev[r,c]!=0:
                                    col=int(prev[r,c])
                                    for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                                        nr,nc=r+dr,c+dc
                                        if 0<=nr<h and 0<=nc<w and out[nr,nc]==0: out[nr,nc]=col
                    return out
                return fn
            cands.append(mk_iter_cross())
        return cands

    def _neighbor_count_recolor(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        
        for nbr_dirs in [
            [(-1,0),(1,0),(0,-1),(0,1)],
            [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        ]:
            mapping = {}; consistent = True
            for inp, out in train:
                if inp.shape != out.shape: consistent = False; break
                h, w = inp.shape
                for r in range(h):
                    for c in range(w):
                        cnt = 0
                        for dr, dc in nbr_dirs:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < h and 0 <= nc < w and inp[nr, nc] != 0: cnt += 1
                        key = (int(inp[r, c]), cnt)
                        oc = int(out[r, c])
                        if key in mapping and mapping[key] != oc:
                            consistent = False; break
                        mapping[key] = oc
                    if not consistent: break
                if not consistent: break
            if consistent and mapping:
                def mk(m=mapping.copy(), dirs=nbr_dirs[:]):
                    def fn(g):
                        h, w = g.shape; out = np.zeros_like(g)
                        for r in range(h):
                            for c in range(w):
                                cnt = 0
                                for dr, dc in dirs:
                                    nr, nc = r+dr, c+dc
                                    if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != 0: cnt += 1
                                key = (int(g[r, c]), cnt)
                                out[r, c] = m.get(key, int(g[r, c]))
                        return out
                    return fn
                cands.append(mk())
        return cands

    def _border_recolor(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        diff_mask = inp0 != out0
        if not np.any(diff_mask): return cands
        new_colors = set(map(int, np.unique(out0[diff_mask])))
        for nc in new_colors:
            def mk(new_c=nc):
                def fn(g):
                    h, w = g.shape; out = g.copy()
                    for r in range(h):
                        for c in range(w):
                            if g[r, c] != 0:
                                is_border = any(
                                    r+dr<0 or r+dr>=h or c+dc<0 or c+dc>=w or g[r+dr, c+dc]==0
                                    for dr,dc in ((-1,0),(1,0),(0,-1),(0,1))
                                )
                                if is_border: out[r, c] = new_c
                    return out
                return fn
            cands.append(mk())
            
            # Interior recolor (opposite of border)
            def mk_int(new_c=nc):
                def fn(g):
                    h, w = g.shape; out = g.copy()
                    for r in range(h):
                        for c in range(w):
                            if g[r, c] != 0:
                                is_interior = all(
                                    0<=r+dr<h and 0<=c+dc<w and g[r+dr, c+dc]!=0
                                    for dr,dc in ((-1,0),(1,0),(0,-1),(0,1))
                                )
                                if is_interior: out[r, c] = new_c
                    return out
                return fn
            cands.append(mk_int())
        return cands

    def _replace_bg_around_objects(self, train) -> list[Prog]:
        cands: list[Prog] = []
        def fill_between_h(g):
            h,w = g.shape; out = g.copy()
            for r in range(h):
                for cl in np.unique(g[r,:]):
                    if cl == 0: continue
                    cols = np.where(g[r,:] == cl)[0]
                    if len(cols) >= 2: out[r, cols[0]:cols[-1]+1] = cl
            return out
        cands.append(fill_between_h)
        
        def fill_between_v(g):
            h,w = g.shape; out = g.copy()
            for c in range(w):
                for cl in np.unique(g[:,c]):
                    if cl == 0: continue
                    rows = np.where(g[:,c] == cl)[0]
                    if len(rows) >= 2: out[rows[0]:rows[-1]+1, c] = cl
            return out
        cands.append(fill_between_v)
        
        # Both H and V
        def fill_between_hv(g):
            out = fill_between_h(g)
            return fill_between_v(out)
        cands.append(fill_between_hv)
        
        return cands

    # ============================================================
    # 20. PANEL & COLOR ANALYSIS
    # ============================================================
    def _panel_majority_threshold(self, train) -> list[Prog]:
        cands: list[Prog] = []
        for dc in range(10):
            def mk(d=dc):
                def fn(g):
                    h, w = g.shape
                    dr = [r for r in range(h) if np.all(g[r,:]==d)]
                    dcc = [c for c in range(w) if np.all(g[:,c]==d)]
                    rs = [-1]+dr+[h]; cs_list = [-1]+dcc+[w]
                    out = g.copy()
                    for i in range(len(rs)-1):
                        r1, r2 = rs[i]+1, rs[i+1]
                        for j in range(len(cs_list)-1):
                            c1, c2 = cs_list[j]+1, cs_list[j+1]
                            if r2>r1 and c2>c1:
                                panel = g[r1:r2, c1:c2]
                                nz = panel[(panel != 0) & (panel != d)]
                                if len(nz) > 0:
                                    cnt = Counter(nz.flatten())
                                    top_c = cnt.most_common(1)[0][0]
                                    out[r1:r2, c1:c2] = top_c
                                else:
                                    out[r1:r2, c1:c2] = 0
                    return out
                return fn
            cands.append(mk())
        return cands

    def _deduce_output_from_panels(self, train) -> list[Prog]:
        cands: list[Prog] = []
        for dc in range(10):
            def mk_diff(d=dc):
                def fn(g):
                    ps = split_panels(g, d)
                    if len(ps) != 2 or ps[0].shape != ps[1].shape: return None
                    diff = (ps[0] != ps[1])
                    out = np.zeros_like(ps[0]); out[diff] = ps[0][diff]; return out
                return fn
            cands.append(mk_diff())
            
            def mk_diff2(d=dc):
                def fn(g):
                    ps = split_panels(g, d)
                    if len(ps) != 2 or ps[0].shape != ps[1].shape: return None
                    diff = (ps[0] != ps[1])
                    out = np.zeros_like(ps[1]); out[diff] = ps[1][diff]; return out
                return fn
            cands.append(mk_diff2())
        return cands

    def _invert_colors(self, train) -> list[Prog]:
        cands: list[Prog] = []
        for c in range(1, 10):
            def mk_swap(col=c):
                def fn(g):
                    out = g.copy(); out[g == 0] = col; out[g == col] = 0; return out
                return fn
            cands.append(mk_swap())
        return cands

    def _sort_rows_cols(self, train) -> list[Prog]:
        cands: list[Prog] = []
        for rev in (False, True):
            def mk_sr(r=rev):
                def fn(g):
                    rows = sorted(range(g.shape[0]), key=lambda r_: np.count_nonzero(g[r_,:]), reverse=r)
                    return g[rows, :]
                return fn
            cands.append(mk_sr())
        
        for rev in (False, True):
            def mk_sc(r=rev):
                def fn(g):
                    cols = sorted(range(g.shape[1]), key=lambda c: np.count_nonzero(g[:,c]), reverse=r)
                    return g[:, cols]
                return fn
            cands.append(mk_sc())
        return cands

    def _majority_per_object(self, train) -> list[Prog]:
        cands: list[Prog] = []
        for conn in (4, 8):
            def mk(c=conn):
                def fn(g):
                    out = g.copy()
                    for o in get_objects_multi(g, conn=c):
                        maj = Counter(int(g[r,cc]) for r,cc in o['cells']).most_common(1)[0][0]
                        for r,cc in o['cells']: out[r,cc] = maj
                    return out
                return fn
            cands.append(mk())
        return cands

    def _extract_repeated_tile(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        ih, iw = inp0.shape; oh, ow = out0.shape
        if oh < ih or ow < iw:
            for th in range(1, ih+1):
                for tw in range(1, iw+1):
                    if ih % th == 0 and iw % tw == 0 and th == oh and tw == ow:
                        tile = inp0[:th, :tw]
                        if np.array_equal(np.tile(tile, (ih//th, iw//tw)), inp0):
                            def mk(t_h=th, t_w=tw): return lambda g: g[:t_h, :t_w]
                            cands.append(mk())
        return cands

    # ============================================================
    # 21. PER-COLOR SHAPE STAMP
    # ============================================================
    def _per_color_shape_stamp(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        
        colors = sorted(set(map(int, np.unique(inp0))) - {0})
        if len(colors) < 1 or len(colors) > 5: return cands
        
        stamps = {}
        for col in colors:
            pts = list(zip(*np.where(inp0 == col)))
            if not pts: continue
            for rad in (1, 2, 3):
                valid = True
                patches = []
                for r, c in pts:
                    r1, r2 = r-rad, r+rad+1
                    c1, c2 = c-rad, c+rad+1
                    if r1 < 0 or r2 > inp0.shape[0] or c1 < 0 or c2 > inp0.shape[1]:
                        valid = False; break
                    patches.append(out0[r1:r2, c1:c2].copy())
                if valid and patches:
                    if all(np.array_equal(patches[0], p) for p in patches):
                        stamps[col] = (rad, patches[0].copy())
                        break
        
        if stamps:
            ok = True
            for inp, out in train[1:]:
                for col, (rad, stamp) in stamps.items():
                    pts = list(zip(*np.where(inp == col)))
                    for r, c in pts:
                        r1, r2 = r-rad, r+rad+1
                        c1, c2 = c-rad, c+rad+1
                        if r1 < 0 or r2 > inp.shape[0] or c1 < 0 or c2 > inp.shape[1]:
                            ok = False; break
                        if not np.array_equal(out[r1:r2, c1:c2], stamp):
                            ok = False; break
                    if not ok: break
                if not ok: break
            
            if ok:
                def mk_preserve(st=dict(stamps)):
                    def fn(g):
                        h, w = g.shape; out = g.copy()
                        for col, (rad, stamp) in st.items():
                            pts = list(zip(*np.where(g == col)))
                            for r, c in pts:
                                r1, r2 = r-rad, r+rad+1
                                c1, c2 = c-rad, c+rad+1
                                if 0 <= r1 and r2 <= h and 0 <= c1 and c2 <= w:
                                    for dr in range(2*rad+1):
                                        for dc in range(2*rad+1):
                                            if stamp[dr, dc] != 0:
                                                out[r1+dr, c1+dc] = stamp[dr, dc]
                        return out
                    return fn
                cands.append(mk_preserve())
        return cands

    # ============================================================
    # 22. MULTI-COLOR OBJECT STAMP
    # ============================================================
    def _multi_color_object_stamp(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        
        for conn in (4, 8):
            objs = get_objects(inp0, conn=conn, mono=False)
            if len(objs) < 1 or len(objs) > 10: continue
            
            norm_shapes = set()
            for o in objs:
                mask = o['mask']
                color_order = []
                norm = np.zeros_like(mask)
                for r in range(mask.shape[0]):
                    for c in range(mask.shape[1]):
                        v = int(mask[r, c])
                        if v == 0: continue
                        if v not in color_order: color_order.append(v)
                        norm[r, c] = color_order.index(v) + 1
                norm_shapes.add(tuple(norm.flatten()))
            if len(norm_shapes) != 1: continue
            
            o0 = objs[0]
            mask0 = o0['mask']
            color_roles = []
            for r in range(mask0.shape[0]):
                for c in range(mask0.shape[1]):
                    v = int(mask0[r, c])
                    if v != 0 and v not in color_roles: color_roles.append(v)
            if len(color_roles) < 1: continue
            
            for rad in (2, 3, 4):
                cr = (o0['bbox'][0] + o0['bbox'][2]) // 2
                cc = (o0['bbox'][1] + o0['bbox'][3]) // 2
                r1, r2 = cr-rad, cr+rad+1
                c1, c2 = cc-rad, cc+rad+1
                if r1 < 0 or r2 > inp0.shape[0] or c1 < 0 or c2 > inp0.shape[1]: continue
                patch = out0[r1:r2, c1:c2].copy()
                
                template = np.zeros_like(patch)
                bad = False
                for r in range(patch.shape[0]):
                    for c in range(patch.shape[1]):
                        v = int(patch[r, c])
                        if v == 0: continue
                        if v in color_roles:
                            template[r, c] = color_roles.index(v) + 1
                        else:
                            bad = True; break
                    if bad: break
                if bad: continue
                
                ok = True
                for inp, out in train:
                    objs2 = get_objects(inp, conn=conn, mono=False)
                    for o2 in objs2:
                        m2 = o2['mask']
                        roles2 = []
                        for rr in range(m2.shape[0]):
                            for cc2 in range(m2.shape[1]):
                                v = int(m2[rr, cc2])
                                if v != 0 and v not in roles2: roles2.append(v)
                        if len(roles2) != len(color_roles): ok = False; break
                        
                        cr2 = (o2['bbox'][0] + o2['bbox'][2]) // 2
                        cc2_c = (o2['bbox'][1] + o2['bbox'][3]) // 2
                        rr1, rr2 = cr2-rad, cr2+rad+1
                        cc1, cc2b = cc2_c-rad, cc2_c+rad+1
                        if rr1 < 0 or rr2 > inp.shape[0] or cc1 < 0 or cc2b > inp.shape[1]:
                            ok = False; break
                        
                        actual = out[rr1:rr2, cc1:cc2b]
                        expected = np.zeros_like(template)
                        for tr in range(template.shape[0]):
                            for tc in range(template.shape[1]):
                                if template[tr, tc] > 0:
                                    expected[tr, tc] = roles2[int(template[tr, tc]) - 1]
                        if not np.array_equal(actual, expected): ok = False; break
                    if not ok: break
                
                if ok:
                    def mk(cn=conn, rd=rad, tmpl=template.copy(), n_roles=len(color_roles)):
                        def fn(g):
                            h, w = g.shape; out = g.copy()
                            objs3 = get_objects(g, conn=cn, mono=False)
                            for o3 in objs3:
                                m3 = o3['mask']
                                roles3 = []
                                for rr in range(m3.shape[0]):
                                    for cc3 in range(m3.shape[1]):
                                        v = int(m3[rr, cc3])
                                        if v != 0 and v not in roles3: roles3.append(v)
                                if len(roles3) != n_roles: continue
                                cr3 = (o3['bbox'][0] + o3['bbox'][2]) // 2
                                cc3 = (o3['bbox'][1] + o3['bbox'][3]) // 2
                                rr1, rr2 = cr3-rd, cr3+rd+1
                                cc1, cc2 = cc3-rd, cc3+rd+1
                                if 0 <= rr1 and rr2 <= h and 0 <= cc1 and cc2 <= w:
                                    for tr in range(tmpl.shape[0]):
                                        for tc in range(tmpl.shape[1]):
                                            if tmpl[tr, tc] > 0:
                                                out[rr1+tr, cc1+tc] = roles3[int(tmpl[tr, tc]) - 1]
                            return out
                        return fn
                    cands.append(mk())
                    break
        return cands

    # ============================================================
    # 23. ROW×COLUMN INTERSECTION
    # ============================================================
    def _row_col_intersection(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        h, w = inp0.shape
        
        colors = sorted(set(map(int, np.unique(inp0))) - {0})
        diff = (inp0 != out0)
        if not np.any(diff): return cands
        fill_colors = set(map(int, np.unique(out0[diff]))) - {0}
        
        for anchor_c in colors:
            for fill_c in fill_colors:
                # Row×Col fill
                def mk2(ac=anchor_c, fc=fill_c):
                    def fn(g):
                        h, w = g.shape; out = g.copy()
                        a_rows = set(); a_cols = set()
                        for r in range(h):
                            for c in range(w):
                                if g[r, c] == ac: a_rows.add(r); a_cols.add(c)
                        for r in a_rows:
                            for c in a_cols:
                                if g[r, c] == 0: out[r, c] = fc
                        return out
                    return fn
                cands.append(mk2())
                
                # Full row/col fill
                def mk_row(ac=anchor_c, fc=fill_c):
                    def fn(g):
                        h, w = g.shape; out = g.copy()
                        for r in range(h):
                            for c in range(w):
                                if g[r, c] == ac:
                                    for cc in range(w):
                                        if out[r, cc] == 0: out[r, cc] = fc
                        return out
                    return fn
                cands.append(mk_row())
                
                def mk_col(ac=anchor_c, fc=fill_c):
                    def fn(g):
                        h, w = g.shape; out = g.copy()
                        for r in range(h):
                            for c in range(w):
                                if g[r, c] == ac:
                                    for rr in range(h):
                                        if out[rr, c] == 0: out[rr, c] = fc
                        return out
                    return fn
                cands.append(mk_col())
        return cands

    # ============================================================
    # 24. DIRECTIONAL TRAIL
    # ============================================================
    def _directional_trail(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        
        colors = sorted(set(map(int, np.unique(inp0))) - {0})
        if len(colors) != 2: return cands
        
        for main_c in colors:
            dir_c = [c for c in colors if c != main_c][0]
            main_pts = list(zip(*np.where(inp0 == main_c)))
            dir_pts = list(zip(*np.where(inp0 == dir_c)))
            if not main_pts or len(dir_pts) != 1: continue
            
            def mk(mc_=main_c, dc_=dir_c):
                def fn(g):
                    h, w = g.shape
                    mpts = list(zip(*np.where(g == mc_)))
                    dpts = list(zip(*np.where(g == dc_)))
                    if not mpts or len(dpts) != 1: return None
                    
                    mr = np.mean([r for r, c in mpts])
                    mc = np.mean([c for r, c in mpts])
                    dp = dpts[0]
                    
                    ddr = dp[0] - mr; ddc = dp[1] - mc
                    if abs(ddr) >= abs(ddc):
                        s_dr = 1 if ddr > 0 else -1
                        s_dc = 1 if ddc > 0 else (-1 if ddc < 0 else 0)
                    else:
                        s_dc = 1 if ddc > 0 else -1
                        s_dr = 1 if ddr > 0 else (-1 if ddr < 0 else 0)
                    
                    out = np.zeros_like(g)
                    for r, c in mpts: out[r, c] = mc_
                    
                    step = 1
                    while True:
                        any_placed = False
                        for r, c in mpts:
                            nr, nc = r + step*s_dr, c + step*s_dc
                            if 0 <= nr < h and 0 <= nc < w:
                                out[nr, nc] = mc_; any_placed = True
                        if not any_placed: break
                        step += 1
                        if step > max(h, w): break
                    return out
                return fn
            cands.append(mk())
        return cands

    # ============================================================
    # 25. CROP AND TILE
    # ============================================================
    def _crop_and_tile(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        oh, ow = out0.shape
        
        rows, cols = np.where(inp0 != 0)
        if len(rows) == 0: return cands
        cropped = inp0[rows.min():rows.max()+1, cols.min():cols.max()+1]
        ch, cw = cropped.shape
        
        for ny in range(1, 6):
            for nx in range(1, 6):
                if ch * ny == oh and cw * nx == ow:
                    tiled = np.tile(cropped, (ny, nx))
                    if np.array_equal(tiled, out0):
                        def mk(r_ny=ny, r_nx=nx):
                            def fn(g):
                                rs, cs = np.where(g != 0)
                                if len(rs) == 0: return None
                                sub = g[rs.min():rs.max()+1, cs.min():cs.max()+1]
                                return np.tile(sub, (r_ny, r_nx))
                            return fn
                        cands.append(mk())
        return cands

    # ============================================================
    # 26. PANEL DIMENSION COUNT
    # ============================================================
    def _panel_dimension_count(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        oh, ow = out0.shape
        
        for dc in range(10):
            h, w = inp0.shape
            dr = [r for r in range(h) if np.all(inp0[r,:]==dc)]
            dcc = [c for c in range(w) if np.all(inp0[:,c]==dc)]
            n_row_panels = len(dr) + 1
            n_col_panels = len(dcc) + 1
            
            if n_row_panels >= 2 and n_col_panels >= 2:
                if oh == n_row_panels and ow == n_col_panels:
                    # Each output cell = some property of corresponding panel
                    ps = split_panels(inp0, dc)
                    if len(ps) == n_row_panels * n_col_panels:
                        # Try: output cell = count of non-bg, non-divider colors
                        out_vals = []
                        for p in ps:
                            nz = p[(p != 0) & (p != dc)]
                            if len(nz) > 0:
                                out_vals.append(int(Counter(nz.flatten()).most_common(1)[0][0]))
                            else:
                                out_vals.append(0)
                        expected = np.array(out_vals, dtype=np.int32).reshape(n_row_panels, n_col_panels)
                        if np.array_equal(expected, out0):
                            def mk(d=dc):
                                def fn(g):
                                    h2, w2 = g.shape
                                    dr2 = [r for r in range(h2) if np.all(g[r,:]==d)]
                                    dcc2 = [c for c in range(w2) if np.all(g[:,c]==d)]
                                    nr = len(dr2) + 1; nc = len(dcc2) + 1
                                    ps2 = split_panels(g, d)
                                    if len(ps2) != nr * nc: return None
                                    out_v = []
                                    for p in ps2:
                                        nz = p[(p != 0) & (p != d)]
                                        if len(nz) > 0:
                                            out_v.append(int(Counter(nz.flatten()).most_common(1)[0][0]))
                                        else:
                                            out_v.append(0)
                                    return np.array(out_v, dtype=np.int32).reshape(nr, nc)
                                return fn
                            cands.append(mk())
        return cands

    # ============================================================
    # 27. ROW EXTENSION WITH COLOR SUB
    # ============================================================
    def _row_extension_with_color_sub(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        ih, iw = inp0.shape; oh, ow = out0.shape
        if iw != ow or oh <= ih: return cands
        
        # Check if output is input vertically tiled
        for ny in range(2, 5):
            if oh == ih * ny:
                # Check if output = tile(color_mapped(input), ny, 1)
                mapping = {}; ok = True
                for r in range(ih):
                    for c in range(iw):
                        ci = int(inp0[r, c])
                        co = int(out0[r, c])
                        if ci in mapping and mapping[ci] != co: ok = False; break
                        mapping[ci] = co
                    if not ok: break
                if ok and mapping:
                    mapped = inp0.copy()
                    for k, v in mapping.items(): mapped[inp0 == k] = v
                    tiled = np.tile(mapped, (ny, 1))
                    if np.array_equal(tiled, out0):
                        def mk(m=mapping.copy(), n=ny):
                            def fn(g):
                                out = g.copy()
                                for k, v in m.items(): out[g == k] = v
                                return np.tile(out, (n, 1))
                            return fn
                        cands.append(mk())
        return cands

    # ============================================================
    # 28. SPIRAL FILL
    # ============================================================
    def _spiral_fill(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        
        fill_colors = sorted(set(map(int, np.unique(out0))) - {0})
        if len(fill_colors) != 1: return cands
        fc = fill_colors[0]
        
        def mk(fill_c=fc):
            def fn(g):
                h, w = g.shape
                out = np.zeros_like(g)
                r1, r2, c1, c2 = 0, h-1, 0, w-1
                ring = 0
                while r1 <= r2 and c1 <= c2:
                    col = fill_c if ring % 2 == 0 else 0
                    for c in range(c1, c2+1): out[r1, c] = col
                    for r in range(r1+1, r2+1): out[r, c2] = col
                    if r1 < r2:
                        for c in range(c2-1, c1-1, -1): out[r2, c] = col
                    if c1 < c2:
                        for r in range(r2-1, r1, -1): out[r, c1] = col
                    r1 += 1; r2 -= 1; c1 += 1; c2 -= 1
                    ring += 1
                return out
            return fn
        cands.append(mk())
        return cands

    # ============================================================
    # 29. CROSS-LINE MARKERS
    # ============================================================
    def _cross_line_markers(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        
        for line_mode in ("full_cross", "row_only", "col_only"):
            def mk(mode=line_mode):
                def fn(g):
                    h, w = g.shape; out = np.zeros_like(g)
                    for r in range(h):
                        for c in range(w):
                            if g[r, c] != 0:
                                col = int(g[r, c])
                                if mode in ("full_cross", "row_only"):
                                    out[r, :] = np.where(out[r, :] == 0, col, out[r, :])
                                if mode in ("full_cross", "col_only"):
                                    out[:, c] = np.where(out[:, c] == 0, col, out[:, c])
                    return out
                return fn
            cands.append(mk())
            
            # Preserve original grid, add lines
            def mk2(mode=line_mode):
                def fn(g):
                    h, w = g.shape; out = g.copy()
                    for r in range(h):
                        for c in range(w):
                            if g[r, c] != 0:
                                col = int(g[r, c])
                                if mode in ("full_cross", "row_only"):
                                    for cc in range(w):
                                        if out[r, cc] == 0: out[r, cc] = col
                                if mode in ("full_cross", "col_only"):
                                    for rr in range(h):
                                        if out[rr, c] == 0: out[rr, c] = col
                    return out
                return fn
            cands.append(mk2())
        return cands

    # ============================================================
    # 30. OBJECT SYMMETRY FILL
    # ============================================================
    def _object_symmetry_fill(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        
        diff = (inp0 != out0)
        if not np.any(diff): return cands
        new_colors = set(map(int, np.unique(out0[diff])))
        
        for conn in (4, 8):
            objs = get_objects(inp0, conn=conn)
            if len(objs) != 1: continue
            o = objs[0]
            
            for new_c in new_colors:
                for axis in ("h", "v"):
                    def mk(cn=conn, nc=new_c, ax=axis):
                        def fn(g):
                            h, w = g.shape; out = g.copy()
                            objs2 = get_objects(g, conn=cn)
                            if len(objs2) != 1: return None
                            o2 = objs2[0]
                            cells2 = set(o2['cells'])
                            mr2, mc2, Mr2, Mc2 = o2['bbox']
                            
                            if ax == "v":
                                ccenter = (mc2 + Mc2) / 2.0
                                for r, c in list(cells2):
                                    mirror_c = int(round(2 * ccenter - c))
                                    if 0 <= mirror_c < w and (r, mirror_c) not in cells2:
                                        out[r, mirror_c] = nc
                            else:
                                rcenter = (mr2 + Mr2) / 2.0
                                for r, c in list(cells2):
                                    mirror_r = int(round(2 * rcenter - r))
                                    if 0 <= mirror_r < h and (mirror_r, c) not in cells2:
                                        out[mirror_r, c] = nc
                            return out
                        return fn
                    cands.append(mk())
        return cands

    # ============================================================
    # 31. PIXEL POSITION RULE
    # ============================================================
    def _pixel_position_rule(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        h, w = inp0.shape
        
        for rmod in range(1, min(h+1, 6)):
            for cmod in range(1, min(w+1, 6)):
                if rmod == 1 and cmod == 1: continue  # Same as palette
                mapping = {}; ok = True
                for inp, out in train:
                    if inp.shape != out.shape: ok = False; break
                    for r in range(inp.shape[0]):
                        for c in range(inp.shape[1]):
                            key = (r % rmod, c % cmod, int(inp[r,c]))
                            val = int(out[r,c])
                            if key in mapping and mapping[key] != val:
                                ok = False; break
                            mapping[key] = val
                        if not ok: break
                    if not ok: break
                if ok and mapping:
                    def mk(m=mapping.copy(), rm=rmod, cm=cmod):
                        def fn(g):
                            h, w = g.shape; out = np.zeros_like(g)
                            for r in range(h):
                                for c in range(w):
                                    key = (r % rm, c % cm, int(g[r,c]))
                                    out[r,c] = m.get(key, int(g[r,c]))
                            return out
                        return fn
                    cands.append(mk())
        return cands

    # ============================================================
    # 32. MOST COMMON OBJECT
    # ============================================================
    def _most_common_object(self, train) -> list[Prog]:
        cands: list[Prog] = []
        for conn in (4, 8):
            def mk(c=conn):
                def fn(g):
                    objs = get_objects(g, conn=c)
                    if not objs: return None
                    shapes = {}
                    for o in objs:
                        key = (o['h'], o['w'], tuple(o['mask'].flatten()))
                        if key not in shapes: shapes[key] = []
                        shapes[key].append(o)
                    most_common = max(shapes.values(), key=len)
                    if len(most_common) > 1:
                        return most_common[0]['mask']
                    return None
                return fn
            cands.append(mk())
        return cands

    # ============================================================
    # 33. PERIODIC FILL
    # ============================================================
    def _periodic_fill(self, train) -> list[Prog]:
        cands: list[Prog] = []
        
        # Fill between same-color points on same row
        def row_fill(g):
            h, w = g.shape; out = g.copy()
            for r in range(h):
                nz = [(c, int(g[r,c])) for c in range(w) if g[r,c] != 0]
                if len(nz) >= 2:
                    for i in range(len(nz)-1):
                        c1, col1 = nz[i]
                        c2, col2 = nz[i+1]
                        if col1 == col2:
                            out[r, c1:c2+1] = col1
            return out
        cands.append(row_fill)
        
        # Fill between same-color points on same col
        def col_fill(g):
            h, w = g.shape; out = g.copy()
            for c in range(w):
                nz = [(r, int(g[r,c])) for r in range(h) if g[r,c] != 0]
                if len(nz) >= 2:
                    for i in range(len(nz)-1):
                        r1, col1 = nz[i]
                        r2, col2 = nz[i+1]
                        if col1 == col2:
                            out[r1:r2+1, c] = col1
            return out
        cands.append(col_fill)
        
        # Both
        def both_fill(g):
            return col_fill(row_fill(g))
        cands.append(both_fill)
        
        # Fill between ANY consecutive non-zero on same row
        def row_fill_any(g):
            h, w = g.shape; out = g.copy()
            for r in range(h):
                nz = [(c, int(g[r,c])) for c in range(w) if g[r,c] != 0]
                if len(nz) >= 2:
                    for i in range(len(nz)-1):
                        c1, col1 = nz[i]
                        c2, col2 = nz[i+1]
                        for cc in range(c1+1, c2):
                            if out[r, cc] == 0: out[r, cc] = col1
            return out
        cands.append(row_fill_any)
        
        def col_fill_any(g):
            h, w = g.shape; out = g.copy()
            for c in range(w):
                nz = [(r, int(g[r,c])) for r in range(h) if g[r,c] != 0]
                if len(nz) >= 2:
                    for i in range(len(nz)-1):
                        r1, col1 = nz[i]
                        r2, col2 = nz[i+1]
                        for rr in range(r1+1, r2):
                            if out[rr, c] == 0: out[rr, c] = col1
            return out
        cands.append(col_fill_any)
        
        return cands

    # ============================================================
    # 34. OBJECT PAIR REFLECTION
    # ============================================================
    def _object_pair_reflection(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        
        diff = (inp0 != out0)
        if not np.any(diff): return cands
        new_colors = set(map(int, np.unique(out0[diff]))) - set(map(int, np.unique(inp0)))
        if not new_colors: return cands
        
        for marker_c in new_colors:
            for conn in (4, 8):
                def mk(mc=marker_c, cn=conn):
                    def fn(g):
                        h, w = g.shape; out = g.copy()
                        objs = get_objects(g, conn=cn)
                        shape_groups = {}
                        for o in objs:
                            key = (o['h'], o['w'], tuple(o['mask'].flatten()))
                            if key not in shape_groups: shape_groups[key] = []
                            shape_groups[key].append(o)
                        
                        for key, group in shape_groups.items():
                            if len(group) == 2:
                                o1, o2 = group
                                dr = (o2['bbox'][0] + o2['bbox'][2]) / 2 - (o1['bbox'][0] + o1['bbox'][2]) / 2
                                dc = (o2['bbox'][1] + o2['bbox'][3]) / 2 - (o1['bbox'][1] + o1['bbox'][3]) / 2
                                
                                for r, c in o1['cells']:
                                    nr = int(round(r + 2*dr))
                                    nc = int(round(c + 2*dc))
                                    if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == 0:
                                        out[nr, nc] = mc
                                for r, c in o2['cells']:
                                    nr = int(round(r - 2*dr))
                                    nc = int(round(c - 2*dc))
                                    if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == 0:
                                        out[nr, nc] = mc
                        return out
                    return fn
                cands.append(mk())
        return cands

    # ============================================================
    # 35. COLOR COUNTING OUTPUT
    # ============================================================
    def _color_counting_output(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        oh, ow = out0.shape
        
        if oh == 1 and ow == 1:
            target = int(out0[0, 0])
            colors = sorted(set(map(int, np.unique(inp0))) - {0})
            cnt = Counter(inp0[inp0 != 0].flatten())
            if cnt:
                most = int(cnt.most_common(1)[0][0])
                least = int(cnt.most_common()[-1][0])
                if most == target:
                    cands.append(lambda g: np.array([[int(Counter(g[g!=0].flatten()).most_common(1)[0][0])]], dtype=np.int32) if np.any(g!=0) else None)
                if least == target:
                    cands.append(lambda g: np.array([[int(Counter(g[g!=0].flatten()).most_common()[-1][0])]], dtype=np.int32) if np.any(g!=0) else None)
            
            n_colors = len(colors)
            if n_colors == target:
                cands.append(lambda g: np.array([[len(set(map(int, np.unique(g))) - {0})]], dtype=np.int32))
            
            # Count of objects
            for conn in (4, 8):
                n_objs = len(get_objects(inp0, conn=conn))
                if n_objs == target:
                    def mk_count(cn=conn):
                        def fn(g):
                            n = len(get_objects(g, conn=cn))
                            return np.array([[n]], dtype=np.int32)
                        return fn
                    cands.append(mk_count())
        
        return cands

    # ============================================================
    # 36. OBJECT RELATIVE MARKERS (stub → improved)
    # ============================================================
    def _object_relative_markers(self, train) -> list[Prog]:
        return []  # Covered by other stamp functions

    # ============================================================
    # 37. SUBGRID MAJORITY VOTE
    # ============================================================
    def _subgrid_majority(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        ih, iw = inp0.shape; oh, ow = out0.shape
        if oh >= ih or ow >= iw: return cands
        if ih % oh != 0 or iw % ow != 0: return cands
        sy, sx = ih // oh, iw // ow
        
        def mk(y=sy, x=sx):
            def fn(g):
                h, w = g.shape
                if h % y != 0 or w % x != 0: return None
                rh, rw = h // y, w // x
                out = np.zeros((rh, rw), dtype=np.int32)
                for r in range(rh):
                    for c in range(rw):
                        blk = g[r*y:(r+1)*y, c*x:(c+1)*x]
                        vals, cnts = np.unique(blk, return_counts=True)
                        out[r, c] = vals[np.argmax(cnts)]
                return out
            return fn
        cands.append(mk())
        
        def mk_nz(y=sy, x=sx):
            def fn(g):
                h, w = g.shape
                if h % y != 0 or w % x != 0: return None
                rh, rw = h // y, w // x
                out = np.zeros((rh, rw), dtype=np.int32)
                for r in range(rh):
                    for c in range(rw):
                        blk = g[r*y:(r+1)*y, c*x:(c+1)*x]
                        nz = blk[blk != 0]
                        if len(nz) > 0:
                            vals, cnts = np.unique(nz, return_counts=True)
                            out[r, c] = vals[np.argmax(cnts)]
                return out
            return fn
        cands.append(mk_nz())
        return cands

    # ============================================================
    # 38. DIAGONAL MIRROR
    # ============================================================
    def _diagonal_mirror(self, train) -> list[Prog]:
        cands: list[Prog] = []
        def diag_mirror(g):
            h, w = g.shape
            if h != w: return None
            out = g.copy()
            for r in range(h):
                for c in range(w):
                    if out[r, c] == 0 and g[c, r] != 0: out[r, c] = g[c, r]
            return out
        cands.append(diag_mirror)
        
        def anti_diag_mirror(g):
            h, w = g.shape
            if h != w: return None
            out = g.copy()
            for r in range(h):
                for c in range(w):
                    mr, mc = h-1-c, w-1-r
                    if out[r, c] == 0 and g[mr, mc] != 0: out[r, c] = g[mr, mc]
            return out
        cands.append(anti_diag_mirror)
        return cands

    # ============================================================
    # 39. EXTENDED NEIGHBORHOOD RULE
    # ============================================================
    def _extended_neighborhood_rule(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        
        def compute_features(g, r, c):
            h, w = g.shape
            self_c = int(g[r, c])
            same = 0; diff = 0
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w:
                    if g[nr, nc] == self_c: same += 1
                    elif g[nr, nc] != 0: diff += 1
            return (self_c, same, diff)
        
        mapping = {}; ok = True
        for inp, out in train:
            if inp.shape != out.shape: ok = False; break
            h, w = inp.shape
            for r in range(h):
                for c in range(w):
                    key = compute_features(inp, r, c)
                    val = int(out[r, c])
                    if key in mapping and mapping[key] != val: ok = False; break
                    mapping[key] = val
                if not ok: break
            if not ok: break
        
        if ok and mapping:
            def mk(m=mapping.copy()):
                def fn(g):
                    h, w = g.shape; out = np.zeros_like(g)
                    for r in range(h):
                        for c in range(w):
                            self_c = int(g[r, c])
                            same = 0; diff = 0
                            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                                nr, nc = r+dr, c+dc
                                if 0 <= nr < h and 0 <= nc < w:
                                    if g[nr, nc] == self_c: same += 1
                                    elif g[nr, nc] != 0: diff += 1
                            key = (self_c, same, diff)
                            out[r, c] = m.get(key, self_c)
                    return out
                return fn
            cands.append(mk())
        return cands

    # ============================================================
    # 40. CONDITIONAL PIXEL TRANSFORM
    # ============================================================
    def _conditional_pixel_transform(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        
        mapping = {}; ok = True
        for inp, out in train:
            if inp.shape != out.shape: ok = False; break
            h, w = inp.shape
            for r in range(h):
                for c in range(w):
                    self_c = int(inp[r, c])
                    on_border = (r == 0 or r == h-1 or c == 0 or c == w-1)
                    adj_diff = False
                    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w and inp[nr, nc] != self_c:
                            adj_diff = True; break
                    key = (self_c, on_border, adj_diff)
                    val = int(out[r, c])
                    if key in mapping and mapping[key] != val: ok = False; break
                    mapping[key] = val
                if not ok: break
            if not ok: break
        
        if ok and mapping:
            def mk(m=mapping.copy()):
                def fn(g):
                    h, w = g.shape; out = np.zeros_like(g)
                    for r in range(h):
                        for c in range(w):
                            self_c = int(g[r, c])
                            on_border = (r == 0 or r == h-1 or c == 0 or c == w-1)
                            adj_diff = False
                            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                                nr, nc = r+dr, c+dc
                                if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != self_c:
                                    adj_diff = True; break
                            key = (self_c, on_border, adj_diff)
                            out[r, c] = m.get(key, self_c)
                    return out
                return fn
            cands.append(mk())
        return cands

    # ============================================================
    # 41. FLOOD FILL PER OBJECT
    # ============================================================
    def _flood_fill_per_object(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        
        for conn in (4, 8):
            def mk(cn=conn):
                def fn(g):
                    h, w = g.shape; out = g.copy()
                    objs = get_objects(g, conn=cn)
                    for o in objs:
                        mr, mc, Mr, Mc = o['bbox']
                        bh, bw = Mr-mr+1, Mc-mc+1
                        sub = g[mr:Mr+1, mc:Mc+1]
                        vis = np.zeros((bh, bw), dtype=bool)
                        stk = []
                        for r in range(bh):
                            for c in (0, bw-1):
                                if sub[r, c] == 0 and not vis[r, c]:
                                    vis[r, c] = True; stk.append((r, c))
                        for c in range(bw):
                            for r in (0, bh-1):
                                if sub[r, c] == 0 and not vis[r, c]:
                                    vis[r, c] = True; stk.append((r, c))
                        while stk:
                            r, c = stk.pop()
                            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                                nr, nc = r+dr, c+dc
                                if 0<=nr<bh and 0<=nc<bw and sub[nr, nc]==0 and not vis[nr, nc]:
                                    vis[nr, nc] = True; stk.append((nr, nc))
                        for r in range(bh):
                            for c in range(bw):
                                if sub[r, c] == 0 and not vis[r, c]:
                                    out[mr+r, mc+c] = o['color']
                    return out
                return fn
            cands.append(mk())
        return cands

    # ============================================================
    # 42. OBJECT SORT AND STACK
    # ============================================================
    def _object_sort_stack(self, train) -> list[Prog]:
        cands: list[Prog] = []
        for conn in (4, 8):
            for sort_key in ("area", "color"):
                for direction in ("v", "h"):
                    def mk(cn=conn, sk=sort_key, d=direction):
                        def fn(g):
                            objs = get_objects(g, conn=cn)
                            if not objs or len(objs) < 2: return None
                            if sk == "area": objs.sort(key=lambda o: o['area'])
                            elif sk == "color": objs.sort(key=lambda o: o['color'])
                            
                            masks = [o['mask'] for o in objs]
                            if d == "v":
                                max_w = max(m.shape[1] for m in masks)
                                padded = []
                                for m in masks:
                                    if m.shape[1] < max_w:
                                        p = np.zeros((m.shape[0], max_w), dtype=np.int32)
                                        p[:, :m.shape[1]] = m; padded.append(p)
                                    else: padded.append(m)
                                return np.vstack(padded)
                            else:
                                max_h = max(m.shape[0] for m in masks)
                                padded = []
                                for m in masks:
                                    if m.shape[0] < max_h:
                                        p = np.zeros((max_h, m.shape[1]), dtype=np.int32)
                                        p[:m.shape[0], :] = m; padded.append(p)
                                    else: padded.append(m)
                                return np.hstack(padded)
                        return fn
                    cands.append(mk())
        return cands

    # ============================================================
    # 43. OUTLINE OBJECTS
    # ============================================================
    def _outline_objects(self, train) -> list[Prog]:
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        
        diff = (inp0 != out0)
        if not np.any(diff): return cands
        new_colors = set(map(int, np.unique(out0[diff])))
        
        for nc in new_colors:
            def mk_interior(new_c=nc):
                def fn(g):
                    h, w = g.shape; out = g.copy()
                    for r in range(h):
                        for c in range(w):
                            if g[r, c] != 0:
                                interior = all(
                                    0<=r+dr<h and 0<=c+dc<w and g[r+dr, c+dc]!=0
                                    for dr,dc in ((-1,0),(1,0),(0,-1),(0,1))
                                )
                                if interior: out[r, c] = new_c
                    return out
                return fn
            cands.append(mk_interior())
        return cands

    # ============================================================
    # 44. COLOR ZONE PROPAGATION (VORONOI)
    # ============================================================
    def _color_zone_propagation(self, train) -> list[Prog]:
        cands: list[Prog] = []
        
        def nearest_color(g):
            h, w = g.shape; out = g.copy()
            q = deque()
            for r in range(h):
                for c in range(w):
                    if g[r, c] != 0:
                        q.append((r, c, int(g[r, c])))
            while q:
                r, c, col = q.popleft()
                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == 0:
                        out[nr, nc] = col
                        q.append((nr, nc, col))
            return out
        cands.append(nearest_color)
        
        # 8-connected voronoi
        def nearest_color_8(g):
            h, w = g.shape; out = g.copy()
            q = deque()
            for r in range(h):
                for c in range(w):
                    if g[r, c] != 0:
                        q.append((r, c, int(g[r, c])))
            while q:
                r, c, col = q.popleft()
                for dr in (-1,0,1):
                    for dc in (-1,0,1):
                        if dr==0 and dc==0: continue
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == 0:
                            out[nr, nc] = col
                            q.append((nr, nc, col))
            return out
        cands.append(nearest_color_8)
        return cands

    # ============================================================
    # 45. ROW/COL DEDUP
    # ============================================================
    def _row_col_dedup(self, train) -> list[Prog]:
        cands: list[Prog] = []
        
        def dedup_rows(g):
            seen = []; result = []
            for r in range(g.shape[0]):
                row = tuple(g[r, :])
                if row not in seen: seen.append(row); result.append(g[r, :])
            return np.array(result, dtype=np.int32) if result else None
        cands.append(dedup_rows)
        
        def dedup_cols(g):
            seen = []; result = []
            for c in range(g.shape[1]):
                col = tuple(g[:, c])
                if col not in seen: seen.append(col); result.append(g[:, c])
            return np.array(result, dtype=np.int32).T if result else None
        cands.append(dedup_cols)
        
        def remove_zero_rows(g):
            mask = np.any(g != 0, axis=1)
            return g[mask] if np.any(mask) else None
        cands.append(remove_zero_rows)
        
        def remove_zero_cols(g):
            mask = np.any(g != 0, axis=0)
            return g[:, mask] if np.any(mask) else None
        cands.append(remove_zero_cols)
        
        for rc in range(10):
            def mk_remove(color=rc):
                def fn(g):
                    mask = ~np.all(g == color, axis=1)
                    return g[mask] if np.any(mask) else None
                return fn
            cands.append(mk_remove())
            
            def mk_remove_c(color=rc):
                def fn(g):
                    mask = ~np.all(g == color, axis=0)
                    return g[:, mask] if np.any(mask) else None
                return fn
            cands.append(mk_remove_c())
        return cands

    # ============================================================
    # 46. TWO-STEP COMPOSITION (SELECTED)
    # ============================================================
    def _two_step(self, train) -> list[Prog]:
        cands: list[Prog] = []
        for rot in (1, 2, 3):
            def mk_cr(r=rot):
                def fn(g):
                    rows,cols = np.where(g!=0)
                    if len(rows)==0: return None
                    return np.rot90(g[rows.min():rows.max()+1, cols.min():cols.max()+1], r)
                return fn
            cands.append(mk_cr())
        
        for fl in ("h","v"):
            def mk_cf(f=fl):
                def fn(g):
                    rows,cols = np.where(g!=0)
                    if len(rows)==0: return None
                    sub = g[rows.min():rows.max()+1, cols.min():cols.max()+1]
                    return np.fliplr(sub) if f=="h" else np.flipud(sub)
                return fn
            cands.append(mk_cf())
        
        # Crop + transpose
        def crop_transpose(g):
            rows,cols = np.where(g!=0)
            if len(rows)==0: return None
            return g[rows.min():rows.max()+1, cols.min():cols.max()+1].T
        cands.append(crop_transpose)
        
        return cands

    # ============================================================
    # NEW v5 PRIMITIVES
    # ============================================================

    # 47. FILL ENCLOSED PER COLOR
    def _fill_enclosed_per_color(self, train) -> list[Prog]:
        """Fill enclosed regions with the enclosing color."""
        cands: list[Prog] = []
        
        for bg in (0,):
            def mk(b=bg):
                def fn(g):
                    h, w = g.shape; out = g.copy()
                    # For each non-bg color, find enclosed bg regions
                    for col in np.unique(g):
                        if col == b: continue
                        col = int(col)
                        # Create binary mask: this color vs everything else
                        mask = (g == col)
                        # Find bg regions enclosed by this color
                        vis = np.zeros((h, w), dtype=bool)
                        stk = []
                        for r in range(h):
                            for c in (0, w-1):
                                if not mask[r,c] and not vis[r,c]:
                                    vis[r,c] = True; stk.append((r,c))
                        for c in range(w):
                            for r in (0, h-1):
                                if not mask[r,c] and not vis[r,c]:
                                    vis[r,c] = True; stk.append((r,c))
                        while stk:
                            r,c = stk.pop()
                            for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                                nr,nc=r+dr,c+dc
                                if 0<=nr<h and 0<=nc<w and not mask[nr,nc] and not vis[nr,nc]:
                                    vis[nr,nc]=True; stk.append((nr,nc))
                        for r in range(h):
                            for c in range(w):
                                if g[r,c]==b and not vis[r,c]:
                                    out[r,c] = col
                    return out
                return fn
            cands.append(mk())
        return cands

    # 48. CONNECT SAME-COLOR H/V
    def _connect_same_color_hv(self, train) -> list[Prog]:
        """Connect all same-color pixels with horizontal/vertical lines."""
        cands: list[Prog] = []
        
        def connect_hv(g):
            h, w = g.shape; out = g.copy()
            for col in np.unique(g):
                if col == 0: continue
                rows, cols = np.where(g == col)
                pts = list(zip(rows, cols))
                for i in range(len(pts)):
                    for j in range(i+1, len(pts)):
                        r1,c1 = pts[i]; r2,c2 = pts[j]
                        if r1 == r2:
                            for c in range(min(c1,c2), max(c1,c2)+1):
                                out[r1, c] = col
                        elif c1 == c2:
                            for r in range(min(r1,r2), max(r1,r2)+1):
                                out[r, c1] = col
            return out
        cands.append(connect_hv)
        return cands

    # 49. RECOLOR BY ENCLOSURE
    def _recolor_by_enclosure(self, train) -> list[Prog]:
        """Objects enclosed by another get a specific color."""
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        
        # For each object in output, check if it's enclosed by another color in input
        for conn in (4, 8):
            objs_in = get_objects(inp0, conn=conn)
            objs_out = get_objects(out0, conn=conn)
            
            # Find which objects changed color
            color_changes = {}
            for oi in objs_in:
                for r, c in oi['cells']:
                    oc = int(out0[r, c])
                    if oc != oi['color']:
                        if oi['color'] not in color_changes:
                            color_changes[oi['color']] = set()
                        color_changes[oi['color']].add(oc)
            
            if color_changes:
                for orig_c, new_cs in color_changes.items():
                    if len(new_cs) == 1:
                        new_c = list(new_cs)[0]
                        def mk(oc=orig_c, nc=new_c, cn=conn):
                            def fn(g):
                                out = g.copy()
                                out[g == oc] = nc
                                return out
                            return fn
                        cands.append(mk())
        return cands

    # 50. DRAW BORDERS AROUND OBJECTS
    def _draw_borders_around_objects(self, train) -> list[Prog]:
        """Draw 1-pixel borders around each object."""
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        
        diff = (inp0 != out0)
        if not np.any(diff): return cands
        new_colors = set(map(int, np.unique(out0[diff])))
        
        for nc in new_colors:
            for conn in (4, 8):
                def mk(new_c=nc, cn=conn):
                    def fn(g):
                        h, w = g.shape; out = g.copy()
                        for o in get_objects(g, conn=cn):
                            for r, c in o['cells']:
                                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                                    nr, nc2 = r+dr, c+dc
                                    if 0<=nr<h and 0<=nc2<w and g[nr,nc2]==0:
                                        out[nr, nc2] = new_c
                        return out
                    return fn
                cands.append(mk())
        return cands

    # 51. CHECKERBOARD FILL
    def _checkerboard_fill(self, train) -> list[Prog]:
        """Fill grid with checkerboard pattern."""
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        
        # Check if output is a checkerboard over input
        colors_out = sorted(set(map(int, np.unique(out0))))
        if len(colors_out) == 2:
            c1, c2 = colors_out
            # Check (r+c)%2 pattern
            ok = True
            for r in range(out0.shape[0]):
                for c in range(out0.shape[1]):
                    expected = c1 if (r+c)%2==0 else c2
                    if out0[r,c] != expected: ok = False; break
                if not ok: break
            if ok:
                def mk(a=c1, b=c2):
                    def fn(g):
                        h, w = g.shape
                        out = np.zeros((h,w), dtype=np.int32)
                        for r in range(h):
                            for c in range(w):
                                out[r,c] = a if (r+c)%2==0 else b
                        return out
                    return fn
                cands.append(mk())
            # Try opposite
            ok2 = True
            for r in range(out0.shape[0]):
                for c in range(out0.shape[1]):
                    expected = c2 if (r+c)%2==0 else c1
                    if out0[r,c] != expected: ok2 = False; break
                if not ok2: break
            if ok2:
                def mk2(a=c2, b=c1):
                    def fn(g):
                        h, w = g.shape
                        out = np.zeros((h,w), dtype=np.int32)
                        for r in range(h):
                            for c in range(w):
                                out[r,c] = a if (r+c)%2==0 else b
                        return out
                    return fn
                cands.append(mk2())
        return cands

    # 52. PIXEL-NEIGHBOR COLOR RULE  
    def _pixel_neighbor_color_rule(self, train) -> list[Prog]:
        """Learn rule based on (self_color, set of neighbor colors) → output."""
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        
        # Feature: (self_color, tuple(sorted unique neighbor colors))
        mapping = {}; ok = True
        for inp, out in train:
            if inp.shape != out.shape: ok = False; break
            h, w = inp.shape
            for r in range(h):
                for c in range(w):
                    nbr_colors = set()
                    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                        nr, nc = r+dr, c+dc
                        if 0<=nr<h and 0<=nc<w:
                            nbr_colors.add(int(inp[nr,nc]))
                    key = (int(inp[r,c]), tuple(sorted(nbr_colors)))
                    val = int(out[r,c])
                    if key in mapping and mapping[key] != val: ok = False; break
                    mapping[key] = val
                if not ok: break
            if not ok: break
        
        if ok and mapping:
            def mk(m=mapping.copy()):
                def fn(g):
                    h, w = g.shape; out = np.zeros_like(g)
                    for r in range(h):
                        for c in range(w):
                            nbr_colors = set()
                            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                                nr, nc = r+dr, c+dc
                                if 0<=nr<h and 0<=nc<w:
                                    nbr_colors.add(int(g[nr,nc]))
                            key = (int(g[r,c]), tuple(sorted(nbr_colors)))
                            out[r,c] = m.get(key, int(g[r,c]))
                    return out
                return fn
            cands.append(mk())
        return cands

    # 53. REPAIR WITH PATTERN  
    def _repair_with_pattern(self, train) -> list[Prog]:
        """Detect the repeating tile and repair broken cells."""
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        h, w = inp0.shape
        
        for th in range(1, h//2+1):
            if h % th != 0: continue
            for tw in range(1, w//2+1):
                if w % tw != 0: continue
                # Determine tile from output
                tile = out0[:th, :tw].copy()
                tiled = np.tile(tile, (h//th, w//tw))
                if np.array_equal(tiled, out0):
                    def mk(t=tile.copy()):
                        def fn(g):
                            hh, ww = g.shape
                            th2, tw2 = t.shape
                            if hh % th2 != 0 or ww % tw2 != 0: return None
                            return np.tile(t, (hh//th2, ww//tw2))
                        return fn
                    cands.append(mk())
                    break
            else:
                continue
            break
        return cands

    # 54. OVERLAY ALL OBJECTS (stack all objects on top of each other)
    def _overlay_all_objects(self, train) -> list[Prog]:
        """Extract all objects and overlay them on a minimal grid."""
        cands: list[Prog] = []
        inp0, out0 = train[0]
        
        for conn in (4, 8):
            objs = get_objects(inp0, conn=conn)
            if len(objs) < 2: continue
            
            # Check if all objects have the same bounding box size
            sizes = set((o['h'], o['w']) for o in objs)
            if len(sizes) != 1: continue
            oh, ow = sizes.pop()
            if oh != out0.shape[0] or ow != out0.shape[1]: continue
            
            def mk(cn=conn):
                def fn(g):
                    objs2 = get_objects(g, conn=cn)
                    if not objs2: return None
                    sizes2 = set((o['h'], o['w']) for o in objs2)
                    if len(sizes2) != 1: return None
                    oh2, ow2 = sizes2.pop()
                    out = np.zeros((oh2, ow2), dtype=np.int32)
                    for o in objs2:
                        mask = o['mask']
                        m = mask != 0
                        out[m] = mask[m]
                    return out
                return fn
            cands.append(mk())
        return cands

    # 55. RECOLOR BY OBJECT SIZE
    def _recolor_by_object_size(self, train) -> list[Prog]:
        """Recolor objects based on their relative size."""
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        
        for conn in (4, 8):
            objs = get_objects(inp0, conn=conn)
            if len(objs) < 2: continue
            
            # Learn: size → new color
            size_to_color = {}; ok = True
            for o in objs:
                new_colors = set(int(out0[r,c]) for r,c in o['cells'])
                if len(new_colors) != 1: ok = False; break
                nc = new_colors.pop()
                if o['area'] in size_to_color and size_to_color[o['area']] != nc:
                    ok = False; break
                size_to_color[o['area']] = nc
            
            if ok and size_to_color and size_to_color != {o['area']: o['color'] for o in objs}:
                def mk(cn=conn, stc=size_to_color.copy()):
                    def fn(g):
                        out = g.copy()
                        for o in get_objects(g, conn=cn):
                            if o['area'] in stc:
                                for r,c in o['cells']:
                                    out[r,c] = stc[o['area']]
                        return out
                    return fn
                cands.append(mk())
        return cands

    # 56. EXTEND LINES TO BORDER
    def _extend_lines_to_border(self, train) -> list[Prog]:
        """Extend colored pixels to grid borders."""
        cands: list[Prog] = []
        
        # Extend horizontally
        def extend_h(g):
            h, w = g.shape; out = g.copy()
            for r in range(h):
                for c in range(w):
                    if g[r,c] != 0:
                        col = int(g[r,c])
                        for cc in range(w):
                            if out[r,cc] == 0: out[r,cc] = col
            return out
        cands.append(extend_h)
        
        # Extend vertically
        def extend_v(g):
            h, w = g.shape; out = g.copy()
            for r in range(h):
                for c in range(w):
                    if g[r,c] != 0:
                        col = int(g[r,c])
                        for rr in range(h):
                            if out[rr,c] == 0: out[rr,c] = col
            return out
        cands.append(extend_v)
        
        return cands

    # 57. PAINT BETWEEN MARKERS
    def _paint_between_markers(self, train) -> list[Prog]:
        """Paint between pairs of same-color markers."""
        cands: list[Prog] = []
        
        # Fill rectangle between any 2 same-color points
        def fill_rect_between(g):
            h, w = g.shape; out = g.copy()
            for col in np.unique(g):
                if col == 0: continue
                rows, cols = np.where(g == col)
                if len(rows) == 2:
                    r1, r2 = rows.min(), rows.max()
                    c1, c2 = cols.min(), cols.max()
                    out[r1:r2+1, c1:c2+1] = col
            return out
        cands.append(fill_rect_between)
        
        return cands

    # 58. CORNER FILL
    def _corner_fill(self, train) -> list[Prog]:
        """Fill corners or specific positions based on context."""
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        
        # Try: for each object, fill its corners
        for conn in (4, 8):
            def mk(cn=conn):
                def fn(g):
                    h, w = g.shape; out = g.copy()
                    for o in get_objects(g, conn=cn):
                        mr, mc, Mr, Mc = o['bbox']
                        col = o['color']
                        # Fill corners of bbox
                        for r, c in [(mr,mc),(mr,Mc),(Mr,mc),(Mr,Mc)]:
                            if 0<=r<h and 0<=c<w and out[r,c]==0:
                                out[r,c] = col
                    return out
                return fn
            cands.append(mk())
        return cands

    # 59. OBJECT INTERIOR FILL WITH BG
    def _object_interior_fill_bg(self, train) -> list[Prog]:
        """Replace interior of objects with background color (extract outline)."""
        cands: list[Prog] = []
        
        def extract_outline(g):
            h, w = g.shape; out = np.zeros_like(g)
            for r in range(h):
                for c in range(w):
                    if g[r,c] != 0:
                        is_border = False
                        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                            nr, nc = r+dr, c+dc
                            if nr<0 or nr>=h or nc<0 or nc>=w or g[nr,nc]==0:
                                is_border = True; break
                        if is_border: out[r,c] = g[r,c]
            return out
        cands.append(extract_outline)
        return cands

    # 60. ROW/COL COLOR RULE
    def _row_col_color_rule(self, train) -> list[Prog]:
        """Learn: (row_color_signature, col, self_color) → output_color."""
        cands: list[Prog] = []
        inp0, out0 = train[0]
        if inp0.shape != out0.shape: return cands
        h, w = inp0.shape
        
        # Try: output pixel determined by (row index color set, col index color set)
        # Simplified: row dominant color
        for inp, out in train:
            pass  # Complex, skip for now
        return cands

    # ============================================================
    # SHAPE-CHANGING NEW PRIMITIVES
    # ============================================================

    # 61. EXTRACT UNIQUE SHAPE
    def _extract_unique_shape(self, train) -> list[Prog]:
        """Extract the object whose shape is unique among all objects."""
        cands: list[Prog] = []
        for conn in (4, 8):
            def mk(cn=conn):
                def fn(g):
                    objs = get_objects(g, conn=cn)
                    if len(objs) < 3: return None
                    shapes = {}
                    for o in objs:
                        key = (o['h'], o['w'], tuple(o['mask'].flatten()))
                        if key not in shapes: shapes[key] = []
                        shapes[key].append(o)
                    unique = [v[0] for v in shapes.values() if len(v) == 1]
                    if len(unique) == 1:
                        return unique[0]['mask']
                    return None
                return fn
            cands.append(mk())
        return cands

    # 62. COUNT OBJECTS TO GRID
    def _count_objects_to_grid(self, train) -> list[Prog]:
        """Output grid dimensions or content based on counting."""
        cands: list[Prog] = []
        inp0, out0 = train[0]
        oh, ow = out0.shape
        
        for conn in (4, 8):
            n_objs = len(get_objects(inp0, conn=conn))
            colors = get_nonbg_colors(inp0)
            n_colors = len(colors)
            
            # Output is n_objs × n_objs grid
            if oh == n_objs and ow == n_objs:
                def mk(cn=conn):
                    def fn(g):
                        n = len(get_objects(g, conn=cn))
                        if n == 0: return None
                        # Fill with most common non-bg color
                        nbc = get_nonbg_colors(g)
                        c = nbc[0] if nbc else 1
                        return np.full((n, n), c, dtype=np.int32)
                    return fn
                cands.append(mk())
            
            # Output is n_colors × n_colors grid
            if oh == n_colors and ow == n_colors:
                def mk2(cn=conn):
                    def fn(g):
                        nc = len(get_nonbg_colors(g))
                        if nc == 0: return None
                        nbc = get_nonbg_colors(g)
                        c = nbc[0] if nbc else 1
                        return np.full((nc, nc), c, dtype=np.int32)
                    return fn
                cands.append(mk2())
        return cands

    # 63. COMPRESS GRID
    def _compress_grid(self, train) -> list[Prog]:
        """Remove rows/cols of specific color to compress grid."""
        cands: list[Prog] = []
        inp0, out0 = train[0]
        
        for remove_c in range(10):
            # Remove rows that are all remove_c
            row_mask = ~np.all(inp0 == remove_c, axis=1)
            col_mask = ~np.all(inp0 == remove_c, axis=0)
            if np.any(row_mask) and np.any(col_mask):
                compressed = inp0[row_mask][:, col_mask]
                if np.array_equal(compressed, out0):
                    def mk(rc=remove_c):
                        def fn(g):
                            rm = ~np.all(g == rc, axis=1)
                            cm = ~np.all(g == rc, axis=0)
                            if not np.any(rm) or not np.any(cm): return None
                            return g[rm][:, cm]
                        return fn
                    cands.append(mk())
            
            # Remove rows only
            if np.any(row_mask):
                compressed_r = inp0[row_mask]
                if np.array_equal(compressed_r, out0):
                    def mk_r(rc=remove_c):
                        def fn(g):
                            rm = ~np.all(g == rc, axis=1)
                            return g[rm] if np.any(rm) else None
                        return fn
                    cands.append(mk_r())
            
            # Remove cols only
            if np.any(col_mask):
                compressed_c = inp0[:, col_mask]
                if np.array_equal(compressed_c, out0):
                    def mk_c(rc=remove_c):
                        def fn(g):
                            cm = ~np.all(g == rc, axis=0)
                            return g[:, cm] if np.any(cm) else None
                        return fn
                    cands.append(mk_c())
        return cands

    # 64. EXTRACT BY FRAME
    def _extract_by_frame(self, train) -> list[Prog]:
        """Find rectangular frame in grid and extract its interior."""
        cands: list[Prog] = []
        
        for frame_c in range(1, 10):
            def mk(fc=frame_c):
                def fn(g):
                    h, w = g.shape
                    rows, cols = np.where(g == fc)
                    if len(rows) < 4: return None
                    r1, r2 = rows.min(), rows.max()
                    c1, c2 = cols.min(), cols.max()
                    if r2-r1 < 2 or c2-c1 < 2: return None
                    # Check frame completeness
                    if (np.all(g[r1, c1:c2+1] == fc) and
                        np.all(g[r2, c1:c2+1] == fc) and
                        np.all(g[r1:r2+1, c1] == fc) and
                        np.all(g[r1:r2+1, c2] == fc)):
                        return g[r1+1:r2, c1+1:c2].copy()
                    return None
                return fn
            cands.append(mk())
        return cands

    # 65. SPLIT AND SELECT BY CONTENT
    def _split_and_select_by_content(self, train) -> list[Prog]:
        """Split grid into panels and select based on content properties."""
        cands: list[Prog] = []
        
        for dc in range(10):
            # Select panel with most/fewest unique colors
            for criterion in ("most_colors", "fewest_colors", "most_nonzero", "fewest_nonzero"):
                def mk(d=dc, cr=criterion):
                    def fn(g):
                        ps = split_panels(g, d)
                        if len(ps) < 2: return None
                        if cr == "most_colors":
                            return max(ps, key=lambda p: len(set(map(int, np.unique(p))) - {0, d}))
                        elif cr == "fewest_colors":
                            return min(ps, key=lambda p: len(set(map(int, np.unique(p))) - {0, d}))
                        elif cr == "most_nonzero":
                            return max(ps, key=lambda p: np.count_nonzero(p != d))
                        elif cr == "fewest_nonzero":
                            return min(ps, key=lambda p: np.count_nonzero(p != d))
                        return None
                    return fn
                cands.append(mk())
        return cands

    # 66. MIRRORED TILING
    def _mirrored_tiling(self, train) -> list[Prog]:
        """Tile with alternating mirrors to create symmetric patterns."""
        cands: list[Prog] = []
        inp0, out0 = train[0]
        ih, iw = inp0.shape; oh, ow = out0.shape
        
        # Check if output is 2x2 mirrored tile
        if oh == 2*ih and ow == 2*iw:
            # h-mirror vertically, v-mirror horizontally
            tiled = np.vstack([
                np.hstack([inp0, np.fliplr(inp0)]),
                np.hstack([np.flipud(inp0), np.fliplr(np.flipud(inp0))])
            ])
            if np.array_equal(tiled, out0):
                def mk():
                    def fn(g):
                        return np.vstack([
                            np.hstack([g, np.fliplr(g)]),
                            np.hstack([np.flipud(g), np.fliplr(np.flipud(g))])
                        ])
                    return fn
                cands.append(mk())
            
            tiled2 = np.vstack([
                np.hstack([inp0, np.fliplr(inp0)]),
                np.hstack([np.flipud(inp0), np.rot90(inp0, 2)])
            ])
            if np.array_equal(tiled2, out0):
                def mk2():
                    def fn(g):
                        return np.vstack([
                            np.hstack([g, np.fliplr(g)]),
                            np.hstack([np.flipud(g), np.rot90(g, 2)])
                        ])
                    return fn
                cands.append(mk2())
        
        # Horizontal double with mirror
        if oh == ih and ow == 2*iw:
            for mirror_fn in [np.fliplr, np.flipud, lambda x: np.rot90(x, 2)]:
                tiled = np.hstack([inp0, mirror_fn(inp0)])
                if np.array_equal(tiled, out0):
                    def mk3(mf=mirror_fn):
                        def fn(g): return np.hstack([g, mf(g)])
                        return fn
                    cands.append(mk3())
        
        # Vertical double with mirror
        if oh == 2*ih and ow == iw:
            for mirror_fn in [np.fliplr, np.flipud, lambda x: np.rot90(x, 2)]:
                tiled = np.vstack([inp0, mirror_fn(inp0)])
                if np.array_equal(tiled, out0):
                    def mk4(mf=mirror_fn):
                        def fn(g): return np.vstack([g, mf(g)])
                        return fn
                    cands.append(mk4())
        
        return cands

    # 67. UPSCALE PATTERN
    def _upscale_pattern(self, train) -> list[Prog]:
        """Scale up each pixel into a NxN block based on its color."""
        cands: list[Prog] = []
        inp0, out0 = train[0]
        ih, iw = inp0.shape; oh, ow = out0.shape
        
        if oh > ih and ow > iw and oh % ih == 0 and ow % iw == 0:
            sy, sx = oh // ih, ow // iw
            
            # Simple scale (already covered in _scaling, but let's add pattern variants)
            # Each pixel's block is the original grid scaled by its color
            # Or: each pixel's block is filled with a specific pattern
            pass
        
        return cands

    # 68. ASSEMBLE FROM OBJECTS
    def _assemble_from_objects(self, train) -> list[Prog]:
        """Extract objects and assemble them in a specific layout."""
        cands: list[Prog] = []
        inp0, out0 = train[0]
        
        for conn in (4, 8):
            for bg in (0,):
                objs = get_objects(inp0, conn=conn, bg=bg)
                if len(objs) < 2: continue
                
                # All objects same size → assemble in grid
                sizes = set((o['h'], o['w']) for o in objs)
                if len(sizes) != 1: continue
                oh, ow = sizes.pop()
                n = len(objs)
                
                # Try various arrangements
                for nr in range(1, n+1):
                    if n % nr != 0: continue
                    nc = n // nr
                    if oh*nr == out0.shape[0] and ow*nc == out0.shape[1]:
                        # Sort by position (top-left first)
                        def mk(cn=conn, b=bg, r_=nr, c_=nc):
                            def fn(g):
                                objs2 = get_objects(g, conn=cn, bg=b)
                                if not objs2: return None
                                sizes2 = set((o['h'], o['w']) for o in objs2)
                                if len(sizes2) != 1: return None
                                oh2, ow2 = sizes2.pop()
                                n2 = len(objs2)
                                if n2 != r_ * c_: return None
                                # Sort by top-left position
                                objs2.sort(key=lambda o: (o['min_r'], o['min_c']))
                                out = np.zeros((oh2*r_, ow2*c_), dtype=np.int32)
                                for i, o in enumerate(objs2):
                                    ri, ci = divmod(i, c_)
                                    out[ri*oh2:(ri+1)*oh2, ci*ow2:(ci+1)*ow2] = o['mask']
                                return out
                            return fn
                        cands.append(mk())
        return cands

    # 69. EXTRACT DIFF REGION
    def _extract_diff_region(self, train) -> list[Prog]:
        """For panel tasks: extract the region that differs between panels."""
        cands: list[Prog] = []
        
        for dc in range(10):
            # Find panels, extract where they differ
            inp0, out0 = train[0]
            ps = split_panels(inp0, dc)
            if len(ps) < 2: continue
            if not all(p.shape == ps[0].shape for p in ps): continue
            
            # Reference = majority agreement
            ref = ps[0].copy()
            for r in range(ref.shape[0]):
                for c in range(ref.shape[1]):
                    votes = Counter(int(p[r,c]) for p in ps)
                    ref[r,c] = votes.most_common(1)[0][0]
            
            # Find which panel differs and extract the diff
            for idx in range(len(ps)):
                diff_mask = ps[idx] != ref
                if np.any(diff_mask):
                    # Check if diff region matches output
                    rows, cols = np.where(diff_mask)
                    if len(rows) > 0:
                        sub = ps[idx][rows.min():rows.max()+1, cols.min():cols.max()+1]
                        if np.array_equal(sub, out0):
                            def mk(d=dc, i=idx):
                                def fn(g):
                                    ps2 = split_panels(g, d)
                                    if len(ps2) <= i: return None
                                    if not all(p.shape == ps2[0].shape for p in ps2): return None
                                    ref2 = ps2[0].copy()
                                    for r in range(ref2.shape[0]):
                                        for c in range(ref2.shape[1]):
                                            votes = Counter(int(p[r,c]) for p in ps2)
                                            ref2[r,c] = votes.most_common(1)[0][0]
                                    diff2 = ps2[i] != ref2
                                    if not np.any(diff2): return None
                                    rows2, cols2 = np.where(diff2)
                                    return ps2[i][rows2.min():rows2.max()+1, cols2.min():cols2.max()+1]
                                return fn
                            cands.append(mk())
        return cands

    # ============================================================
    # 2-STEP COMPOSITION ENGINE
    # ============================================================
    def _compose(self, train, t0) -> list[Prog]:
        """Try 2-step compositions: preprocess → solve."""
        cands = []
        
        # Define preprocessors
        preprocessors = []
        
        # Geometric preprocessors
        for k in (1, 2, 3):
            def mk_rot(kk=k): return lambda g: np.rot90(g, kk)
            preprocessors.append(("rot"+str(k*90), mk_rot()))
        preprocessors.append(("flipLR", lambda g: np.fliplr(g)))
        preprocessors.append(("flipUD", lambda g: np.flipud(g)))
        preprocessors.append(("transpose", lambda g: g.T))
        preprocessors.append(("flipT", lambda g: np.fliplr(g.T)))
        
        # Crop preprocessors
        preprocessors.append(("crop_nz", lambda g: crop_nz(g)))
        for bg in range(1, 10):
            def mk_crop_bg(b=bg): return lambda g: crop_bg(g, b)
            preprocessors.append((f"crop_bg{bg}", mk_crop_bg()))
        
        # Color-specific crop
        for col in range(1, 10):
            def mk_crop_col(c=col): return lambda g: crop_color(g, c)
            preprocessors.append((f"crop_c{col}", mk_crop_col()))
        
        # Extract specific color object mask
        for col in range(1, 10):
            def mk_extract_mask(c=col):
                def fn(g):
                    r, cc = np.where(g == c)
                    if len(r) == 0: return None
                    sub = g[r.min():r.max()+1, cc.min():cc.max()+1].copy()
                    out = np.zeros_like(sub)
                    out[sub == c] = c
                    return out
                return fn
            preprocessors.append((f"mask_c{col}", mk_extract_mask()))
        
        # Frame extraction
        for fc in range(1, 10):
            def mk_frame(f=fc):
                def fn(g):
                    h, w = g.shape
                    rows, cols = np.where(g == f)
                    if len(rows) < 4: return None
                    r1, r2 = rows.min(), rows.max()
                    c1, c2 = cols.min(), cols.max()
                    if r2-r1 < 2 or c2-c1 < 2: return None
                    return g[r1+1:r2, c1+1:c2].copy()
                return fn
            preprocessors.append((f"frame_{fc}", mk_frame()))
        
        # Simple single-step solvers for composition
        simple_solvers = [
            self._rigid,
            self._palette,
            self._holes,
            self._gravity,
            self._symmetry,
            self._mirror_complete,
            self._invert_colors,
            self._cellular,
        ]
        
        for prep_name, prep_fn in preprocessors:
            if time.perf_counter() - t0 > TASK_TIMEOUT - 1: break
            
            # Apply preprocessor to all training inputs
            processed_train = []
            valid = True
            for inp, out in train:
                try:
                    p = prep_fn(inp)
                    if p is None or not isinstance(p, np.ndarray) or p.ndim != 2:
                        valid = False; break
                    if p.shape[0] == 0 or p.shape[1] == 0:
                        valid = False; break
                    processed_train.append((p, out))
                except:
                    valid = False; break
            
            if not valid or not processed_train: continue
            
            # Check shape compatibility
            shapes_match = all(p.shape == out.shape for p, out in processed_train)
            
            # Only try relevant solvers
            for solver_fn in simple_solvers:
                if time.perf_counter() - t0 > TASK_TIMEOUT - 0.5: break
                try:
                    for c in solver_fn(processed_train):
                        try:
                            ok = True
                            for p_inp, out in processed_train:
                                result = safe_call(c, p_inp)
                                if not exact(result, out):
                                    ok = False; break
                            if ok:
                                def mk_compose(pp=prep_fn, cc=c):
                                    def fn(g):
                                        mid = pp(g)
                                        if mid is None: return None
                                        return cc(mid)
                                    return fn
                                cands.append(mk_compose())
                                if len(cands) >= 3: return cands
                        except: pass
                except: pass
        
        # Also try: solve → postprocess (for invertible transforms)
        invertible = [
            (lambda g: np.rot90(g, 1), lambda g: np.rot90(g, 3)),
            (lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 2)),
            (lambda g: np.rot90(g, 3), lambda g: np.rot90(g, 1)),
            (lambda g: np.fliplr(g), lambda g: np.fliplr(g)),
            (lambda g: np.flipud(g), lambda g: np.flipud(g)),
            (lambda g: g.T, lambda g: g.T),
        ]
        
        for post_fn, inv_fn in invertible:
            if time.perf_counter() - t0 > TASK_TIMEOUT - 1: break
            
            # Compute inverted training targets
            inv_train = []
            valid = True
            for inp, out in train:
                try:
                    inv_out = inv_fn(out)
                    if inv_out is None or not isinstance(inv_out, np.ndarray):
                        valid = False; break
                    inv_train.append((inp, inv_out))
                except:
                    valid = False; break
            
            if not valid or not inv_train: continue
            
            for solver_fn in simple_solvers:
                if time.perf_counter() - t0 > TASK_TIMEOUT - 0.5: break
                try:
                    for c in solver_fn(inv_train):
                        try:
                            ok = True
                            for inp, inv_out in inv_train:
                                result = safe_call(c, inp)
                                if not exact(result, inv_out):
                                    ok = False; break
                            if ok:
                                def mk_post(cc=c, pf=post_fn):
                                    def fn(g):
                                        mid = cc(g)
                                        if mid is None: return None
                                        return pf(mid)
                                    return fn
                                cands.append(mk_post())
                                if len(cands) >= 3: return cands
                        except: pass
                except: pass
        
        return cands


# ============================================================
# BENCHMARK EVALUATOR
# ============================================================

def run_benchmark(data_dir="arc_data", split="training", limit=0):
    root = Path(data_dir)
    if split == "training": tasks = sorted((root/"training").glob("*.json"))
    elif split == "evaluation": tasks = sorted((root/"evaluation").glob("*.json"))
    elif split == "all": tasks = sorted(root.rglob("*.json"))
    else: tasks = sorted(root.glob("*.json"))
    if limit > 0: tasks = tasks[:limit]

    print("="*80, flush=True)
    print("MATHX PURE SYMBOLIC ENGINE v5 (STRICT NON-LLM / 200+ PRIMITIVES)", flush=True)
    print("="*80, flush=True)
    print(f"Split: {split.upper()}, Tasks: {len(tasks)}\n", flush=True)

    solver = PureSymbolicSolverV5()
    solved1 = solved2 = fit = 0
    t0 = time.perf_counter()
    solved_names = []

    for idx, fp in enumerate(tasks, 1):
        task = json.loads(fp.read_text(encoding="utf-8"))
        ts = time.perf_counter()
        sols = solver.solve(task)
        dt = time.perf_counter() - ts

        ti = [G(ex["input"]) for ex in task.get("test",[])]
        to = [G(ex["output"]) for ex in task.get("test",[]) if "output" in ex]
        
        s1 = s2 = False
        if sols:
            fit += 1
            if to:
                try:
                    p = safe_call(sols[0], ti[0])
                    s1 = exact(p, to[0]); s2 = s1
                except: pass
                if not s1 and len(sols) > 1:
                    try:
                        p = safe_call(sols[1], ti[0])
                        s2 = exact(p, to[0])
                    except: pass
                if not s2 and len(sols) > 2:
                    try:
                        p = safe_call(sols[2], ti[0])
                        s2 = exact(p, to[0])
                    except: pass
        if s1: solved1 += 1
        if s2: solved2 += 1
        if s1 or s2: solved_names.append(fp.stem)
        
        st = "SOLVED(1)" if s1 else ("SOLVED(2)" if s2 else ("FIT" if sols else "MISS"))
        if idx<=15 or idx%50==0 or idx==len(tasks) or s1 or s2:
            print(f"[{idx:03d}/{len(tasks)}] {fp.stem:12s} | {st:10s} | rules={len(sols):2d} | {dt*1000:.0f}ms", flush=True)

    total = time.perf_counter() - t0
    print(f"\n{'='*80}", flush=True)
    print("FINAL RESULTS (STRICT NON-LLM)", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Tasks:       {len(tasks)}", flush=True)
    print(f"Train Fit:   {fit}/{len(tasks)} ({100*fit/len(tasks):.2f}%)", flush=True)
    print(f"Solved Top1: {solved1}/{len(tasks)} ({100*solved1/len(tasks):.2f}%)", flush=True)
    print(f"Solved Top2: {solved2}/{len(tasks)} ({100*solved2/len(tasks):.2f}%)", flush=True)
    print(f"Total Time:  {total:.2f}s ({total/len(tasks)*1000:.1f}ms/task)", flush=True)
    print(f"{'='*80}", flush=True)
    if solved_names:
        print(f"\nSolved tasks: {', '.join(solved_names[:30])}", flush=True)
        if len(solved_names) > 30:
            print(f"  ... and {len(solved_names)-30} more", flush=True)

    Path("mathx_symbolic_benchmark_report.json").write_text(json.dumps({
        "engine":"Pure Symbolic Engine v5 (Strict Non-LLM)",
        "split":split,"tasks":len(tasks),
        "fit":fit,"top1":solved1,"top2":solved2,
        "total_time_seconds":total,
        "avg_ms_per_task": total/len(tasks)*1000 if tasks else 0,
        "solved_tasks": solved_names,
    },indent=2), encoding="utf-8")


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--data", default="arc_data")
    pa.add_argument("--split", default="training", choices=["all","training","evaluation"])
    pa.add_argument("--limit", type=int, default=0)
    a = pa.parse_args()
    run_benchmark(a.data, a.split, a.limit)
