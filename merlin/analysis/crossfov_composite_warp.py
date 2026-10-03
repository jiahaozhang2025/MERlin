"""Cross-FOV composite warp: fill the margins that stage drift pushed out of a FOV's
camera with the same tissue as imaged by the neighbouring FOV in that round.

Problem this solves (20260824): between rounds the sample slid by up to ~300 px, so
in late rounds the content of each FOV's right/top strip (round-0 frame) was imaged
by the neighbouring FOV instead. Per-FOV registration (PolyWarp) can only mark that
strip invalid (adaptive crop), losing ~19% of the footprint although the pixels exist.

CrossFOVCompositeWarp is a warp task layered on top of a finished PolyWarp task:

  * _run_analysis(fov): for every neighbour n of fov k whose frame overlaps k's
    invalid margins, measure the LINK between k's and n's round-0 bead fields:
    rigid xy offset t (stage seed + correlation refinement) and focus offset dz.
    Measured on several z planes of the 3D bead stacks; gated on SNR and spread.
    A round-0 self-test (own vs neighbour-derived content in the shared overlap
    band) is written with the link so a sign error cannot pass silently.
  * get_aligned_image(fov, ch, z): the PolyWarp own-FOV aligned plane, with the
    pixels the own warp could not sample (outside the raw frame) replaced by the
    neighbour's own-registered plane, shifted by the link (xy) and resampled by
    the link dz. Rounds whose shift is inside the fixed crop_width are left alone.
  * get_transformation(fov, ...): the transforms Decode's adaptive crop reads,
    reduced to the margins that could NOT be filled (no neighbour / failed link),
    so decoding covers everything the composite covers and nothing more.

Conventions (same as merlin.analysis.decode.compute_crop_bounds): the aligned image
is output(q) = raw(q + t) with t = (ty, tx) from PolyWarp's transformation table;
tx > 0 invalidates the RIGHT |tx| columns, tx < 0 the LEFT ones, ty > 0 the BOTTOM
rows, ty < 0 the TOP rows. A neighbour link t_kn = (row, col) offset of n's round-0
frame origin expressed in k's round-0 frame; content at n-frame position p sits at
k-frame position p + t_kn.
"""
import json
import os
import threading

import cv2
import numpy as np
import pandas as pd
from skimage import transform

from merlin.analysis import warp
from merlin.util import aberration


class CrossFOVCompositeWarp(warp.FiducialPolynomialWarp3D):

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        p = self.parameters
        p.setdefault('polywarp_task', 'PolyWarp')
        # um/px used ONLY to convert stage steps into the pixel seed of the
        # neighbour search. None = the dataset's microns_per_pixel. On the 40x
        # scope the microscope file says 0.14625 but the measured frame pitch is
        # 270 um / ~1810 px = 0.1491, a 36 px seed error at one FOV pitch.
        p.setdefault('seed_mpp', None)
        # px: half-width of the correlation search about the stage seed
        p.setdefault('link_bound', 50)
        # width (px) of the strip inside each frame used for the link crops
        p.setdefault('link_band', 512)
        # data z indices at which the link is measured
        p.setdefault('link_planes', [30, 70, 110, 150, 190])
        # um: half-range and step of the focus-offset search between the two FOVs
        p.setdefault('link_dz_range', 6.0)
        p.setdefault('link_dz_step', 0.5)
        p.setdefault('min_link_snr', 15.0)
        p.setdefault('max_link_spread', 3.0)      # px, IQR of per-plane offsets
        p.setdefault('max_link_dz_spread', 1.0)   # um
        # margins at or below this (px) are handled by crop_width anyway: no fill
        p.setdefault('fill_threshold', 100)
        # neighbour frames closer than this fraction of a frame are candidates
        p.setdefault('neighbor_search_frames', 1.6)
        p.setdefault('write_composite_FOVs', [])
        p.setdefault('write_composite_channels', [])
        p.setdefault('write_composite_z', [])
        self._pw = None
        self._linkCache = {}
        self._linkLock = threading.Lock()

    # ------------------------------------------------------------------
    # task plumbing
    # ------------------------------------------------------------------
    def get_dependencies(self):
        return [self.parameters['polywarp_task']]

    def get_estimated_memory(self):
        return 32768

    def get_estimated_time(self):
        return 30

    def _polywarp(self):
        if self._pw is None:
            self._pw = self.dataSet.load_analysis_task(
                self.parameters['polywarp_task'])
        return self._pw

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------
    def _frame_offset(self, k: int, n: int):
        """Nominal (row, col) offset of n's frame origin in k's frame, from the
        stage positions; MERlin's global frame maps image col -> stage X and
        image row -> stage Y (SimpleGlobalAlignment)."""
        mpp = self.parameters['seed_mpp'] or self.dataSet.get_microns_per_pixel()
        xk, yk = self.dataSet.get_fov_offset(k)
        xn, yn = self.dataSet.get_fov_offset(n)
        return (yn - yk) / mpp, (xn - xk) / mpp

    def _own_margins(self, fov: int):
        """Extreme invalid margins (top, bottom, left, right) over all data
        channels and planes of fov's own PolyWarp table."""
        t = self._polywarp().get_transformation_table(fov)
        tx, ty = t['xshift'].to_numpy(float), t['yshift'].to_numpy(float)
        return (max(0.0, (-ty).max()), max(0.0, ty.max()),
                max(0.0, (-tx).max()), max(0.0, tx.max()))

    def _candidate_neighbors(self, fov: int):
        """Neighbours whose frame overlaps one of fov's invalid margins, ordered
        by the area of that overlap (largest first)."""
        h, w = self.dataSet.get_image_dimensions()
        top, bottom, left, right = self._own_margins(fov)
        thr = self.parameters['fill_threshold']
        if max(top, bottom, left, right) <= thr:
            return []
        # the invalid region = union of up to four strips
        strips = []
        if top > thr:
            strips.append((0, top, 0, w))
        if bottom > thr:
            strips.append((h - bottom, h, 0, w))
        if left > thr:
            strips.append((0, h, 0, left))
        if right > thr:
            strips.append((0, h, w - right, w))
        out = []
        lim = self.parameters['neighbor_search_frames']
        for n in self.dataSet.get_fovs():
            if n == fov:
                continue
            ro, co = self._frame_offset(fov, n)
            if abs(ro) > lim * h or abs(co) > lim * w:
                continue
            area = 0.0
            for (r0, r1, c0, c1) in strips:
                rr = min(r1, ro + h) - max(r0, ro)
                cc = min(c1, co + w) - max(c0, co)
                if rr > 0 and cc > 0:
                    area += rr * cc
            if area > 0:
                out.append((area, int(n)))
        out.sort(reverse=True)
        return [n for _, n in out]

    # ------------------------------------------------------------------
    # link measurement
    # ------------------------------------------------------------------
    def _overlap_box(self, k: int, n: int):
        """Box (r0, r1, c0, c1) in k's frame where k's and n's nominal frames
        overlap, widened inward by link_bound so the search has room."""
        h, w = self.dataSet.get_image_dimensions()
        ro, co = self._frame_offset(k, n)
        b = int(self.parameters['link_bound'])
        r0, r1 = max(0, ro), min(h, ro + h)
        c0, c1 = max(0, co), min(w, co + w)
        r0i, r1i = int(np.floor(r0)) - b, int(np.ceil(r1)) + b
        c0i, c1i = int(np.floor(c0)) - b, int(np.ceil(c1)) + b
        return (max(0, r0i), min(h, r1i), max(0, c0i), min(w, c1i))

    def _plane_um(self, fov: int, zum: float):
        """Filtered round-0 bead plane of fov at depth zum (um)."""
        return self._fiducial_plane_at_z(0, fov, float(zum))

    def _measure_link(self, k: int, n: int):
        """Rigid (row, col) offset of n's frame in k's frame and focus offset dz
        (um; n's plane at z + dz matches k's plane at z), with QC."""
        h, w = self.dataSet.get_image_dimensions()
        ro, co = self._frame_offset(k, n)
        r0, r1, c0, c1 = self._overlap_box(k, n)
        bound = int(self.parameters['link_bound'])
        zs = np.array(self.dataSet.get_z_positions(), dtype=float)
        dzs = np.arange(-self.parameters['link_dz_range'],
                        self.parameters['link_dz_range'] + 1e-9,
                        self.parameters['link_dz_step'])
        per = []
        for zi in self.parameters['link_planes']:
            zum = float(zs[int(zi)])
            fixed = self._plane_um(k, zum)[r0:r1, c0:c1]
            # the cached FFT is crop-shaped: key it by the neighbour too
            fixedKey = ('link', k, n, zi)
            best = None
            for dz in dzs:
                moving = self._plane_um(n, zum + dz)
                # n's content at n-frame position p sits at k-frame p + (ro, co):
                # bring n's plane into k's frame with the nominal offset, then crop
                shifted = self._shift_image(moving, ro, co)[r0:r1, c0:c1]
                dy, dx, snr, edge = self._bounded_correlation_peak(
                    fixed, shifted, 0.0, 0.0, fixedKey=fixedKey, bound=bound)
                if best is None or snr > best[2]:
                    best = (dy, dx, snr, float(dz), edge)
            per.append(dict(z=zum, dy=float(best[0]), dx=float(best[1]),
                            snr=float(best[2]), dz=best[3], on_edge=bool(best[4])))
        dyv = np.array([q['dy'] for q in per]); dxv = np.array([q['dx'] for q in per])
        dzv = np.array([q['dz'] for q in per]); snr = np.array([q['snr'] for q in per])
        good = (snr >= self.parameters['min_link_snr']) & ~np.array([q['on_edge'] for q in per])
        res = dict(k=int(k), n=int(n), seed_row=float(ro), seed_col=float(co),
                   planes=per, ok=False, reason='')
        if good.sum() < 3:
            res['reason'] = 'only %d/%d planes pass snr>=%.0f' % (
                good.sum(), len(per), self.parameters['min_link_snr'])
            return res
        iqr = lambda v: float(np.percentile(v, 75) - np.percentile(v, 25))
        spread = max(iqr(dyv[good]), iqr(dxv[good]))
        if spread > self.parameters['max_link_spread']:
            res['reason'] = 'xy spread %.1f px' % spread
            return res
        if iqr(dzv[good]) > self.parameters['max_link_dz_spread']:
            res['reason'] = 'dz spread %.1f um' % iqr(dzv[good])
            return res
        res.update(ok=True,
                   row=float(ro + np.median(dyv[good])),
                   col=float(co + np.median(dxv[good])),
                   dz_um=float(np.median(dzv[good])),
                   snr_median=float(np.median(snr[good])),
                   xy_spread=spread, dz_spread=iqr(dzv[good]))
        return res

    def _self_test(self, k: int, link: dict):
        """Round-0 check: own vs neighbour-derived content in the shared overlap
        band must agree at zero residual. Uses the first data channel (round 0),
        where the own image is valid everywhere."""
        h, w = self.dataSet.get_image_dimensions()
        zi = int(self.parameters['link_planes'][len(self.parameters['link_planes']) // 2])
        pw = self._polywarp()
        own = pw.get_aligned_image(k, 0, zi).astype(np.float32)
        nb = self._neighbor_plane(link, 0, zi)
        ro, co = link['row'], link['col']
        r0, r1 = int(max(0, ro)) + 8, int(min(h, ro + h)) - 8
        c0, c1 = int(max(0, co)) + 8, int(min(w, co + w)) - 8
        if r1 - r0 < 64 or c1 - c0 < 64:
            return dict(ok=False, reason='overlap too small')
        a = self._filter(own[r0:r1, c0:c1]); b = self._filter(nb[r0:r1, c0:c1])
        dy, dx, snr, _ = self._bounded_correlation_peak(a, b, 0.0, 0.0, bound=12)
        corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        # The link is measured on 488 nm beads; this compares the 650/560 bit
        # channel in two different field positions, so a 1-2 px chromatic
        # difference between k's edge and n's edge is expected (Optimize's
        # chromatic corrector removes it at decode time). Gate on a sign error
        # or a wrong neighbour, not on that.
        return dict(ok=bool(abs(dy) < 3.0 and abs(dx) < 3.0 and corr > 0.2 and snr > 20),
                    residual_dy=float(dy), residual_dx=float(dx), snr=float(snr),
                    corr=corr, band=[r0, r1, c0, c1], z_index=zi)

    def _links_path(self, fov: int):
        d = self.dataSet.get_analysis_subdirectory(self, subdirectory='links')
        return os.path.join(d, 'links_%d.json' % fov)

    def _run_analysis(self, fragmentIndex: int):
        fov = list(self.dataSet.get_fovs())[fragmentIndex]
        top, bottom, left, right = self._own_margins(fov)
        out = dict(fov=int(fov), margins=dict(top=top, bottom=bottom, left=left, right=right),
                   links=[])
        for n in self._candidate_neighbors(fov):
            link = self._measure_link(fov, n)
            if link['ok']:
                link['self_test'] = self._self_test(fov, link)
                if not link['self_test']['ok']:
                    link['ok'] = False
                    link['reason'] = 'self-test failed: %s' % json.dumps(link['self_test'])
            print('fov %d neighbour %d: %s %s' % (
                fov, n, 'OK' if link['ok'] else 'REJECTED', json.dumps(
                    {q: link[q] for q in link if q not in ('planes',)}, default=str)),
                flush=True)
            out['links'].append(link)
        with open(self._links_path(fov), 'w') as f:
            json.dump(out, f, indent=1)
        self._planeCache.clear(); self._fftCache.clear()
        with self._linkLock:
            self._linkCache[fov] = out
        # optional composite QC images
        if fov in self.parameters['write_composite_FOVs']:
            with self.dataSet.writer_for_analysis_images(self, 'composite', fov) as tif:
                for ch in self.parameters['write_composite_channels']:
                    for z in self.parameters['write_composite_z']:
                        tif.write(self.get_aligned_image(fov, ch, z).astype(np.float32),
                                  photometric='MINISBLACK', contiguous=True)

    # ------------------------------------------------------------------
    # application
    # ------------------------------------------------------------------
    def _links(self, fov: int):
        with self._linkLock:
            if fov in self._linkCache:
                return self._linkCache[fov]
        path = self._links_path(fov)
        with open(path) as f:
            out = json.load(f)
        with self._linkLock:
            self._linkCache[fov] = out
        return out

    def _plane_shift(self, fov: int, dataChannel: int, zIndex: int):
        t = self._polywarp().get_transformation_table(fov)
        row = t[(t['dataChannel'] == dataChannel) & (t['zIndex'] == zIndex)].iloc[0]
        return float(row['yshift']), float(row['xshift'])

    def _own_valid_mask(self, ty: float, tx: float):
        """Pixels of the aligned output that the own warp sampled inside the raw
        frame: output(q) = raw(q + t) valid iff 0 <= q + t < size."""
        h, w = self.dataSet.get_image_dimensions()
        rows = np.arange(h) + ty; cols = np.arange(w) + tx
        vr = (rows >= 0) & (rows <= h - 1); vc = (cols >= 0) & (cols <= w - 1)
        return vr[:, None] & vc[None, :]

    def _neighbor_plane(self, link: dict, dataChannel: int, zIndex: int,
                        chromaticCorrector=None):
        """Neighbour n's own-registered plane (chromatic correction applied in
        n's own camera frame, as PolyWarp does), resampled to k's depth (link dz)
        and shifted into k's frame (link row/col). Zero where n has no data."""
        pw = self._polywarp()
        n = link['n']
        zs = np.array(self.dataSet.get_z_positions(), dtype=float)
        zk = zs[zIndex] + link['dz_um']
        fi = float(np.interp(zk, zs, np.arange(len(zs))))
        lo = int(np.clip(np.floor(fi), 0, len(zs) - 1)); hi = min(lo + 1, len(zs) - 1)
        wgt = fi - lo
        a = pw.get_aligned_image(n, dataChannel, lo, chromaticCorrector).astype(np.float32)
        if hi != lo and wgt > 1e-6:
            b = pw.get_aligned_image(n, dataChannel, hi, chromaticCorrector).astype(np.float32)
            a = (1.0 - wgt) * a + wgt * b
        return self._shift_image(a, link['row'], link['col'])

    def _neighbor_valid_mask(self, link: dict, dataChannel: int, zIndex: int):
        """Where the shifted neighbour plane carries real (own-valid) data."""
        h, w = self.dataSet.get_image_dimensions()
        ty, tx = self._plane_shift(link['n'], dataChannel, zIndex)
        m = self._own_valid_mask(ty, tx).astype(np.float32)
        # erode one pixel so bilinear edges of the shift do not leak zeros
        m = cv2.erode(m, np.ones((3, 3), np.uint8))
        return self._shift_image(m, link['row'], link['col']) > 0.999

    def get_aligned_image(self, fov: int, dataChannel: int, zIndex: int,
                          chromaticCorrector: aberration.ChromaticCorrector = None
                          ) -> np.ndarray:
        pw = self._polywarp()
        own = pw.get_aligned_image(fov, dataChannel, zIndex, chromaticCorrector)
        ty, tx = self._plane_shift(fov, dataChannel, zIndex)
        if max(abs(ty), abs(tx)) <= self.parameters['fill_threshold']:
            return own
        links = [l for l in self._links(fov)['links'] if l['ok']]
        if not links:
            return own
        invalid = ~self._own_valid_mask(ty, tx)
        out = own.astype(np.float32, copy=True)
        filled = np.zeros_like(invalid)
        for link in links:
            need = invalid & ~filled
            if not need.any():
                break
            nvalid = self._neighbor_valid_mask(link, dataChannel, zIndex)
            take = need & nvalid
            if not take.any():
                continue
            nb = self._neighbor_plane(link, dataChannel, zIndex, chromaticCorrector)
            out[take] = nb[take]
            filled |= take
        return out.astype(own.dtype)

    def _fill_sides(self, fov: int):
        """Which invalid sides the OK links can fill: straight neighbours fill a
        side strip, the matching diagonal neighbour fills the corner where two
        strips meet. Returns a dict side -> bool and a set of diagonals."""
        h, w = self.dataSet.get_image_dimensions()
        straight = {'right': False, 'left': False, 'bottom': False, 'top': False}
        diagonal = set()
        for l in self._links(fov)['links']:
            if not l['ok']:
                continue
            ro, co = l['row'], l['col']
            vert = 'bottom' if ro > 0.5 * h else ('top' if ro < -0.5 * h else None)
            horiz = 'right' if co > 0.5 * w else ('left' if co < -0.5 * w else None)
            if vert and horiz:
                diagonal.add((vert, horiz))
            elif vert:
                straight[vert] = True
            elif horiz:
                straight[horiz] = True
        return straight, diagonal

    def _residual_margins(self, fov: int, ty: float, tx: float):
        """Per-plane translation reduced to the margins no neighbour fills.
        Adaptive crop can only express a rectangle, so when both a vertical and
        a horizontal strip are invalid, both are dropped from the crop only if
        the corner between them is filled by the diagonal neighbour; otherwise
        the smaller strip stays cropped so no unfilled corner gets decoded."""
        straight, diagonal = self._fill_sides(fov)
        thr = self.parameters['fill_threshold']
        hside = 'right' if tx > thr else ('left' if tx < -thr else None)
        vside = 'bottom' if ty > thr else ('top' if ty < -thr else None)
        fill_h = hside is not None and straight[hside]
        fill_v = vside is not None and straight[vside]
        if fill_h and fill_v and (vside, hside) not in diagonal:
            # keep the smaller strip cropped
            if abs(tx) >= abs(ty):
                fill_v = False
            else:
                fill_h = False
        rtx = 0.0 if fill_h else tx
        rty = 0.0 if fill_v else ty
        return rty, rtx

    def get_transformation(self, fov: int, dataChannel: int = None,
                           zIndex: int = None):
        table = self._polywarp().get_transformation_table(fov)

        def _transform(row):
            rty, rtx = self._residual_margins(fov, float(row['yshift']), float(row['xshift']))
            return transform.SimilarityTransform(translation=[rtx, rty])

        if dataChannel is not None:
            channelTable = table[table['dataChannel'] == dataChannel]
            if zIndex is None:
                zIndex = len(channelTable) // 2
            row = channelTable[channelTable['zIndex'] == zIndex]
            if len(row) == 0:
                row = channelTable.iloc[[min(zIndex, len(channelTable) - 1)]]
            return _transform(row.iloc[0])
        if zIndex is not None:
            planeTable = table[table['zIndex'] == zIndex]
            return [_transform(r) for _, r in planeTable.iterrows()]
        return [_transform(r) for _, r in table.iterrows()]

    def get_transformation_table(self, fov: int) -> pd.DataFrame:
        return self._polywarp().get_transformation_table(fov)
