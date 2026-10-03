import numpy as np
import pandas
import os
import tempfile
import zarr
import tifffile
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from merlin.core import dataset
from merlin.core import analysistask
from merlin.util import decoding
from merlin.util import barcodedb
from merlin.data.codebook import Codebook
from merlin.util import barcodefilters
from merlin.util import imagesample

# Output channels of a decoded plane, in the order _process_independent_z_slice
# returns them; unique_id only exists with write_unique_id_images.
DECODED_IMAGE_CHANNELS = ('barcodes', 'magnitude', 'distance', 'unique_id')


def compute_crop_bounds(dataSet, warpTaskName, fov, cropWidth, adaptive):
    """(rowStart, rowEnd, colStart, colEnd) of the valid region of one FOV.

    Each edge is trimmed by max(crop_width, that edge's invalid margin), NOT by
    their sum. crop_width exists because the frame edges are unreliable even
    with zero shift (vignetting, the filter's border, chromatic roll-off); the
    adaptive margin exists because a warp leaves an actually-invalid strip. A
    shift smaller than crop_width is already inside the region crop_width
    discards, so adding them double-counts and throws away good pixels: with
    crop_width 100 and a 60 px shift the old code cropped 160.

    transform.warp maps output->input, so output (r, c) samples input
    (r+ty, c+tx): tx<0 invalidates |tx| columns on the LEFT, tx>0 that many on
    the RIGHT, ty<0 rows on the TOP, ty>0 on the BOTTOM.
    """
    h, w = dataSet.get_image_dimensions()
    top = bottom = left = right = 0
    if adaptive:
        warpTask = dataSet.load_analysis_task(warpTaskName)
        tforms = warpTask.get_transformation(fov)
        tx = np.array([t.params[0, 2] for t in tforms])
        ty = np.array([t.params[1, 2] for t in tforms])
        left = int(np.ceil(max(0.0, float((-tx).max()))))
        right = int(np.ceil(max(0.0, float(tx.max()))))
        top = int(np.ceil(max(0.0, float((-ty).max()))))
        bottom = int(np.ceil(max(0.0, float(ty.max()))))
    top, bottom = max(top, cropWidth), max(bottom, cropWidth)
    left, right = max(left, cropWidth), max(right, cropWidth)
    return (top, h - bottom, left, w - right)


class BarcodeSavingParallelAnalysisTask(analysistask.ParallelAnalysisTask):

    """
    An abstract analysis class that saves barcodes into a barcode database.
    """

    def __init__(self, dataSet: dataset.DataSet, parameters=None,
                 analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def _reset_analysis(self, fragmentIndex: int = None) -> None:
        super()._reset_analysis(fragmentIndex)

        # Empty unless resumable decoding is explicitly ON.
        #
        # This used to key off the ABSENCE of 'resumable_z_decoding', with an
        # `elif ... == True` for the present-and-true case -- so
        # present-and-FALSE matched neither branch and the table was neither
        # emptied nor deliberately kept. Re-running such a fragment appended a
        # second copy of every barcode, silently. Measured on 20260609: fov 0
        # of BFDecode and fov 20 of L2DecodeL each held every row exactly
        # twice, which propagates through the filters and inflates counts and
        # any enrichment computed from them. Every Decode task in this project
        # sets the key explicitly, so every one of them was exposed.
        if not self.parameters.get('resumable_z_decoding', False):
            self.parameters['resumable_z_decoding'] = False
            print(f'emptying barcode database for fragment {fragmentIndex}')
            self.get_barcode_database().empty_database(fragmentIndex)
        else:
            print(f'keeping barcode database for fragment {fragmentIndex}')

    def get_barcode_database(self) -> barcodedb.BarcodeDB:
        """ Get the barcode database this analysis task saves barcodes into.

        Returns: The barcode database reference.
        """
        return barcodedb.PyTablesBarcodeDB(self.dataSet, self)


class Decode(BarcodeSavingParallelAnalysisTask):

    """
    An analysis task that extracts barcodes from images.
    """

    outputGroup = 'Decode'

    def __init__(self, dataSet: dataset.MERFISHDataSet,
                 parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        # Which names the caller actually passed, as opposed to the
        # defaults filled in below. The rename migration needs this so an
        # explicit new-style setting always beats a stale old-style one.
        explicitParameters = set(parameters or {})

        # Image filtering now lives entirely in the preprocess task, so that
        # Optimize and Decode cannot disagree about how the pixels were
        # filtered. A lowpass_sigma here is an error rather than a silent
        # no-op, because old configs that set it would otherwise change meaning
        # without warning. Checked first so the message is what the user sees.
        if 'lowpass_sigma' in self.parameters:
            raise ValueError(
                'lowpass_sigma is no longer a Decode parameter -- set it on the '
                'preprocess task instead, where Optimize sees it too.')

        if 'crop_width' not in self.parameters:
            self.parameters['crop_width'] = 100
        if 'crop_in_image_space' not in self.parameters:
            # When True, crop_width is applied in IMAGE space (the image is
            # cropped by crop_width on every edge BEFORE decoding) and ALL
            # decoded barcodes are kept (no barcode-space edge filter). Local
            # x,y stay in the cropped frame; crop_width is added back only when
            # computing global coordinates.
            # Default is image-space: it places the FOV corner at raw pixel
            # (0,0) (crop_offset = +crop_width), matching how segmentation and
            # stage positions are defined. The barcode-space path (False)
            # assumes stage positions are calibrated to pixel (crop_width,
            # crop_width) and subtracts crop_width, which shifts barcodes by
            # -crop_width*mpp relative to the segmentation -> mispartitioning.
            self.parameters['crop_in_image_space'] = True
        if 'write_decoded_images' not in self.parameters:
            self.parameters['write_decoded_images'] = True
        # Default to a few fovs and one z plane rather than everything: a
        # decoded image per fov per plane is enormous, which is why runs used to
        # switch this off entirely and end up with nothing to look at. The fovs
        # and plane come from merlin.util.imagesample with write_images_seed,
        # the same draw Preprocess and the adaptive filters make, so all three
        # write the same fields. write_decoded_z None = every decoded plane.
        if 'write_images_seed' not in self.parameters:
            self.parameters['write_images_seed'] = imagesample.DEFAULT_SEED
        if 'write_decoded_FOVs' not in self.parameters \
                or 'write_decoded_z' not in self.parameters:
            fovs, zIndexes = imagesample.default_image_selection(
                self.dataSet, self.parameters['write_images_seed'])
            self.parameters.setdefault('write_decoded_FOVs', fovs)
            self.parameters.setdefault('write_decoded_z', zIndexes)
        # 'tif': one 2D file per output channel per written plane,
        #   images/decoded_<channel>_fov<fov>_z<z>.tif, channel one of
        #   DECODED_IMAGE_CHANNELS (unique_id only with write_unique_id_images).
        # 'zarr': the previous single (z, channel, y, x) float32 array per fov.
        # Either way read them back with get_decoded_image.
        if 'decoded_image_format' not in self.parameters:
            self.parameters['decoded_image_format'] = 'tif'
        if self.parameters['decoded_image_format'] not in ('tif', 'zarr'):
            raise ValueError("decoded_image_format must be 'tif' or 'zarr'")
        if 'minimum_area' not in self.parameters:
            self.parameters['minimum_area'] = 2
        if 'distance_threshold' not in self.parameters:
            self.parameters['distance_threshold'] = 0.5167
        # Buffer added around each tile so objects spanning a tile edge survive.
        if 'tiling_overlap' not in self.parameters:
            self.parameters['tiling_overlap'] = 20
        # Crop each FOV to its own valid region, on top of crop_width.
        # transform.warp maps output->input, so output (r, c) samples input
        # (r+ty, c+tx): tx<0 invalidates |tx| columns on the LEFT, tx>0 that many
        # on the RIGHT, ty<0 rows on the TOP, ty>0 on the BOTTOM. A fixed
        # crop_width has to be sized for the worst FOV in the dataset and throws
        # that away from every other one; here each FOV pays only what it owes.
        if 'adaptive_crop' not in self.parameters:
            self.parameters['adaptive_crop'] = True
        if 'remove_z_duplicated_barcodes' not in self.parameters:
            self.parameters['remove_z_duplicated_barcodes'] = False
        if self.parameters['remove_z_duplicated_barcodes']:
            if 'z_duplicate_zPlane_threshold' not in self.parameters:
                self.parameters['z_duplicate_zPlane_threshold'] = 1
            if 'z_duplicate_xy_pixel_threshold' not in self.parameters:
                self.parameters['z_duplicate_xy_pixel_threshold'] = np.sqrt(2)
        
        # special case where every FOV was optimized
        if "single_fov_optimization" not in self.parameters:
            self.parameters['single_fov_optimization'] = False
        # only decode in segmentation mask
        if 'use_segmentation_mask' not in self.parameters:
            self.parameters['use_segmentation_mask'] = False

        # gpu decoding
        if 'use_gpu' not in self.parameters:
            self.parameters['use_gpu'] = False

        # write decoded images with unique id channel
        if 'write_unique_id_images' not in self.parameters:
            self.parameters['write_unique_id_images'] = False
            
        # magnitude threshold
        # Default 0 (was 1.0). Magnitude is ||pixel_trace / scale_factors||, so what a
        # fixed cut removes depends entirely on the scale-factor convention: with
        # absolute factors (save_pixel_histogram=True -> ~50-180 here) magnitudes run
        # ~10-100 and a cut at 1.0 is nearly inert, while with normalized mean-1 factors
        # they are ~50x larger and it is fully inert. Same threshold, different meaning
        # per dataset -- MB3 set it to 0 explicitly, M1 and M2 inherited 1.0. 0 makes the
        # behaviour convention-independent; the adaptive filter does the real selection.
        if 'magnitude_threshold' not in self.parameters:
            self.parameters['magnitude_threshold'] = 0.0
        
        # tiling factor for large images to avoid OOM
        if 'tiling_factor' not in self.parameters:
            self.parameters['tiling_factor'] = None
            
        # distance metric
        if 'distance_metric' not in self.parameters:
            self.parameters['distance_metric'] = 'dot_product'
        if 'softmax_temperature' not in self.parameters:
            self.parameters['softmax_temperature'] = 0.15
        if 'decode_chunk_size' not in self.parameters:
            # 8192, not 65536. Measured on a 246x21 codebook, 4M pixels,
            # Xeon 8480CL: at one intra-op thread 65536 costs 1965 ms and the
            # 1024-8192 plateau costs 1765-1792; at eight it costs 279 against
            # 235-243 for 8192-32768. The optimum MOVES with thread count --
            # small when serial, large when parallel -- and 8192 is the value
            # inside both plateaus. Results are bit-identical at every chunk
            # size (verified 100.000000% agreement); this is purely cache
            # behaviour. It also cuts the per-thread similarity buffer 8x,
            # which matters when per_z_slice_num_threads planes run at once.
            self.parameters['decode_chunk_size'] = 8192
             
        # Threads for tile decoding, i.e. how many tiles of ONE z plane are
        # decoded at once. Only meaningful with tiling_factor set.
        if 'tiling_num_threads' not in self.parameters:
            self.parameters['tiling_num_threads'] = 1
        # optional z-index restriction: an int or a list of ints
        if 'decode_z_index' not in self.parameters:
            self.parameters['decode_z_index'] = None
        # Decode is fragmented by fov, so one fov is one process and its z
        # planes run in sequence. They are genuinely independent, so they can
        # be fanned out over a thread pool: the heavy steps (OpenCV filtering,
        # FFTs, the decode matmul, the neighbour search) all release the GIL.
        # Only the barcode append and the zarr write are serialised, under
        # _sliceWriteLock. Memory scales with this: each in-flight plane holds
        # a bitCount x H x W float32 image set, ~350 MB for 21 bits at 2048^2.
        if 'per_z_slice_num_threads' not in self.parameters:
            self.parameters['per_z_slice_num_threads'] = 1
        # None = arbitrate automatically against per_z_slice_num_threads;
        # an integer overrides. See _pin_torch_threads.
        if 'torch_num_threads' not in self.parameters:
            self.parameters['torch_num_threads'] = None

        # These were renamed. task.json records whatever names were current
        # when it was written, so refusing outright would make every existing
        # analysis directory unloadable. Migrate instead, and say so.
        for oldName, newName in (('tile_overlap', 'tiling_overlap'),
                 ('num_threads', 'tiling_num_threads'),
                 ('decode_threads', 'per_z_slice_num_threads')):
            if oldName in self.parameters:
                value = self.parameters.pop(oldName)
                if newName not in explicitParameters:
                    self.parameters[newName] = value
                print('%s was renamed to %s; using %s=%s'
                      % (oldName, newName, newName, self.parameters[newName]))
        # Declares which fovs this decode covers. Purely a declaration -- it
        # does not stop any fragment from running -- but downstream tasks need
        # to know that a partial run is complete on purpose rather than still
        # in progress. GenerateAdaptiveThreshold reads it.
        if 'fovs' not in self.parameters:
            self.parameters['fovs'] = None
        if 'extract_intensity_traces' not in self.parameters:
            self.parameters['extract_intensity_traces'] = False
            
        self.cropWidth = self.parameters['crop_width']
        self.imageSize = dataSet.get_image_dimensions()
        self._sliceWriteLock = threading.Lock()
        # When decoding planes concurrently, barcodes are buffered here and
        # written once at the end instead of per plane. PyTables' flavor.toarray
        # wraps its work in `warnings.catch_warnings(); simplefilter('error')`,
        # and catch_warnings mutates PROCESS-GLOBAL filter state -- so an HDF5
        # write in one thread makes every other thread's DeprecationWarning
        # fatal (pandas .iloc -> np.find_common_type raises, and the fragment
        # dies). A lock cannot fix that: the hazard is a write racing with any
        # other work, not with another write. Keeping PyTables out of the
        # threaded region closes the window entirely.
        self._pendingBarcodes = None

        # method for resumable decoding
        # load the previous barcodes
        # finds unique z planes then can assume that z plane has been decoded

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def _crop_bounds(self, fov: int):
        warpTaskName = self.dataSet.load_analysis_task(
            self.parameters['preprocess_task']).parameters['warp_task']
        return compute_crop_bounds(self.dataSet, warpTaskName, fov,
                                   self.cropWidth,
                                   self.parameters['adaptive_crop'])

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        dependencies = [self.parameters['preprocess_task'],
                        self.parameters['optimize_task'],
                        self.parameters['global_align_task']]
        if self.parameters['use_segmentation_mask']:
            dependencies += [self.parameters['use_segmentation_mask']]
        return dependencies

    def get_codebook(self) -> Codebook:
        preprocessTask = self.dataSet.load_analysis_task(
            self.parameters['preprocess_task'])
        return preprocessTask.get_codebook()

    def _run_analysis(self, fragmentIndex):
        """This function decodes the barcodes in a fov and saves them to the
        barcode database.
        """
        preprocessTask = self.dataSet.load_analysis_task(
                self.parameters['preprocess_task'])
        optimizeTask = self.dataSet.load_analysis_task(
                self.parameters['optimize_task'])

        codebook = self.get_codebook()
        decoder = decoding.PixelBasedDecoder(codebook)

        # for single FOV optimization
        if self.parameters['single_fov_optimization']:
            scaleFactors = optimizeTask.get_scale_factors(fragmentIndex)
            backgrounds = optimizeTask.get_backgrounds(fragmentIndex)
        else:
            scaleFactors = optimizeTask.get_scale_factors()
            backgrounds = optimizeTask.get_backgrounds()
        
        # The corrections the optimize task DECODED UNDER, not the ones estimated
        # afterwards from its own barcodes: its scale factors were fit on images
        # corrected with these, so this is the self-consistent pairing. It is
        # also already cached, so no decode job ever triggers an estimate.
        chromaticCorrector = optimizeTask.get_previous_chromatic_corrector()

        zPositions = self.dataSet.get_z_positions()
        zPositionCount = len(zPositions)
        bitCount = codebook.get_bit_count()
        imageShape = self.dataSet.get_image_dimensions()
        # image-space cropping: decoded images (and zarr) follow the cropped size
        if self.parameters['crop_in_image_space']:
            r0, r1, c0, c1 = self._crop_bounds(fragmentIndex)
            imageShape = (r1 - r0, c1 - c0)
        
        # get decoded image path for a zarr file
        # this may be easier way to save
        
        if self.parameters['write_unique_id_images']:
            zarrChannels = 4
        else:
            zarrChannels = 3
            
        writeZarr = self.parameters['decoded_image_format'] == 'zarr'
        if writeZarr and self.parameters['write_decoded_images'] \
                and (fragmentIndex in self.parameters['write_decoded_FOVs']):
            zarr_path = self.dataSet._analysis_zarr_name(self, "decoded", fragmentIndex)
            zarr_out = zarr.open(zarr_path, mode = 'a',
                shape = (zPositionCount, zarrChannels, *imageShape),
                chunks = (1, zarrChannels, *imageShape),
                dtype = np.float32)
        
        # find what z planes exist in the barcode file already
        self.decoded_z_planes = self._get_decoded_z_planes(fragmentIndex)
        
        decodeZIndexes = self._get_z_indexes_to_decode(zPositionCount)

        # Only planes that are actually decoded can be written. If the
        # requested ones are not among them -- the default [0] against a run
        # restricted to deeper planes, say -- fall back to the first decoded
        # plane so the run still leaves an image behind.
        requestedZ = self.parameters['write_decoded_z']
        if requestedZ is None:
            writeZIndexes = list(decodeZIndexes)
        else:
            writeZIndexes = [z for z in requestedZ if z in decodeZIndexes]
            if not writeZIndexes and decodeZIndexes:
                writeZIndexes = [decodeZIndexes[0]]
                print('write_decoded_z %s is not among the decoded planes; '
                      'writing zIndex %d instead'
                      % (requestedZ, writeZIndexes[0]), flush=True)

        def decode_one_plane(zIndex):
            if zIndex in self.decoded_z_planes:
                print(f'barcodes in zIndex {zIndex} detected. Skipping plane!')
                return

            outputImages = self._process_independent_z_slice(
                fragmentIndex, zIndex, chromaticCorrector, scaleFactors,
                backgrounds, preprocessTask, decoder)

            if self.parameters['write_decoded_images'] \
                    and (fragmentIndex in self.parameters['write_decoded_FOVs']) \
                    and zIndex in writeZIndexes:
                with self._sliceWriteLock:
                    if writeZarr:
                        zarr_out[zIndex,0,:,:] = outputImages[0]
                        zarr_out[zIndex,1,:,:] = outputImages[1]
                        zarr_out[zIndex,2,:,:] = outputImages[2]
                        if self.parameters['write_unique_id_images']:
                            zarr_out[zIndex,3,:,:] = outputImages[3]
                    else:
                        self._write_decoded_tifs(
                            fragmentIndex, zIndex, outputImages)

        perZThreads, tilingThreads, _ = self._thread_allocation()
        self._pin_torch_threads(perZThreads)
        if perZThreads > 1 and len(decodeZIndexes) > 1:
            print('decoding %d z planes over %d threads (%d tile threads each)'
                  % (len(decodeZIndexes), perZThreads, tilingThreads),
                  flush=True)
            self._pendingBarcodes = []
            try:
                with ThreadPoolExecutor(
                        max_workers=min(perZThreads,
                                        len(decodeZIndexes))) as pool:
                    list(pool.map(decode_one_plane, decodeZIndexes))
                pending = [d for d in self._pendingBarcodes if len(d)]
            finally:
                self._pendingBarcodes = None
            if pending:
                self.get_barcode_database().write_barcodes(
                    pandas.concat(pending, ignore_index=True),
                    fov=fragmentIndex)
        else:
            for zIndex in decodeZIndexes:
                decode_one_plane(zIndex)

        if self.parameters['remove_z_duplicated_barcodes']:
            bcDB = self.get_barcode_database()
            bc = self._remove_z_duplicate_barcodes(
                bcDB.get_barcodes(fov=fragmentIndex))
            bcDB.empty_database(fragmentIndex)
            bcDB.write_barcodes(bc, fov=fragmentIndex)

    def _emit_barcodes(self, df, fov: int) -> None:
        """Hand a plane's barcodes on: buffered while a thread pool is live,
        written straight through otherwise."""
        if self._pendingBarcodes is not None:
            with self._sliceWriteLock:
                self._pendingBarcodes.append(df)
            return
        self.get_barcode_database().write_barcodes(df, fov=fov)

    def _thread_allocation(self):
        """How the two nested pools and the neighbour search share the cores.

        There are three things that can fan out: z planes within a fov, tiles
        within a z plane, and scikit-learn's neighbour search within a tile.
        Nesting them multiplies, so they have to be arbitrated rather than each
        taking what it wants.

        Per-z has first claim, because it is the one that scales -- planes are
        wholly independent and each is a large unit of work, whereas tiling
        exists to bound memory and the neighbour search saturates early. So
        when per_z_slice_num_threads > 1 the inner two are pinned to one
        thread; only when it is 1 does tiling get to fan out, and only when
        both are 1 does the neighbour search get every core (n_jobs=-1, which
        is what it did unconditionally before).

        Returns (perZThreads, tilingThreads, neighborNumJobs).
        """
        perZThreads = max(1, int(self.parameters['per_z_slice_num_threads']))
        tilingThreads = max(1, int(self.parameters['tiling_num_threads']))
        if perZThreads > 1:
            tilingThreads = 1
        neighborNumJobs = 1 if (perZThreads > 1 or tilingThreads > 1) else -1
        return perZThreads, tilingThreads, neighborNumJobs

    def _pin_torch_threads(self, perZThreads: int) -> None:
        """Arbitrate torch's intra-op pool against the per-z thread pool.

        The existing arbitration covers tiling and the neighbour search but not
        torch, which is the one that actually does the decode. Left alone torch
        sizes its pool to the whole machine, so per_z_slice_num_threads planes
        each ask for every core and contend.

        Measured, 16 planes of 500k pixels, 16 cores, chunk 4096:

            torch left at default (16 intra-op)   364.2 ms   1.37x scaling
            torch pinned to 1                     254.6 ms  14.12x scaling

        so pinning is worth 1.43x on the decode core and costs nothing. The
        single-plane numbers show why it is conditional rather than always:
        one plane alone takes 31.1 ms on 16 intra-op threads and 224.7 ms on
        one, so when there is no per-z pool torch should keep the machine.

        (numpy is not an alternative here -- under the same pattern it does not
        scale at all, 1.05x over 16 planes, because a multi-threaded OpenBLAS
        gemm called from many python threads serialises. 1738 ms against 255.)
        """
        requested = self.parameters.get('torch_num_threads')
        if not decoding.TORCH_AVAILABLE:
            return
        if requested is None:
            requested = 1 if perZThreads > 1 else None
        if requested is None:
            return
        try:
            decoding.torch.set_num_threads(max(1, int(requested)))
        except Exception as e:
            print('could not set torch threads: %s' % e, flush=True)

    def get_decoded_image_path(self, fov: int, zIndex: int,
                               channel: str = 'barcodes') -> str:
        if channel not in DECODED_IMAGE_CHANNELS:
            raise ValueError('channel must be one of %s'
                             % (DECODED_IMAGE_CHANNELS,))
        return os.sep.join([
            self.dataSet.get_analysis_subdirectory(self, subdirectory='images'),
            'decoded_%s_fov%d_z%d.tif' % (channel, fov, zIndex)])

    def _write_decoded_tifs(self, fov: int, zIndex: int, outputImages) -> None:
        for channel, image in zip(DECODED_IMAGE_CHANNELS, outputImages):
            tifffile.imwrite(self.get_decoded_image_path(fov, zIndex, channel),
                             np.asarray(image, dtype=np.float32),
                             photometric='minisblack')

    def get_decoded_image(self, fov: int, zIndex: int,
                          channel: str = 'barcodes'):
        """One decoded plane of one output channel, from whichever format it
        was written in (tif, or the older zarr). None if it was not written.

        'barcodes' is the codeword index per pixel (-1 where nothing decoded),
        'magnitude' the pixel trace norm, 'distance' the distance to that
        codeword, 'unique_id' the barcode label (write_unique_id_images only).
        """
        path = self.get_decoded_image_path(fov, zIndex, channel)
        if os.path.exists(path):
            return tifffile.imread(path)
        zarrPath = self.dataSet._analysis_zarr_name(self, 'decoded', fov)
        if os.path.exists(zarrPath):
            source = zarr.open(zarrPath, mode='r')
            channelIndex = DECODED_IMAGE_CHANNELS.index(channel)
            if zIndex < source.shape[0] and channelIndex < source.shape[1]:
                return np.asarray(source[zIndex, channelIndex])
        return None

    def get_decoded_image_z(self, fov: int):
        """z indexes with a decoded tif for this fov, or None when the fov
        has an older zarr instead -- a zarr does not record which of its
        planes were written."""
        tifZ = [z for z in range(len(self.dataSet.get_z_positions()))
                if os.path.exists(self.get_decoded_image_path(fov, z))]
        if tifZ:
            return tifZ
        if os.path.exists(self.dataSet._analysis_zarr_name(
                self, 'decoded', fov)):
            return None
        return []

    def _get_z_indexes_to_decode(self, zPositionCount: int) -> list[int]:
        decodeZIndex = self.parameters.get('decode_z_index')
        if decodeZIndex is None:
            return list(range(zPositionCount))
        if isinstance(decodeZIndex, (list, tuple)):
            requested = [int(z) for z in decodeZIndex]
        else:
            requested = [int(decodeZIndex)]
        for z in requested:
            if z < 0 or z >= zPositionCount:
                raise ValueError(
                    f'decode_z_index {z} out of range for '
                    f'{zPositionCount} z-positions')
        return sorted(set(requested))

    # finding what z planes are already in the barcode file
    def _get_decoded_z_planes(self, fragmentIndex):
            if self.parameters['resumable_z_decoding']:
                    print('resumable decoding enabled!\nbarcode files are not emptied!')
                    bcDB = self.get_barcode_database()
                    bcs = bcDB.get_barcodes(fov=fragmentIndex)
                    decoded_z_planes = bcs.z.unique()
            else:
                decoded_z_planes = [] # otherwise set to empty so all z planes are decoded
            return decoded_z_planes

    # used to load in the segmentation mask
    def _get_segmentation_mask(self, fovIndex, zIndex):
        segmentTask = self.dataSet.load_analysis_task(
            self.parameters['use_segmentation_mask'])
        return segmentTask._load_mask_image(fovIndex, zIndex)

    def _process_independent_z_slice(
            self, fov: int, zIndex: int, chromaticCorrector, scaleFactors,
            backgrounds, preprocessTask, decoder):

        t0 = time.time()
        imageSet = preprocessTask.get_processed_image_set(
            fov, zIndex, chromaticCorrector)
        imageSet = imageSet.reshape(
            (imageSet.shape[0], imageSet.shape[-2], imageSet.shape[-1]))
        # image-space crop: trim the invalid margin before decoding
        if self.parameters['crop_in_image_space']:
            r0, r1, c0, c1 = self._crop_bounds(fov)
            imageSet = imageSet[:, r0:r1, c0:c1]
        t1 = time.time()

        decodeMask = None
        if self.parameters['use_segmentation_mask']:
            decodeMask = self._get_segmentation_mask(fov, zIndex)
        
        accumulatePixelTraces = self.parameters['extract_intensity_traces']
        onTileDone = None
        dfs = []
        
        # If tiling is used, we want to extract barcodes per tile to avoid memory overhead
        if self.parameters['tiling_factor'] is not None and self.parameters['tiling_factor'] > 1:
            accumulatePixelTraces = False
            tileResults = []
            
            def process_tile(tDi, tPm, tNpt, tD, sliceInfo, validBBox):
                # We need to extract barcodes from the tile
                # cropWidth is set to 0 here because we handle the overlap/validity manually using valid_bbox
                outputLabels = self.parameters['write_unique_id_images']
                minimumArea = self.parameters['minimum_area']

                tileBarcodeOutput = decoder.extract_barcodes_with_index(
                    tDi, tPm, tNpt, tD, fov, cropWidth=0, zIndex=zIndex,
                    globalAligner=None, minimumArea=minimumArea,
                    outputLabels=outputLabels,
                    extractIntensityTraces=self.parameters[
                        'extract_intensity_traces'])

                labels = None
                if outputLabels:
                    dfTile, labels = tileBarcodeOutput 
                else:
                    dfTile = tileBarcodeOutput
                
                # Store strictly local data for sequential post-processing
                # This avoids concurrency issues with ID generation
                tileResults.append((dfTile, labels, sliceInfo, validBBox))
                    
            onTileDone = process_tile

        di, pm, npt, d = decoder.decode_pixels(
            imageSet, scaleFactors, backgrounds,
            lowPassSigma=0,
            tilingOverlap=self.parameters['tiling_overlap'],
            magnitudeThreshold=self.parameters['magnitude_threshold'],
            distanceThreshold=self.parameters['distance_threshold'],
            distanceMetric=self.parameters['distance_metric'],
            softmaxTemperature=self.parameters['softmax_temperature'],
            decodeChunkSize=self.parameters['decode_chunk_size'],
            nnAlgorithm=self.parameters.get('nn_algorithm', 'brute'),
            decodeMask = decodeMask,
            tilingNumThreads = self._thread_allocation()[1],
            neighborNumJobs = self._thread_allocation()[2],
            useGpu = self.parameters['use_gpu'],
            tilingFactor = self.parameters['tiling_factor'],
            accumulatePixelTraces = accumulatePixelTraces,
            onTileDone = onTileDone)
        
        t2 = time.time()
        
        uid = None
        if self.parameters['tiling_factor'] is not None and self.parameters['tiling_factor'] > 1:
            dfs = []
            currentMaxId = 0
            if self.parameters['write_unique_id_images']:
                uid = np.zeros_like(di, dtype=np.int32)
            
            for res in tileResults:
                dfTile, labels, sliceInfo, validBBox = res
                hStart, hEnd, wStart, wEnd = sliceInfo
                vHMin, vHMax, vWMin, vWMax = validBBox
                
                if len(dfTile) > 0:
                     # Filter DataFrame for overlap (using local coords) based on centroids
                     validMask = (
                        (dfTile['y'] >= vHMin) & (dfTile['y'] < vHMax) &
                        (dfTile['x'] >= vWMin) & (dfTile['x'] < vWMax)
                     )
                     dfTile = dfTile[validMask].copy()
                
                if len(dfTile) > 0:
                     # Generate global unique IDs
                     oldIds = dfTile['unique_id'].values
                     newIds = np.arange(currentMaxId + 1, currentMaxId + 1 + len(dfTile), dtype=np.int32)
                     
                     # Map IDs in DataFrame
                     dfTile['unique_id'] = newIds
                     currentMaxId += len(dfTile)
                     
                     # Map IDs in Label Image (if needed)
                     if uid is not None and labels is not None:
                         # Crop to valid region
                         lCrop = labels[vHMin:vHMax, vWMin:vWMax]
                         
                         # Create LUT for fast mapping
                         # Pixels not in old_ids (i.e. outside valid_bbox centroids) become 0
                         maxLabel = labels.max()
                         if maxLabel > 0:
                             lut = np.zeros(maxLabel + 1, dtype=np.int32)
                             lut[oldIds] = newIds
                             lMapped = lut[lCrop]
                             
                             # Paste into global image
                             gHStart = hStart + vHMin
                             gHEnd = hStart + vHMax
                             gWStart = wStart + vWMin
                             gWEnd = wStart + vWMax
                             
                             uid[gHStart:gHEnd, gWStart:gWEnd] = lMapped

                     # Adjust coordinates to global image frame
                     dfTile.loc[:, 'x'] += wStart
                     dfTile.loc[:, 'y'] += hStart
                     dfs.append(dfTile)
                     
            if len(dfs) > 0:
                df = pandas.concat(dfs, ignore_index=True)
                
                # Apply cropWidth filter on the full image coordinates
                cw = self.cropWidth
                if cw > 0:
                     df = df[(df['x'].between(cw, di.shape[1] - cw)) &
                             (df['y'].between(cw, di.shape[0] - cw))]
                
                # Apply global alignment
                globalTask = self.dataSet.load_analysis_task(self.parameters['global_align_task'])
                # Calculate global coordinates
                if len(df) > 0:
                     # FOV corner = raw pixel (0,0), matching segmentation and
                     # positions.csv. In image-space mode the image was trimmed by
                     # cropWidth before decoding, so cropWidth is added back; in
                     # barcode-space mode x,y are already full-frame, so no offset.
                     # Matches the equivalent cropOffset logic in _extract_and_save_barcodes.
                     if self.parameters['crop_in_image_space'] and cw > 0:
                         cropOffset = cw
                     else:
                         cropOffset = 0
                     centroids = np.zeros([len(df), 3], dtype=np.float32)
                     centroids[:, 0] = df['z'].values
                     centroids[:, 1] = df['x'].values + cropOffset # col
                     centroids[:, 2] = df['y'].values + cropOffset # row

                     g = globalTask.fov_coordinate_array_to_global(fov, centroids)
                     df['global_z'] = g[:, 0]
                     df['global_x'] = g[:, 1]
                     df['global_y'] = g[:, 2]
                
            else:
                df = pandas.DataFrame()

            self._emit_barcodes(df, fov)
            
            barcodeOutputs = df

        else:
             barcodeOutputs = self._extract_and_save_barcodes(
                decoder, di, pm, npt, d, fov, zIndex)
        
        if self.parameters['write_unique_id_images']:
            if self.parameters['tiling_factor'] is not None and self.parameters['tiling_factor'] > 1:
                 # uid is already constructed above
                 return di, pm, d, uid
            else:
                 df, uid = barcodeOutputs 
                 return di,pm,d,uid
        else:
            df = barcodeOutputs 
        t3 = time.time()
        print(f'time retrieving fov {fov} zindex {zIndex}: {t1-t0}')
        print(f'time decoding fov {fov} zindex {zIndex}: {t2-t1}')
        if getattr(decoder, 'last_decode_timings', None):
            for stageName, stageTime in decoder.last_decode_timings.items():
                print(
                    f'decode stage {stageName} fov {fov} zindex {zIndex}: '
                    f'{stageTime}'
                )
        print(f'time extracting fov {fov} zindex {zIndex}: {t3-t2}')
        print(f'total time in fov {fov} zindex {zIndex}: {t3-t0}')
        
        if self.parameters['write_unique_id_images']:
            return di,pm,d,uid
            
        return di,pm,d
        
    # leave this for 3d decoding, currently zarr is used for easier resumable decoding
    def _save_decoded_images(self, fov: int, zPositionCount: int,
                             decodedImages: np.ndarray,
                             magnitudeImages: np.ndarray,
                             distanceImages: np.ndarray) -> None:
            imageDescription = self.dataSet.analysis_tiff_description(
                zPositionCount, 3)
            with self.dataSet.writer_for_analysis_images(
                    self, 'decoded', fov) as outputTif:
                for i in range(zPositionCount):
                    outputTif.write(decodedImages[i].astype(np.float32),
                                   photometric='MINISBLACK',
                                   contiguous=True,
                                   metadata=imageDescription)
                    outputTif.write(magnitudeImages[i].astype(np.float32),
                                   photometric='MINISBLACK',
                                   contiguous=True,
                                   metadata=imageDescription)
                    outputTif.write(distanceImages[i].astype(np.float32),
                                   photometric='MINISBLACK',
                                   contiguous=True,
                                   metadata=imageDescription)

    def _extract_and_save_barcodes(
            self, decoder: decoding.PixelBasedDecoder, decodedImage: np.ndarray,
            pixelMagnitudes: np.ndarray, pixelTraces: np.ndarray,
            distances: np.ndarray, fov: int, zIndex: int=None) -> None:

        globalTask = self.dataSet.load_analysis_task(
            self.parameters['global_align_task'])
        minimumArea = self.parameters['minimum_area']
        outputLabels = self.parameters['write_unique_id_images']
        
        if self.parameters['crop_in_image_space']:
            # keep all barcodes; the image is already cropped. x is a column and
            # y is a row, so the offsets that put them back in the full frame are
            # colStart and rowStart respectively -- not one shared value.
            r0, _, c0, _ = self._crop_bounds(fov)
            effCropWidth, cropOffset = 0, (c0, r0)
        else:
            # Barcode-space: image is NOT cropped; x,y are full-frame and edge
            # barcodes are removed via effCropWidth. FOV corner = raw pixel (0,0)
            # (same convention as segmentation and positions.csv), so no offset
            # is applied to global coordinates.
            effCropWidth, cropOffset = self.cropWidth, (0, 0)
        barcodeOutput = decoder.extract_barcodes_with_index(
            decodedImage, pixelMagnitudes, pixelTraces, distances, fov,
            effCropWidth, zIndex, globalTask, minimumArea, outputLabels,
            extractIntensityTraces=self.parameters['extract_intensity_traces'],
            crop_offset=cropOffset)
        
        if outputLabels:
            df, uid = barcodeOutput
        else:
            df = barcodeOutput
            
        self._emit_barcodes(df, fov)
        
        return barcodeOutput
        
    def _remove_z_duplicate_barcodes(self, bc):
        bc = barcodefilters.remove_zplane_duplicates_all_barcodeids(
            bc, self.parameters['z_duplicate_zPlane_threshold'],
            self.parameters['z_duplicate_xy_pixel_threshold'],
            self.dataSet.get_z_positions())
        return bc
