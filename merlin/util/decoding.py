import numpy as np
import pandas
import cv2
import time
from typing import Tuple
from typing import Dict
from skimage import measure
from skimage import morphology
from sklearn.neighbors import NearestNeighbors
import gc
from concurrent.futures import ThreadPoolExecutor

from merlin.util import binary
from merlin.data import codebook as mcodebook

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

"""
Utility functions for pixel based decoding.
"""

def normalize(x):
    norm = np.linalg.norm(x)
    if norm > 0:
        return x/norm
    else:
        return x

# Try to import cupy for GPU decoding
# installing is annoying and only beneficial in limited (long codebook cases)
try:
    import cupy as cp
    from cupyx.scipy.spatial.distance import cdist

    # gpu nearest neighbor
    def calculate_distances_gpu(pixel_traces, codebook_matrix):
    
            if isinstance(pixel_traces, np.ndarray):
                pixel_traces = cp.asarray(pixel_traces,dtype=cp.float32)
            if isinstance(codebook_matrix, np.ndarray):
                codebook_matrix = cp.asarray(codebook_matrix,dtype=cp.float32)
            
            distances = cdist(cp.ascontiguousarray(pixel_traces),cp.ascontiguousarray(codebook_matrix), metric='euclidean')
            #distances = cdist(cp.ascontiguousarray(pixel_traces.T),cp.ascontiguousarray(codebook_matrix), metric='euclidean')
            min_indices = cp.argmin(distances, axis=1)
            min_distances = cp.min(distances, axis=1)

            del pixel_traces, codebook_matrix
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            
            return cp.asnumpy(min_distances), cp.asnumpy(min_indices)
except ImportError:
    # is this the right way of doing this...
    print('cupy not found, GPU decoding disabled')
    def calculate_distances_gpu():
        pass

"""
Back to Decoder class
"""

class PixelBasedDecoder(object):

    def __init__(self, codebook: mcodebook.Codebook,
                 scaleFactors: np.ndarray=None, backgrounds: np.ndarray=None):
        self._codebook = codebook
        self._decodingMatrix = self._calculate_normalized_barcodes()
        self._barcodeCount = self._decodingMatrix.shape[0]
        self._bitCount = self._decodingMatrix.shape[1]

        if scaleFactors is None:
            self._scaleFactors = np.ones(self._decodingMatrix.shape[1])
        else:
            self._scaleFactors = scaleFactors.copy()

        if backgrounds is None:
            self._backgrounds = np.zeros(self._decodingMatrix.shape[1])
        else:
            self._backgrounds = backgrounds.copy()

        self.refactorAreaThreshold = 4 # this gets reset in optimize task!
        self.last_decode_timings = {}
        self.last_softmax_top1_probability = None

    def _decode_pixels_by_similarity_numpy(
            self, pixels_to_decode: np.ndarray, chunk_size: int,
            softmax_temperature: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        best_indices = np.empty(len(pixels_to_decode), dtype=np.int32)
        best_dists = np.empty(len(pixels_to_decode), dtype=np.float32)

        if softmax_temperature is not None:
            top1_probability = np.empty(len(pixels_to_decode), dtype=np.float32)
            safe_temperature = max(float(softmax_temperature), 1e-8)
        else:
            top1_probability = None

        for start in range(0, len(pixels_to_decode), chunk_size):
            stop = min(start + chunk_size, len(pixels_to_decode))
            chunk = pixels_to_decode[start:stop]
            similarities = np.dot(chunk, self._decodingMatrix.T).astype(
                np.float32, copy=False)
            chunk_indices = np.argmax(similarities, axis=1).astype(np.int32)
            chunk_scores = similarities[
                np.arange(similarities.shape[0]), chunk_indices]
            chunk_scores = np.minimum(chunk_scores, 1.0)

            best_indices[start:stop] = chunk_indices
            best_dists[start:stop] = np.sqrt(
                np.maximum(2.0 * (1.0 - chunk_scores), 0.0)
            ).astype(np.float32, copy=False)

            if top1_probability is not None:
                scaled = similarities / safe_temperature
                scaled -= np.max(scaled, axis=1, keepdims=True)
                np.exp(scaled, out=scaled)
                denom = np.sum(scaled, axis=1)
                top1_probability[start:stop] = (
                    scaled[np.arange(scaled.shape[0]), chunk_indices] / denom
                ).astype(np.float32, copy=False)

        return best_indices, best_dists, top1_probability

    def _decode_pixels_by_similarity_torch(
            self, pixels_to_decode: np.ndarray, chunk_size: int,
            softmax_temperature: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not TORCH_AVAILABLE:
            return self._decode_pixels_by_similarity_numpy(
                pixels_to_decode, chunk_size, softmax_temperature)

        best_indices = np.empty(len(pixels_to_decode), dtype=np.int32)
        best_dists = np.empty(len(pixels_to_decode), dtype=np.float32)

        if softmax_temperature is not None:
            top1_probability = np.empty(len(pixels_to_decode), dtype=np.float32)
            safe_temperature = max(float(softmax_temperature), 1e-8)
        else:
            top1_probability = None
            safe_temperature = None

        codebook_tensor = torch.from_numpy(
            self._decodingMatrix.astype(np.float32, copy=False))

        with torch.inference_mode():
            for start in range(0, len(pixels_to_decode), chunk_size):
                stop = min(start + chunk_size, len(pixels_to_decode))
                chunk_tensor = torch.from_numpy(
                    pixels_to_decode[start:stop].astype(np.float32, copy=False))
                similarities = torch.matmul(chunk_tensor, codebook_tensor.T)
                chunk_scores, chunk_indices = torch.max(similarities, dim=1)
                chunk_scores = torch.clamp(chunk_scores, max=1.0)
                chunk_dists = torch.sqrt(torch.clamp(
                    2.0 * (1.0 - chunk_scores), min=0.0))

                best_indices[start:stop] = chunk_indices.numpy().astype(np.int32, copy=False)
                best_dists[start:stop] = chunk_dists.numpy().astype(np.float32, copy=False)

                if top1_probability is not None:
                    probabilities = torch.softmax(
                        similarities / safe_temperature, dim=1)
                    chunk_prob = probabilities.gather(
                        1, chunk_indices.unsqueeze(1)).squeeze(1)
                    top1_probability[start:stop] = chunk_prob.numpy().astype(
                        np.float32, copy=False)

        return best_indices, best_dists, top1_probability
                       
    def decode_pixels(self, imageData: np.ndarray,
                      scaleFactors: np.ndarray=None,
                      backgrounds: np.ndarray=None,
                      distanceThreshold: float=0.5176,
                      magnitudeThreshold: float=1.0,
                      lowPassSigma: float=1.0,
                      distanceMetric = None,
                      softmaxTemperature: float=0.15,
                      decodeChunkSize: int=65536,
                      nnAlgorithm = 'brute',
                      decodeMask = None,
                      useGpu = False,
                      tilingFactor = None,
                      accumulatePixelTraces = True,
                      onTileDone = None,
                      numThreads = 1,
                      overlap = None):
        """Assign barcodes to the pixels in the provided image stock.

        Each pixel is assigned to the nearest barcode from the codebook if
        the distance between the normalized pixel trace and the barcode is
        less than the distance threshold.

        Args:
            imageData: input image stack. The first dimension indexes the bit
                number and the second and third dimensions contain the
                corresponding image.
            scaleFactors: factors to rescale each bit prior to normalization.
                The length of scaleFactors must be equal to the number of bits.
            backgrounds: background to subtract from each bit prior to applying
                the scale factors and prior to normalization. The length of
                backgrounds must be equal to the number of bits.
            distanceThreshold: the maximum distance between an assigned pixel
                and the nearest barcode. Pixels for which the nearest barcode
                is greater than distanceThreshold are left unassigned.
            magnitudeThreshold: the minimum pixel magnitude for which a
                barcode can be assigned that pixel. All pixels that fall
                below the magnitude threshold are not assigned a barcode
                in the decoded image.
            lowPassSigma: standard deviation for the low pass filter that is
                applied to the images prior to decoding.
            overlap: (Optional) Buffer size for tile overlaps. if None,
                defaults to 2x filter size.
            
        Returns:
            Four results are returned as a tuple (decodedImage, pixelMagnitudes,
                normalizedPixelTraces, distances). decodedImage is an image
                indicating the barcode index assigned to each pixel. Pixels
                for which a barcode is not assigned have a value of -1.
                pixelMagnitudes is an image where each pixel is the norm of
                the pixel trace after scaling by the provided scaleFactors.
                normalizedPixelTraces is an image stack containing the
                normalized intensities for each pixel. distances is an
                image containing the distance for each pixel to the assigned
                barcode.
        """
        self.last_decode_timings = {}
        self.last_softmax_top1_probability = None

        if tilingFactor is not None and tilingFactor > 1:
            image_shape = imageData.shape[1:]
            
            # prepare outputs
            decodedImage = np.zeros(image_shape, dtype=np.int32)
            pixelMagnitudes = np.zeros(image_shape, dtype=np.float32)
            if accumulatePixelTraces:
                normalizedPixelTraces = np.zeros((imageData.shape[0], *image_shape), dtype=np.float32)
            else:
                normalizedPixelTraces = None
            distanceImage = np.zeros(image_shape, dtype=np.float32)
            
            # tile iterations
            full_height = image_shape[0]
            full_width = image_shape[1]
            
            if overlap is None:
                 # Default overlap to cover filter size + safety margin
                 overlap = int(2 * np.ceil(2 * lowPassSigma) + 1) + 10
            
            # check if divisible
            if full_height % tilingFactor != 0 or full_width % tilingFactor != 0:
                 print(f"Warning: Image size ({full_height}, {full_width}) is not divisible by tiling factor {tilingFactor}. truncating last tiles.")

            tile_height = int(full_height // tilingFactor)
            tile_width = int(full_width // tilingFactor)
            
            # Pad size for overlap to handle boundary objects
            # Sufficient to cover filter checks and spot sizes
            
            # Define the tile processing function for parallel execution
            def process_single_tile(params):
                index_h, index_w = params
                
                h_start = index_h * tile_height
                h_end = (index_h + 1) * tile_height
                
                if index_h == tilingFactor - 1:
                    h_end = full_height

                # Calculate padded bounds
                h_start_pad = max(0, h_start - overlap)
                h_end_pad = min(full_height, h_end + overlap)
                
                w_start = index_w * tile_width
                w_end = (index_w + 1) * tile_width

                if index_w == tilingFactor - 1:
                    w_end = full_width
                
                w_start_pad = max(0, w_start - overlap)
                w_end_pad = min(full_width, w_end + overlap)

                # Extract tile with padding
                # Note: This slice creates a copy/view depending on memory layout
                # For threading, this read is safe.
                tile_image_data = imageData[:, h_start_pad:h_end_pad, w_start_pad:w_end_pad]
                
                tile_decode_mask = None
                if decodeMask is not None:
                    tile_decode_mask = decodeMask[h_start_pad:h_end_pad, w_start_pad:w_end_pad]
                    
                # Recurse for the tile
                # If we are parallelizing tiles, we shouldn't parallelize neighbors too aggressively
                # to avoid oversubscription. If numThreads > 1 (parallel tiles), inner numThreads should be 1.
                inner_numThreads = 1 if numThreads > 1 else -1
                
                t_di, t_pm, t_npt, t_dist = self.decode_pixels(
                    tile_image_data,
                    scaleFactors=scaleFactors, 
                    backgrounds=backgrounds,
                    distanceThreshold=distanceThreshold, 
                    magnitudeThreshold=magnitudeThreshold,
                    lowPassSigma=lowPassSigma,
                    distanceMetric=distanceMetric,
                    softmaxTemperature=softmaxTemperature,
                    decodeChunkSize=decodeChunkSize,
                    nnAlgorithm=nnAlgorithm,
                    decodeMask=tile_decode_mask,
                    useGpu=useGpu, 
                    tilingFactor=None,
                    numThreads=inner_numThreads 
                )
                
                # Offsets relative to the padded tile
                inner_h_start = h_start - h_start_pad
                inner_w_start = w_start - w_start_pad
                inner_h_end = inner_h_start + (h_end - h_start)
                inner_w_end = inner_w_start + (w_end - w_start)
                
                # Callback for extraction (happens in thread)
                if onTileDone is not None:
                    validBBox = (inner_h_start, inner_h_end, inner_w_start, inner_w_end)
                    onTileDone(t_di, t_pm, t_npt, t_dist, (h_start_pad, h_end_pad, w_start_pad, w_end_pad), validBBox)
                    
                # Crop back to original tile size for stitching
                results = {
                    'di': t_di[inner_h_start:inner_h_end, inner_w_start:inner_w_end],
                    'pm': t_pm[inner_h_start:inner_h_end, inner_w_start:inner_w_end],
                    'dist': t_dist[inner_h_start:inner_h_end, inner_w_start:inner_w_end],
                    'npt': None,
                    'coords': (h_start, h_end, w_start, w_end)
                }
                
                if accumulatePixelTraces and t_npt is not None:
                    results['npt'] = t_npt[:, inner_h_start:inner_h_end, inner_w_start:inner_w_end]
                    
                return results

            # Generate task list
            tile_indices = [(h, w) for h in range(tilingFactor) for w in range(tilingFactor)]
            
            # Execute
            if numThreads > 1:
                with ThreadPoolExecutor(max_workers=numThreads) as executor:
                    futures = executor.map(process_single_tile, tile_indices)
                    
                    for res in futures:
                        h_s, h_e, w_s, w_e = res['coords']
                        decodedImage[h_s:h_e, w_s:w_e] = res['di']
                        pixelMagnitudes[h_s:h_e, w_s:w_e] = res['pm']
                        distanceImage[h_s:h_e, w_s:w_e] = res['dist']
                        if accumulatePixelTraces and res['npt'] is not None:
                             normalizedPixelTraces[:, h_s:h_e, w_s:w_e] = res['npt']
            else:
                # Sequential fallback (avoids overhead)
                for indices in tile_indices:
                    res = process_single_tile(indices)
                    h_s, h_e, w_s, w_e = res['coords']
                    decodedImage[h_s:h_e, w_s:w_e] = res['di']
                    pixelMagnitudes[h_s:h_e, w_s:w_e] = res['pm']
                    distanceImage[h_s:h_e, w_s:w_e] = res['dist']
                    if accumulatePixelTraces and res['npt'] is not None:
                            normalizedPixelTraces[:, h_s:h_e, w_s:w_e] = res['npt']
            
            return decodedImage, pixelMagnitudes, normalizedPixelTraces, distanceImage
             
        if scaleFactors is None:
            scaleFactors = self._scaleFactors
        if backgrounds is None:
            backgrounds = self._backgrounds
             
        # the dimensions are num_bits x image_rows x image_cols
        t0 = time.perf_counter()
        if lowPassSigma == 0:
            filteredImages = imageData.astype(np.float32, copy=True)
        else:
            filteredImages = np.zeros(imageData.shape, dtype=np.float32)
            filterSize = int(2 * np.ceil(2 * lowPassSigma) + 1)
            for i in range(imageData.shape[0]):
                filteredImages[i, :, :] = cv2.GaussianBlur(
                    imageData[i, :, :], (filterSize, filterSize), lowPassSigma)
        self.last_decode_timings['lowpass_seconds'] = (
            time.perf_counter() - t0)

        # going to try to make this more numpy-ish
        # and add decode mask parameter, ie decode only in segmented regions...

        t1 = time.perf_counter()
        image_shape = filteredImages.shape[1:] # dimensions of one image plane
        scaledPixelTraces = ((filteredImages-backgrounds[:,None,None])/scaleFactors[:,None,None]).astype(np.float32)
        pixelMagnitudes = np.linalg.norm(scaledPixelTraces, axis = 0).astype(np.float32)
        pixelMagnitudes[pixelMagnitudes == 0] = 1.0
        normalizedPixelTraces = scaledPixelTraces/pixelMagnitudes # this should be a float32 to save a little mem
        self.last_decode_timings['scale_normalize_seconds'] = (
            time.perf_counter() - t1)
        
        # Flatten everything for uniform processing (simplifies mask/no-mask logic)
        rows, cols = image_shape
        N = rows * cols
        
        flat_traces = normalizedPixelTraces.reshape(normalizedPixelTraces.shape[0], -1).T
        flat_mags = pixelMagnitudes.reshape(-1)
        
        # Initialize outputs
        flat_di = np.full(N, -1, dtype=np.int32)
        flat_dist = np.full(N, 2.0, dtype=np.float32)

        # Identify pixels to process
        process_mask = (flat_mags >= magnitudeThreshold)
        if decodeMask is not None:
             process_mask &= (decodeMask.reshape(-1) > 0)

        t2 = time.perf_counter()
        if np.any(process_mask):
            pixels_to_decode = flat_traces[process_mask]
             
            # 1. GPU Path (Legacy support if needed, assumed CPU preferred for now due to complexity of re-integration)
            if useGpu and 'cupy' in globals():
                 # For brevity in cleanup, we rely on the Dot Product or NN path unless explicitly re-added.
                 # The previous complex block is removed for simplicity as requested.
                 pass

            # 2. Dot Product Path
            if distanceMetric in ('dot_product', 'softmax', 'softmax_dot_product'):
                softmax_mode = distanceMetric in ('softmax', 'softmax_dot_product')
                best_indices, best_dists, top1_probability = (
                    self._decode_pixels_by_similarity_torch(
                        pixels_to_decode,
                        chunk_size=int(decodeChunkSize),
                        softmax_temperature=(
                            softmaxTemperature if softmax_mode else None)
                    )
                )
                flat_di[process_mask] = best_indices
                flat_dist[process_mask] = best_dists
                if softmax_mode:
                    self.last_softmax_top1_probability = np.zeros(
                        N, dtype=np.float32)
                    self.last_softmax_top1_probability[process_mask] = (
                        top1_probability)

            # 3. Nearest Neighbors Path
            else:
                metric_arg = 'euclidean' if distanceMetric is None else distanceMetric
                jobs_arg = numThreads if numThreads is not None else -1
                
                nbrs = NearestNeighbors(n_neighbors=1, algorithm=nnAlgorithm, 
                                      metric=metric_arg, n_jobs=jobs_arg)
                nbrs.fit(self._decodingMatrix)
                
                dists, inds = nbrs.kneighbors(pixels_to_decode, return_distance=True)
                 
                flat_di[process_mask] = inds.flatten()
                flat_dist[process_mask] = dists.flatten()
        self.last_decode_timings['decode_core_seconds'] = (
            time.perf_counter() - t2)

        # Final filtering
        t3 = time.perf_counter()
        flat_di[flat_dist > distanceThreshold] = -1
        
        decodedImage = flat_di.reshape(rows, cols)
        distanceImage = flat_dist.reshape(rows, cols)
        self.last_decode_timings['threshold_reshape_seconds'] = (
            time.perf_counter() - t3)
        
        return decodedImage, pixelMagnitudes, normalizedPixelTraces, distanceImage
        
    def extract_barcodes_with_index(
            self, decodedImage: np.ndarray,
            pixelMagnitudes: np.ndarray, pixelTraces: np.ndarray,
            distances: np.ndarray, fov: int, cropWidth: int, zIndex: int = None,
            globalAligner=None, minimumArea: int = 1, outputLabels: bool = False,
            extractIntensityTraces: bool = False, crop_offset: int = 0
    ) -> pandas.DataFrame:
        """Extract the barcode information from the decoded image for barcodes
        that were decoded to the specified barcode index.

        Args:
            decodedImage: the image indicating the barcode index assigned to
                each pixel
            pixelMagnitudes: an image containing norm of the intensities for
                each pixel across all bits after scaling by the scale factors
            pixelTraces: an image stack containing the normalized pixel
                intensity traces
            distances: an image indicating the distance between the normalized
                pixel trace and the assigned barcode for each pixel
            fov: the index of the field of view
            cropWidth: the number of pixels around the edge of each image within
                which barcodes are excluded from the output list.
            zIndex: the index of the z position
            globalAligner: the aligner used for converted to local x,y
                coordinates to global x,y coordinates
            minimumArea: the minimum area of barcodes to identify. Barcodes
                less than the specified minimum area are ignored.
            outputLabels: output barcode labels and filtered labels
            
        Returns:
            a pandas dataframe containing all the barcodes decoded with the
                specified barcode index
            an image indicating the unique barcode index assigned to each pixel
            an image indicating the unique barcode index assigned to each pixel post minimum area filter
        """
        
        is3D = pixelTraces is not None and len(pixelTraces.shape) == 4
        if is3D:
            raise ValueError("3D barcode extraction not implimented")

        # Label connected decoded regions. We keep label ids stable so the
        # optional label image output remains compatible with downstream code.
        labels = measure.label(decodedImage + 1)
        labelAreas = np.bincount(labels.ravel())
        if len(labelAreas) <= 1:
            filteredLabels = labels
            validLabels = np.array([], dtype=np.int32)
            validAreas = np.array([], dtype=np.int32)
        else:
            keepMask = labelAreas >= minimumArea
            keepMask[0] = False
            filteredLabels = labels.copy()
            filteredLabels[~keepMask[labels]] = 0
            validLabels = np.flatnonzero(keepMask).astype(np.int32, copy=False)
            validAreas = labelAreas[validLabels].astype(np.int32, copy=False)

        flatLabels = filteredLabels.ravel()
        positiveMask = flatLabels > 0

        if np.any(positiveMask):
            labelIds = flatLabels[positiveMask].astype(np.int32, copy=False)
            flatIndexes = np.flatnonzero(positiveMask)
            imageWidth = decodedImage.shape[1]

            rowCoords = (flatIndexes // imageWidth).astype(np.float64, copy=False)
            colCoords = (flatIndexes % imageWidth).astype(np.float64, copy=False)
            validAreasFloat = validAreas.astype(np.float64, copy=False)

            magnitudeValues = pixelMagnitudes.ravel()[positiveMask].astype(
                np.float32, copy=False)
            distanceValues = distances.ravel()[positiveMask].astype(
                np.float32, copy=False)
            decodedValues = decodedImage.ravel()[positiveMask].astype(
                np.int32, copy=False)

            maxLabel = int(flatLabels.max())
            sumRows = np.bincount(
                labelIds, weights=rowCoords, minlength=maxLabel + 1)
            sumCols = np.bincount(
                labelIds, weights=colCoords, minlength=maxLabel + 1)
            sumMagnitudes = np.bincount(
                labelIds, weights=magnitudeValues, minlength=maxLabel + 1)
            sumDistances = np.bincount(
                labelIds, weights=distanceValues, minlength=maxLabel + 1)

            maxMagnitudes = np.full(maxLabel + 1, -np.inf, dtype=np.float32)
            np.maximum.at(maxMagnitudes, labelIds, magnitudeValues)
            minDistances = np.full(maxLabel + 1, np.inf, dtype=np.float32)
            np.minimum.at(minDistances, labelIds, distanceValues)

            uniqueLabels, firstIndexes = np.unique(labelIds, return_index=True)
            barcodeIds = np.zeros(maxLabel + 1, dtype=np.int32)
            barcodeIds[uniqueLabels] = decodedValues[firstIndexes]

            x = (sumCols[validLabels] / validAreasFloat).astype(np.float32)
            y = (sumRows[validLabels] / validAreasFloat).astype(np.float32)
            z = np.full(len(validLabels), zIndex, dtype=np.float32)

            data = {
                'unique_id': validLabels.astype(np.int32, copy=False),
                'barcode_id': barcodeIds[validLabels].astype(np.int32, copy=False),
                'area': validAreas.astype(np.int16, copy=False),
                'mean_intensity': (
                    sumMagnitudes[validLabels] / validAreasFloat).astype(np.float32),
                'max_intensity': maxMagnitudes[validLabels].astype(
                    np.float32, copy=False),
                'mean_distance': (
                    sumDistances[validLabels] / validAreasFloat).astype(np.float32),
                'min_distance': minDistances[validLabels].astype(
                    np.float32, copy=False),
                'x': x,
                'y': y,
                'z': z,
                'fov': np.full(len(validLabels), fov, dtype=np.int32),
                'cell_index': np.full(len(validLabels), -1, dtype=np.int32)
            }

            if globalAligner is not None:
                # crop_offset adds back the image-space crop so global coords
                # are in the full-FOV frame while local x,y stay cropped.
                centroids = np.stack([z, x + crop_offset, y + crop_offset], axis=1)
                globalCentroids = globalAligner.fov_coordinate_array_to_global(
                    fov, centroids)
                data['global_z'] = globalCentroids[:, 0]
                data['global_x'] = globalCentroids[:, 1]
                data['global_y'] = globalCentroids[:, 2]
            else:
                data['global_z'] = z
                data['global_x'] = x
                data['global_y'] = y

            if extractIntensityTraces:
                if pixelTraces is None:
                    raise ValueError(
                        'pixelTraces are required when '
                        'extractIntensityTraces=True')
                for i in range(pixelTraces.shape[0]):
                    traceValues = pixelTraces[i].ravel()[positiveMask].astype(
                        np.float32, copy=False)
                    traceSums = np.bincount(
                        labelIds, weights=traceValues, minlength=maxLabel + 1)
                    data[f'intensity_{i}'] = (
                        traceSums[validLabels] / validAreasFloat).astype(
                            np.float32)
        else:
            data = {
                'unique_id': np.array([], dtype=np.int32),
                'barcode_id': np.array([], dtype=np.int32),
                'area': np.array([], dtype=np.int16),
                'mean_intensity': np.array([], dtype=np.float32),
                'max_intensity': np.array([], dtype=np.float32),
                'mean_distance': np.array([], dtype=np.float32),
                'min_distance': np.array([], dtype=np.float32),
                'x': np.array([], dtype=np.float32),
                'y': np.array([], dtype=np.float32),
                'z': np.array([], dtype=np.float32),
                'fov': np.array([], dtype=np.int32),
                'cell_index': np.array([], dtype=np.int32),
                'global_z': np.array([], dtype=np.float32),
                'global_x': np.array([], dtype=np.float32),
                'global_y': np.array([], dtype=np.float32)
            }
            if extractIntensityTraces and pixelTraces is not None:
                for i in range(pixelTraces.shape[0]):
                    data[f'intensity_{i}'] = np.array([], dtype=np.float32)

        df = pandas.DataFrame(data)

        # Crop filter
        df = df[(df['x'].between(cropWidth, decodedImage.shape[0] - cropWidth)) &
                (df['y'].between(cropWidth, decodedImage.shape[1] - cropWidth))]
        
        if outputLabels:
            return df, filteredLabels
            
        return df

    def extract_refactors(
            self, decodedImage, pixelMagnitudes, normalizedPixelTraces,
            extractBackgrounds = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the scale factors that would result in the mean
        on bit intensity for each bit to be equal.

        This code follows the legacy matlab decoder.

        If the scale factors for this decoder are not set to 1, then the
        calculated scale factors are dependent on the input scale factors
        used for the decoding.

        Caution the refactorAreaThreshold is set in 
        Optimize parameters['area_threshold']

        Args:
            decodedImage
            pixelMagnitudes
            normalizedPixelTraces

        Returns:
             a tuple containing an array of the scale factors, an array
                of the backgrounds, and an array of the abundance of each
                barcode determined during the decoding. For the scale factors
                and the backgrounds, the i'th entry is the scale factor
                for bit i. If extractBackgrounds is false, the returned
                background array is all zeros.
        """

        # make labels of the decoded image
        # note the +1 to set the non decoded -1 pixels to zero for skimage
        labels = measure.label(decodedImage + 1)
        # remove small objects here to make regionprops quicker
        labels = morphology.remove_small_objects(labels, 
                                                min_size = self.refactorAreaThreshold)

        # get props of decoded regions to find the barcode id
        properties = measure.regionprops_table(labels,
                    intensity_image = decodedImage,
                    properties=('area', 'intensity_mean'), # intensity_mean is just a way to get barcode id
                    cache=False)

        # this dataframe stores the barcode identity
        df = pandas.DataFrame()
        df['barcode_id']  = (properties['intensity_mean']).astype(int)

        # count the barcodes seen
        barcodesSeen = np.zeros(self._barcodeCount)
        vc = df['barcode_id'].value_counts()
        barcodesSeen[vc.index] = vc.values

        # deal with the image intensity traces
        # annoying but to save time put the pixel mag and pixel traces together
        intensity_image = np.vstack([pixelMagnitudes.reshape(1,*pixelMagnitudes.shape),
                                     normalizedPixelTraces])
        intensity_image = np.moveaxis(intensity_image, 0,-1) # last axis needs to be channel axis

        # get the pixel values at each label region
        properties_traces = measure.regionprops_table(labels,
                    intensity_image = intensity_image,
                    properties=('area', 'image_intensity'), # for some reason need to call area first...
                    cache=False)

        # note again the first column is the pixel mag followed by pixel trace
        num_intensity_images = intensity_image.shape[-1] # num images is #bits + 1, first slice is the pixel mag followed by traces
        image_intensity_traces = [im.reshape(-1,num_intensity_images).T 
                                  for im in properties_traces['image_intensity']]
        # this returns #labels x numbits+1 x num_pixels in label
        # where a label is an identified spot

        # do background refactors here since it uses the same traces
        if extractBackgrounds:
            sumMinPixelTraces = np.zeros((self._barcodeCount, self._bitCount))
            # first item is pixelmag which is mult by pixeltrace
            # this finds the minimum value in each region in each bit image
            # it retuns a #labels x #bits array
            minPixelTrace = np.array([np.min(t[0] * t[1:], axis = 1) 
                                      for t in image_intensity_traces])
            # dataframe with barcode_id and min pixel trace
            df_min = pandas.concat([df,pandas.DataFrame(minPixelTrace)], 
                                   axis = 1)
            # for each barcode_id sum up the min pixel trace
            for bid, group in df_min.groupby('barcode_id'):
                num_bcs_in_group = len(group)

                if num_bcs_in_group > self.barcodesSeenThreshold: # ignore low abundance barcodes?
                    sumMinPixelTraces[bid] = group.iloc[:,1:].sum(axis = 0)

            offPixelTraces = sumMinPixelTraces.copy() # necessary?
            offPixelTraces[self._decodingMatrix > 0] = np.nan

            # if some barcodes are not seen don't include them
            # see also this step for the intensity refactoring...
            offPixelTraces[offPixelTraces == 0] = np.nan

            offBitIntensity = np.nansum(offPixelTraces, axis = 0) / np.sum(
                (self._decodingMatrix == 0) * barcodesSeen[:, None], axis = 0)
            backgroundRefactors = offBitIntensity
        else:
            backgroundRefactors = np.zeros(self._bitCount)

        # back to calculating refactors
        # this is mean of pixelmag * pixeltrace for every pixel
        meanPixelTrace = np.array([np.mean(t[0] * t[1:], axis = 1) 
                                for t in image_intensity_traces]) - backgroundRefactors
        normPixelTrace = meanPixelTrace/np.linalg.norm(meanPixelTrace, axis = 1)[:,None]

        # dataframe with the barcode id and norm pixel trace
        df_npt = pandas.concat([df, pandas.DataFrame(normPixelTrace)], axis = 1)

        sumPixelTraces = np.zeros((self._barcodeCount, self._bitCount))
        # for each barcode get the average pixel trace
        for bid, group in df_npt.groupby('barcode_id'):
            num_bcs_in_group = len(group)

            if num_bcs_in_group > self.barcodesSeenThreshold: # ignore low abundance barcodes?
                sumPixelTraces[bid] = group.iloc[:,1:].sum(axis = 0)/num_bcs_in_group

        sumPixelTraces[self._decodingMatrix == 0] = np.nan

        # add extra step here to deal with barcodes we don't see
        # relevant for large codebooks??
        sumPixelTraces[sumPixelTraces == 0] = np.nan

        onBitIntensity = np.nanmean(sumPixelTraces, axis = 0)
        refactors = onBitIntensity/np.mean(onBitIntensity)

        return refactors, backgroundRefactors, barcodesSeen

    def _calculate_normalized_barcodes(
            self, ignoreBlanks=False, includeErrors=False):
        """Normalize the barcodes present in the provided codebook so that
        their L2 norm is 1.

        Args:
            ignoreBlanks: Flag to set if the barcodes corresponding to blanks
                should be ignored. If True, barcodes corresponding to a name
                that contains 'Blank' are ignored.
            includeErrors: Flag to set if barcodes corresponding to single bit 
                errors should be added.
        Returns:
            A 2d numpy array where each row is a normalized barcode and each
                column is the corresponding normalized bit value.
        """
        
        barcodeSet = self._codebook.get_barcodes(ignoreBlanks=ignoreBlanks)
        
        if not includeErrors:
            weightedBarcodes = np.array(
                [normalize(x) for x in barcodeSet])
                
            return weightedBarcodes

        else:
            barcodesWithSingleErrors = []
            for b in barcodeSet:
                barcodeSet = np.array([b]
                                      + [binary.flip_bit(b, i)
                                         for i in range(len(b))])
                bcMagnitudes = np.sqrt(np.sum(barcodeSet*barcodeSet, axis=1))
                weightedBC = np.array(
                    [x/m for x, m in zip(barcodeSet, bcMagnitudes)])
                barcodesWithSingleErrors.append(weightedBC)
                
            return np.array(barcodesWithSingleErrors)
    
        
