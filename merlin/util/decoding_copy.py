import numpy as np
import pandas
import cv2
from typing import Tuple
from typing import Dict
from skimage import measure
from skimage import morphology
from sklearn.neighbors import NearestNeighbors
import gc

from merlin.util import binary
from merlin.data import codebook as mcodebook

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

        # test for if we want to disregard low abundant barcodes in refactors
        # value of zero will not do anything
        self.barcodesSeenThreshold = 0 


    def decode_pixels(self, imageData: np.ndarray,
                      scaleFactors: np.ndarray=None,
                      backgrounds: np.ndarray=None,
                      distanceThreshold: float=0.5176,
                      magnitudeThreshold: float=1,
                      lowPassSigma: float=1,
                      decodeMask = None,
                      use_gpu = False):
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
        if scaleFactors is None:
            scaleFactors = self._scaleFactors
        if backgrounds is None:
            backgrounds = self._backgrounds

        # the dimensions are num_bits x image_rows x image_cols
        filteredImages = np.zeros(imageData.shape, dtype= np.float32)
        filterSize = int(2 * np.ceil(2 * lowPassSigma) + 1)
        for i in range(imageData.shape[0]):
            filteredImages[i, :, :] = cv2.GaussianBlur(
                imageData[i, :, :], (filterSize, filterSize), lowPassSigma)

        # going to try to make this more numpy-ish
        # and add decode mask parameter, ie decode only in segmented regions...

        image_shape = filteredImages.shape[1:] # dimensions of one image plane

        scaledPixelTraces = ((filteredImages-backgrounds[:,None,None])/scaleFactors[:,None,None]).astype(np.float32)
        pixelMagnitudes = np.linalg.norm(scaledPixelTraces, axis = 0).astype(np.float32)
        pixelMagnitudes[pixelMagnitudes == 0] = 1.0
        normalizedPixelTraces = scaledPixelTraces/pixelMagnitudes # this should be a float32 to save a little mem

        neighbors = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        neighbors.fit(self._decodingMatrix) # fit takes n_samples x n_features

        if decodeMask is None: # decode the full image

            if use_gpu == False:
                # sklearn kneighbors wants n_queries x n_features
                distances, indexes = neighbors.kneighbors(
                        normalizedPixelTraces.reshape(normalizedPixelTraces.shape[0], -1).T,
                        return_distance=True)
                
            else: # gpu decode here

                # hard coding some numbers here be careful
                step = 128*128 # 256*256 # how to best determine this number here!?
                # this seems reasonable without too much additional overhead
                start = 0
                stop = np.prod(normalizedPixelTraces.shape[1:])
                normalizedPixelTraces_flat = normalizedPixelTraces.reshape(normalizedPixelTraces.shape[0], -1)

                #distanceImage = np.ones(np.prod(image_shape), dtype = np.float32)
                #decodedImage = np.zeros(np.prod(image_shape), dtype = np.int32)

                # lazy list right now make it numpy
                distances = []
                indexes = []
                for idx in range(start, stop, step):
                    pixels_to_decode = normalizedPixelTraces_flat[:,idx:idx+step]
                    # cdist wants mA x n and mB x n arrays where n is the number of dimensions ie bits      
                    ds, inds = calculate_distances_gpu(pixels_to_decode.T, # .T here or in function?
                                                    self._decodingMatrix)
                    distances.append(ds)
                    indexes.append(inds)

                distances = np.array(distances).flatten()
                indexes = np.array(indexes).flatten()

            # remove index that are greater than distance threshold
            indexes[distances > distanceThreshold] = -1

            # turn the filtered indexes back into the decoded image and do the magnitude filter
            decodedImage = indexes.reshape(image_shape).astype(np.int32)
            decodedImage[pixelMagnitudes < magnitudeThreshold] = -1

            # reshape the distance image
            distanceImage = distances.reshape(image_shape).astype(np.float32)

        else: # decode using a mask

            mask = decodeMask > 0 # binarize it just in case

            # only take pixels that are in the mask
            normalizedPixelTracesToDecode = np.array(
                [frame[mask] for frame in normalizedPixelTraces]).T

            if use_gpu == False:
                distances, indexes = neighbors.kneighbors(
                        normalizedPixelTracesToDecode,
                        return_distance=True)
            else: # gpu decode here
                raise Exception('Not implemented yet') 
 
            # remove index that are greater than distance threshold
            indexes[distances > distanceThreshold] = -1

            decodedImage = np.full(image_shape, -1, dtype = np.int32)
            decodedImage[mask] = indexes.flatten()
            decodedImage[pixelMagnitudes < magnitudeThreshold] = -1

            # reshape the distance image
            distanceImage = np.full(image_shape, -1, dtype = np.float32)
            distanceImage[mask] = distances.flatten()

        return decodedImage, pixelMagnitudes, normalizedPixelTraces, distanceImage

    # try to speed up this bottleneck with morphology and regionprops_table
    # no longer need to iterate through all the barcode IDs
    # should be done in one shot...

    def extract_barcodes_with_index(
            self, decodedImage: np.ndarray,
            pixelMagnitudes: np.ndarray, pixelTraces: np.ndarray,
            distances: np.ndarray, fov: int, cropWidth: int, zIndex: int = None,
            globalAligner=None, minimumArea: int = 1
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
        Returns:
            a pandas dataframe containing all the barcodes decoded with the
                specified barcode index
        """
        
        is3D = len(pixelTraces.shape) == 4
        if is3D:
            raise ValueError("3D barcode extraction not implimented")

        # make labels of the decoded image, note the +1 to set the non decoded -1 pixels to zero
        labels = measure.label(decodedImage + 1)
        # remove small objects here to make region props faster
        labels = morphology.remove_small_objects(labels, min_size = minimumArea)

        # this will get plugged into regionprops_table
        intensity_image = np.stack([decodedImage, pixelMagnitudes, distances], axis = -1)
        # prop-0 is decoded, prop-1 is mag, prop-2 is dist

        properties = measure.regionprops_table(labels,
            intensity_image = intensity_image,
            properties = ('area',
                        'centroid', 
                        'intensity_mean', 
                        'intensity_max', 
                        'intensity_min'),
            cache = False)
        

        # make an empty dataframe
        columnNames = ['barcode_id', 'fov', 'mean_intensity', 'max_intensity',
                       'area', 'mean_distance', 'min_distance', 'x', 'y', 'z',
                       'global_x', 'global_y', 'global_z', 'cell_index']

        df = pandas.DataFrame(columns=columnNames)

        barcode_id = properties['intensity_mean-0'] # gets the barcode id from decoded image
        df['barcode_id'] = barcode_id.astype(np.int32)
        df['area'] = properties['area'].astype(np.int16)

        # get magnitude properties
        df['mean_intensity'] = properties['intensity_mean-1'].astype(np.float32)
        df['max_intensity'] = properties['intensity_max-1'].astype(np.float32)

        # get distance properties
        df['mean_distance'] = properties['intensity_mean-2'].astype(np.float32)
        df['min_distance'] = properties['intensity_min-2'].astype(np.float32)

        # get centroid properties
        # not going to bother with weighted centroid right now! adds too much time for little gain
        centroids = np.zeros([len(barcode_id),3], dtype = np.float32) # order it z x y for merlin
        centroids[:,0] = zIndex
        centroids[:,1] = properties['centroid-1'] # centroid-1 is col = xcoord
        centroids[:,2] = properties['centroid-0'] # centroid-0 is row = ycoord

        df['x'] = centroids[:,1]
        df['y'] = centroids[:,2]
        df['z'] = centroids[:,0]
        df['fov'] = np.int32(fov)
        df['cell_index'] = -1

        if globalAligner is not None:
            globalCentroids = globalAligner.fov_coordinate_array_to_global(
                fov, centroids)
        else:
            globalCentroids = centroids

        df['global_z'] = globalCentroids[:,0]
        df['global_x'] = globalCentroids[:,1]
        df['global_y'] = globalCentroids[:,2]
        
        # now do the pixel traces
        npt = np.moveaxis(pixelTraces,0,-1) # move the channel axis to the last...

        properties_npt = measure.regionprops_table(labels,
                    intensity_image = npt,
                    properties = ('area', 'intensity_mean'), # for some reason you cannot call intensity_mean alone
                    cache = False)

        # make a dataframe of the pixel intensities
        columnNames_intensity = [f'intensity_{i}' for i in range(npt.shape[-1])]
        df_npt = pandas.DataFrame(np.array(list(properties_npt.values()))[1:].T,
                     columns = columnNames_intensity)

        # better this way if there are many bits...
        df = pandas.concat([df, df_npt], axis = 1)

        # finally filter with cropwidth
        df = df[(df['x'].between(cropWidth, decodedImage.shape[0] - cropWidth)) &
                (df['y'].between(cropWidth, decodedImage.shape[1] - cropWidth))]

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
        magnitudes = np.sqrt(np.sum(barcodeSet*barcodeSet, axis=1))
       
        if not includeErrors:
            weightedBarcodes = np.array(
                [normalize(x) for x, m in zip(barcodeSet, magnitudes)])
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