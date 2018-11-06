import io
import os

import h5py
import numpy as np
from scipy.signal import detrend
from skimage import feature


def hyperslice(hyper, start, stop, rows=None, cols=None):
    """
    Sums a range of wavenumbers within a hyperspectral image. 
    
    Parameters 
    ----------
        hyper: class from HyperImage
        start: int
            start wavenumber
        stop: int
            stop wavenumber
        rows: tuple, (start, stop)
    	    starting and ending indices for rows within image 
            be displayed. If all rows are desired, can 
            leave blank.  
        cols: tuple, (start, stop)
            same as above, but for columns. 
    Returns
    -------
        slc: ndarray
            sum of intensities between the specified start and
            stop wavenumbers 
        
    """
    # show entire hyper image if no tuples are passed into arguments
    if rows == None:
        rows = (0, hyper.channel_data.shape[0])
    if cols == None:
        cols = (0, hyper.channel_data.shape[1])

    wavenumber = hyper.wavelength_data.tolist()
    wavenumberlist = [int(x) for x in wavenumber]
    # flip start and stop indices because of the
    # way the wavenumber data is stored.
    start_index = wavenumberlist.index(stop)
    stop_index = wavenumberlist.index(start)
    span = stop_index - start_index
    slc = hyper.hyper_image[rows[0]:rows[1], cols[0]:cols[1], start_index]
    for i in range(span):
        slc += hyper.hyper_image[rows[0]:rows[1], cols[0]:cols[1], start_index + i]

    return slc


def align_images(image, offset_image):
    """
    Flattens, aligns and crops two images to a common area. Retains original size by 
    padding cropped area with zeros. 
    
    Input: 
        image: any data channel
        offset_image: same data channel of a different scan. 
        
    Output: 
        ref_imagepadded, offset_image
    """
    # flatten images
    image = detrend(image, axis=1, type="linear")
    offset_image = detrend(offset_image, axis=1, type="linear")

    # find shift, error, and phase difference between the two images
    shift, error, diffphase = feature.register_translation(image, offset_image)

    # shift the offset image
    offset_imagecrop = offset_image[:-int(shift[0]), -int(shift[1]):]
    offset_imagepadded = np.zeros((256, 256))
    offset_imagepadded[:offset_imagecrop.shape[0], \
    :offset_imagecrop.shape[1]] = offset_imagecrop

    # swap the reference and offset image. take offset_image as the new ref.
    newref_image = offset_imagepadded
    offset_image = image

    # detect pixel shift again
    shift1, error1, diffphase1 = feature.register_translation(newref_image, offset_image)

    # shift original reference image to match offset image
    ref_imagecrop = offset_image[-int(shift1[0]):, :]
    ref_imagepadded = np.zeros((256, 256))
    ref_imagepadded[:ref_imagecrop.shape[0], :ref_imagecrop.shape[1]] = ref_imagecrop

    return ref_imagepadded, offset_imagepadded


def read_anfatec_params(path):
    """
    Reads in an ANFATEC parameter file. This file is produced by the Molecular
    Vista PiFM system and describes all parameters need to interpret the data 
    files produced when the data is saved.
    
    Input:
        path: a path to the ANFATEC parameter file.
        
    Output:
        file_descriptions: A list of dictionaries, with each item in the list 
            corresponding to a channel that was recorded by the PiFM.
        scan_params: A dictionary of non-channel specific scan parameters.
        
    """
    file_descriptions = []
    spectra_descriptions = []
    scan_params = {}
    parameters = {}
    inside_description = False

    with io.open(path, 'r', encoding="ISO-8859-1") as f:

        for i, row in enumerate(f):

            # Get rid of newline characters at the end of the line.
            row = row.strip()
            # check to make sure its  not empty
            if row:
                # First line of the file is useless. We tell the reader to stop at ';'
                if row[0] == unicode(';'):
                    continue

                # This string indicates that we have reached a channel description.
                if row.endswith('Begin') & row.startswith('File'):
                    inside_description = True
                    continue
                if row.endswith('SpectrumDescBegin'):
                    inside_description = True
                    continue
                if row.endswith('End') & row.startswith('File'):
                    file_descriptions.append(parameters)
                    parameters = {}
                    inside_description = False
                if row.endswith('SpectrumDescEnd'):
                    spectra_descriptions.append(parameters)
                    parameters = {}

                # split between :; creates list of two elements
                split_row = row.split(':')

                for i, el in enumerate(split_row):
                    split_row[i] = el.strip()

                # We want to save the channel parameters to a separate structure.
                if inside_description:
                    parameters[split_row[0]] = split_row[-1]
                else:
                    scan_params[split_row[0]] = split_row[-1]

    return scan_params, file_descriptions, spectra_descriptions


def load_hyper_numpy(folder_path):
    """
    Loads a hyper image that has previously been saved into a numpy format.
    INPUT: Folder containing each line of a hyperspectral image, which are 
    in .npy format. 
    
    OUTPUT: Three dimensional numpy array with the first two dimensions
    corresponding to x, y space and the third to the IR spectra obtained. 
    The third number indicates how many data points there are in 
    a spectrum, but does not contain information about the wavenumber. 
    **For our datasets only: the spectra were taken from 760 to 1875 cm-1
    with a 5 cm-1 spacing. 
    """
    files = os.listdir(folder_path)
    image_list = []

    for i, file_name in enumerate(files):
        path = os.path.join(folder_path, file_name)
        image_list.append(np.load(path))

    return np.column_stack(tuple(image_list))


def hyper_to_hdf5(path, dest_path):
    """
    Convert a series of MolecularVista files into hdf5 format.
    
    Input:
        path- The path to the Anfatec Parameter file that contains references to
        all necessary data.
        
        dest_path- desired path for the hdf5 file.
    Output:
        Saves a .hdf5 file to the dest_path path. 
    """
    name = os.path.basename(path)
    base, ext = os.path.splitext(name)
    new_name = base + '_HDF5' + '.hdf5'

    hyper_data = HyperImage(path)
    hyper_image = hyper_data.hyper_image

    channel_data = hyper_data.channel_data
    channel_names = hyper_data.channel_names

    h5f = h5py.File(os.path.join(dest_path, new_name))

    h5f.create_dataset('hyper_image', data=hyper_image)

    for i, channel_name in enumerate(channel_names):
        h5f.create_dataset(channel_name, data=channel_data[:, :, i])

    h5f.close()

    return


def hyper_from_hdf5(path):
    """
    Loads Hyperspectral data from a hdf5 file.
    """

    h5f = h5py.File(path, mode='r')

    hyper_image = h5f['hyper_image']
    channel_data = h5f['channel_data']

    h5f.close()

    return hyper_image, channel_data
