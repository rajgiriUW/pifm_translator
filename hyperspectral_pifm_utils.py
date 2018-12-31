import sys
import os
import numpy as np
from pyUSID.io.translator import Translator
from pyUSID.io import write_utils

sys.path.append('./../../../../../hyperspectral_pifm')
path='./../../../../../hyperspectral_pifm/Film15_0010.txt'

class HyperspectralTranslator(Translator):
    """Writes images, spectrograms, point spectra and associated ancillary data sets to h5 file."""
    def __init__(self, path=None, *args, **kwargs):
        self.path = path
        super(HyperspectralTranslator, self).__init__(*args, **kwargs)

    def get_path(self, path=path):
        # get paths/get params dictionary, img/spectrogram/spectrum descriptions
        full_path = os.path.realpath(path)
        directory = os.path.dirname(full_path)
        # file name
        basename = os.path.basename(path)
        self.full_path = full_path
        self.directory = directory
        self.basename = basename

    #these dictionary parameters will be written to hdf5 file under measurement attributes
    ## make into function
    def read_anfatec_params(self):
        params_dictionary = {}
        params = True
        with open(self.path, 'r', encoding="ISO-8859-1") as f:
            for line in f:
                if params:
                    sline = [val.strip() for val in line.split(':')]
                    if len(sline) == 2 and sline[0][0] != ';':
                        params_dictionary[sline[0]] = sline[1]
                    #in ANFATEC parameter files, all attributes are written before file references.
                    if sline[0].startswith('FileDesc'):
                        params = False
        self.params_dictionary = params_dictionary

    def read_file_desc(self):
        img_desc = {}
        spectrogram_desc = {}
        spectrum_desc = {}
        with open(path,'r', encoding="ISO-8859-1") as f:
            ## can be made more concise...by incorporating conditons with loop control
            lines = f.readlines()
            for index, line in enumerate(lines):
                sline = [val.strip() for val in line.split(':')]
                #if true, then file describes image.
                if sline[0].startswith('FileDescBegin'):
                    no_descriptors = 5
                    file_desc = []
                    for i in range(no_descriptors):
                        line_desc = [val.strip() for val in lines[index+i+1].split(':')]
                        file_desc.append(line_desc[1])
                    #img_desc['filename'] = caption, scale, physical unit, offset
                    img_desc[file_desc[0]] = file_desc[1:]
                #if true, file describes spectrogram (ie hyperspectral image)
                if sline[0].startswith('FileDesc2Begin'):
                    no_descriptors = 10
                    file_desc = []
                    for i  in range(no_descriptors):
                        line_desc = [val.strip() for val in lines[index+i+1].split(':')]
                        file_desc.append(line_desc[1])
                    #caption, bytes perpixel, scale, physical unit, offset, offset, datatype, bytes per reading
                    #filename wavelengths, phys units wavelengths.
                    spectrogram_desc[file_desc[0]] = file_desc[1:]
                if sline[0].startswith('AFMSpectrumDescBegin'):
                    no_descriptors = 3
                    file_desc = []
                    for i in range(no_descriptors):
                        line_desc = [val.strip() for val in lines[index+i+1].split(':')]
                        file_desc.append(line_desc[1])
                    #file name, position x, position y
                    spectrum_desc[file_desc[0]] = file_desc[1:]
        self.img_desc = img_desc
        self.spectrogram_desc = spectrogram_desc
        self.spectrum_desc = spectrum_desc

    def read_spectrograms(self):
        spectrograms = {}
        spectrogram_spec_vals = {}
        for file_name, descriptors in self.spectrogram_desc.items():
            #load and save spectroscopic values
            spec_vals_i = np.loadtxt(os.path.join(self.directory, file_name.strip('.int') + 'Wavelengths.txt'))
            spectrogram_spec_vals[file_name] = spec_vals_i
            #load and save spectrograms
            spectrogram_i = np.fromfile(os.path.join(self.directory, file_name), dtype='i4')
            spectrograms[file_name] = np.zeros((x_len, y_len, len(spec_vals_i)))
            for y, line in enumerate(np.split(spectrogram_i, y_len)):
                for x, pt_spectrum in enumerate(np.split(line, x_len)):
                    spectrograms[file_name][x, y, :] = pt_spectrum * float(descriptors[2])
        self.spectrograms = spectrograms
        self.spectrogram_spec_vals = spectrogram_spec_vals

    def read_imgs(self):
        imgs = {}
        for file_name, descriptors in self.img_desc.items():
            img_i = np.fromfile(os.path.join(directory, file_name), dtype='i4')
            imgs[file_name] = np.zeros((x_len, y_len))
            for y, line in enumerate(np.split(img_i, y_len)):
                for x, pixel in enumerate(np.split(line, x_len)):
                    imgs[file_name][x, y] = pixel * float(descriptors[1])
        self.imgs = imgs

    def read_spectra(self):
        spectra = {}
        for file_name, descriptors in self.spectrum_desc.items():
            spectrum_i = np.fromfile(os.path.join(directory, file_name), dtype='i4')
            spectra[file_name] = spectrum_i
        self.spectra = spectra

    def make_pos_vals_inds(self):
        x_len = int(self.params_dictionary['xPixel'])
        y_len = int(self.params_dictionary['yPixel'])
        x_range = float(self.params_dictionary['XScanRange'])
        y_range = float(self.params_dictionary['YScanRange'])
        x_center = float(self.params_dictionary['xCenter'])
        y_center = float(self.params_dictionary['yCenter'])

        x_start = x_center-(x_range/2); x_end = x_center+(x_range/2)
        y_start = y_center-(y_range/2); y_end = y_center+(y_range/2)

        dx = x_range/x_len
        dy = y_range/y_len
        #assumes y scan direction:down; scan angle: 0 deg
        y_linspace = -np.arange(y_start, y_end, step=dy)
        x_linspace = np.arange(x_start, x_end, step=dx)
        pos_ind, pos_val = write_utils.build_ind_val_matrices(unit_values=(x_linspace, y_linspace), is_spectral=False)
        self.pos_ind, self.pos_val = pos_ind, pos_val

    def create_hdf5_file(self):
        h5_path = os.path.join(self.directory, self.basename.replace('.txt', '.h5'))
        h5_f = h5py.File(h5_path, mode='w')
        h5_meas_grp = usid.hdf_utils.create_indexed_group(h5_f, 'Measurement_')

    def write_spectrograms(self):
        if bool(self.spectrogram_desc):
            for key in self.spectrogram_desc:
                channel_i = usid.hdf_utils.create_indexed_group(h5_f['Measurement_000'], 'Channel_')

                h5_raw = usid.hdf_utils.write_main_dataset(channel_i,  # parent HDF5 group
                                                           (self.params_dictionary['xPixel'] *
                                                            self.params_dictionary['yPixel'], 559),  # shape of Main dataset
                                                           'Raw_Data',  # Name of main dataset
                                                           'Spectrogram',  # Physical quantity contained in Main dataset
                                                           'V',  # Units for the physical quantity
                                                           pos_dims,  # Position dimensions
                                                           spectrogram_spec_dims,  # Spectroscopic dimensions
                                                           dtype=np.float32,  # data type / precision
                                                           main_dset_attrs={'Caption': descriptors[0],
                                                                            'Bytes_Per_Pixel': descriptors[1],
                                                                            'Scale': descriptors[2],
                                                                            'Physical_Units': descriptors[3],
                                                                            'Offset': descriptors[4],
                                                                            'Datatype': descriptors[5],
                                                                            'Bytes_Per_Reading': descriptors[7],
                                                                            'Wavelengths': descriptors[7],
                                                                            'Wavelength_Units': descriptors[8]})
                h5_raw.h5_pos_vals[:, :] = pos_vals
                h5_raw[:, :] = spectrograms['Film15_0010hyPIRFwd.int'].reshape(h5_raw.shape)

    def write_images(self):
        if bool(self.img_desc):


    def write_spectra(self):
        if bool(self.spectrum_desc):

    def translate(self):
        self.get_path()
        self.read_anfatec_params()
        self.read_file_desc()
        self.read_spectrograms()
        self.read_imgs()
        self.read_spectra()
        self.make_pos_vals_inds()
        self.create_hdf5_file()
        self.write_spectrograms()
        self.write_images()
        self.write_spectra()
