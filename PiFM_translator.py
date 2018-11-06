import os
import pandas as pd
import numpy as np

class PiFMImage():
    """
    A class representing a PiFM image. Give the path to the PiFM data
    and receive a class that stores this information as a hyper image,
    and series of channel images.

    Input:
        path: Path to ANFATEC parameter file. This is the text file that
        is generated with each scan.

    Attributes:
        channel_names: list of all data channels in channel_data
        channel_data: a resolution x resolution x no. channel array.
                    channel_data[:,:,0] will return the matrix
                    corresponding to the channel in channel_names[0]
        parms: a dictionary of all scan parameters
        spectra_files: list of filenames containing point spectra
        point_spectra: list of df containing point spectra associated
                        with scan. point_spectra[0] will return the df
                        corresponding to the file in spectra_file[0]

    """

    def __init__(self, path):

        self.channel_names = []
        self.spectra_files = []
        full_path = os.path.realpath(path)
        directory = os.path.dirname(full_path)

        # Get the scan parameters and channel details.
        self.parms, channels, spectra = read_anfatec_params(full_path)

        x_pixel = int(self.parms['xPixel'])
        y_pixel = int(self.parms['yPixel'])

        # Make one big array for all the data channels.
        channel_data = np.zeros((x_pixel, y_pixel, len(channels)))
        for i, channel in enumerate(channels):
            self.channel_names.append(channel['Caption'])
            data = np.fromfile(os.path.join(directory, channel['FileName']), \
                               dtype='i4')
            # scaling = float(channel['Scale'])
            channel_data[:, :, i] = np.reshape(data, (256, 256))

            # for i,line in enumerate(np.split(data,y_pixel)):
            #    for j, pixel in enumerate(np.split(line,x_pixel)):
            #            channel_data[j,i,:] = (scaling*pixel)
        point_spectra = []
        for i in range(len(spectra)):
            self.spectra_files.append(spectra[i]['FileName'])
            data = pd.read_csv(os.path.join(directory, spectra[i]['FileName']), \
                               delimiter='\t')
            data.columns = ['wavenumber', 'intensity']
            point_spectra.append(data)

        # Here's how we access the different hyper and channel data.
        self.channel_data = channel_data
        self.point_spectra = point_spectra
