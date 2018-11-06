import os
import pandas as pd
import numpy as np

class HyperImage():
    """
    A class representing a Hyper image. Give the path to the Hyper data,
    and receive a class that stores this information as a hyper image,
    and series of channel images.
    """
    def __init__(self, path):

        self.wavelength_data = None
        self.channel_names = []
        full_path = os.path.realpath(path)
        directory = os.path.dirname(full_path)

        # Get the scan parameters and channel details.
        self.parms, channels, e =  read_anfatec_params(full_path)

        x_pixel = int(self.parms['xPixel'])
        y_pixel = int(self.parms['yPixel'])

        self.wavelength_data = np.loadtxt(os.path.join(directory ,str(channels[0]['FileNameWavelengths'])))
        wavenumber_length = self.wavelength_data.shape[0]
        image_shape = (x_pixel ,y_pixel ,wavenumber_length)

        hyper_image = np.zeros(image_shape)

        # This scales the integer data into floats.
        pifm_scaling = float(channels[0]['Scale'])

        # Read the Raw Hyper data from the bitfile.
        data = np.fromfile(os.path.join(directory ,channels[0]['FileName']) ,dtype='i4')
        for i ,line in enumerate(np.split(data ,y_pixel)):
            for j, pixel in enumerate(np.split(line ,x_pixel)):
                hyper_image[j ,i ,:] = pifm_scaling *pixel

        # Put all the different channels into one big array.
        channel_data = np.zeros((x_pixel, y_pixel, len(channels[1:])))
        for ch, channel in enumerate(channels[1:]):
            self.channel_names.append(channel['Caption'])
            data = np.fromfile(os.path.join(directory ,channel['FileName']) ,dtype='i4')
            scaling = float(channel['Scale'])

            for i ,line in enumerate(np.split(data ,y_pixel)):
                for j, pixel in enumerate(np.split(line ,x_pixel)):
                    channel_data[j ,i ,ch] = (scaling *pixel)

        # Here's how we access the different hyper and channel data.
        self.hyper_image = np.rot90(hyper_image, k=-1)
        self.channel_data = np.rot90(channel_data, k=-1)

        self.hyper_image = self.hyper_image[: ,::-1 ,:]
        self.channel_data = self.channel_data[: ,::-1 ,:]