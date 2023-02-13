import codecs
import ntpath
from datetime import datetime
import json
import yaml
import glob  # find files
import pandas as pd  # file import
import numpy as np
import pickle
import os
import fnmatch
import h5py
import pandas as pd
from mat4py import loadmat
from mat4py.loadmat import ParseError
import logging


logger = logging.getLogger(__name__)


def importCVS(file, FPS=30, dwnSample=1, useSemiColSep=True, fpa_temp=20):
    """
    Main function of the program.

    Imports data from specified directory, demodulates the image and plots the
    results.

    Parameters
    ----------
    file : string
        path to the folder that contains the csv files
    FPS: frames per second
    dwnSample : int
        Donwsample factor. FPS ans number of images is divided by dwnSample
    useSemiColSep : bool
        Selects the separator caracter: True for semicolon and False for comma

    Returns
    ----------
    D : 3darray
        Video data in a matrix with the shape (Nx,Ny,Nima)
    Nx : int
        Number of images points in x direction
    Ny : int
        Number of image points in x direction
    Nima : int
        Number of images


    """
    # use slashes instead of backslashes
    file = file.replace("\\", "/")
    # wether to use a semicolon or a comma as separator for CSV file
    if useSemiColSep:
        sep = ';'
    else:
        sep = ','

    # find files
    all_files = glob.glob(file + "/*.csv")
    if (len(all_files)) == 0:
        raise FileNotFoundError("No files found in directory : " + file)

    print("Import " + str(len(all_files)) + " files from " + file)
    # import files
    li = []  # list of dataframes
    for i in range(0, len(all_files) - dwnSample + 1, dwnSample):
        impack = 0;
        for j in range(dwnSample):
            currentfile = glob.glob(file + "/*_" + str(i + j) + ".csv")[0]
            df = pd.read_csv(currentfile, sep=sep, header=None, skip_blank_lines=True,
                             usecols=range(336))
            impack = impack + np.array(df.astype(float))
        #img = TImage(impack, fpa_temp, i, 1e6 / FPS * dwnSample, i*1e6 / FPS * dwnSample, None)
        li.append(impack)

    #vid = TVideo(li)

    print(f"Imported video data with a resolution {len(li)} frames.")
    return li


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_dict_as_json(file: dict, filename: str = None):
    """
    Export dictionary with JSON.
    :param file: dictionary containing measure information
    :type file:
    :param path: path + filename
    :type path: string
    :return:
    :rtype:
    """
    if not isinstance(file, dict):
        return

    dico = json.dumps(file, cls=NumpyEncoder,
                      separators=('\t', ':'),
                      sort_keys=True,
                      indent=4)
    # create yaml object from dictionary
    # dico = yaml.dump(file, default_flow_style=False)
    # print(dico)
    # create json object from dictionary
    #dico = json.dumps(file)
    # open file for writing, "w"
    f = open(filename, "w")
    # write json object to file
    f.write(str(dico))
    # close file
    f.close()


def read_json_dcit(filename):
    with open(filename, 'r') as json_file:
        dico = json.loads(json_file)
    # Print the type of data variable
    print("Type:", type(dico))
    # close file
    json_file.close()
    return dico


def save_dict_as_text(file, filename: str = None):
    """
    Export dictionary as TXT.
    :param file: dictionary containing measure information
    :type file:
    :param path: path + filename
    :type path: string
    :return:
    :rtype:
    """
    # df = pd.DataFrame(data=mat.astype(np.float))
    # with open(filename, 'a') as f:
    #     dfAsString = df.to_csv('myfile.dat', sep=' ', float_format='%+.8e', header=False)
    #     f.write(dfAsString)
    # f.close()
    with open(filename, 'w+') as f:
        #f.write(str(file))
        for key, value in file.items():
            if isinstance(value, dict):
                f.write("%s\n" % key)
                for key1, value1 in value.items():
                    if isinstance(value1, np.ndarray):
                        for row in range(value1.shape[0]):
                            for column in range(value1.shape[1]):
                                f.write('\t%f' % (value1[row, column]))
                            f.write("\n")
                    else:
                        f.write('\t%s: %s\n' % (key1, value1))
            else:
                f.write('%s: %s\n' % (key, value))

    f.close()
    print('file saved.')
    #np.savetxt(r'test.txt', df.values, fmt='%d', delimiter='\t')


def save_image_as_ascii(artist = None, filename: str = None):
    """
    Save image with X, Y axes as ascii data.
    :param image: canvas image
    :type image: np.ndarray
    :param path: savepath
    :type path: str
    :return: .txt file
    :rtype:
    """
    # concatenate image, x_vector and y_vector
    img = artist.get_array().data
    data = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=float)
    data[0, 1:] = artist.x_vector
    data[1:, 0] = artist.y_vector
    data[1:, 1:] = img
    if filename:
        np.savetxt(filename, data, delimiter="\t", fmt="%.2f",
                   header="", comments="")


def read_hdf5_result(file):
    """
    Read Result file encoded as a .cls dictionary.
    :param file:
    :type file:
    :return:
    :rtype:
    """
    weights = {}

    keys = []
    with h5py.File("path.h5", 'r') as f:
        f.visit(keys.append)
        for key in keys:
            if ':' in key:
                print(f[key].name)
                weights[f[key].name] = f[key][()]

        # Print all root level object names (aka keys)
        # these can be group or dataset names
        print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        a_group_key = list(f.keys())[0]

        # get the object type for a_group_key: usually group or dataset
        print(type(f[a_group_key]))

        # If a_group_key is a group name,
        # this gets the object names in the group and returns as a list
        data = list(f[a_group_key])

        # If a_group_key is a dataset name,
        # this gets the dataset values and returns as a list
        data = list(f[a_group_key])
        # preferred methods to get dataset values:
        ds_obj = f[a_group_key]  # returns as a h5py dataset object
        ds_arr = f[a_group_key][()]  # returns as a numpy array
    return weights


def read_matfile4py(filename):
    """
    READ_MATFILE4PY only reads Matlab level 5 MAT-files using the mat4py
    library.
    """
    data = loadmat(filename)
    return data


def read_matfile(filename):
    """
    READ_MATFILE converts v.7 data.mat file into a dictionary. It uses the
    h5py library. The process can be time consuming if the hdf5 file is highly
    nested.
    """
    ret = {}
    with h5py.File(filename, 'r') as f:
        for key, value in f.items():
            if type(value).__name__ == 'Group':
                ret[key] = {}
                for key1, value1 in value.items():
                    if type(value1).__name__ == 'Dataset':
                        if value1.shape == (1,1):
                            ret[key][key1] = value1[:][0][0]
                        else:
                            ret[key][key1] = value1[:]
                    else:
                        ret[key][key1] = value1
            elif type(value).__name__ == 'Dataset':
                if value.shape == (1, 1):
                    ret[key] = value[:][0][0]
                else:
                    ret[key] = value[:]
            else:
                ret[key] = value
    f.close()
    return ret


if __name__ == '__main__':
    file = "C:/Users/sylvi/Documents/Stage_TB/ADEMOS/example data/WIC 336_17-25-47_20-08-2020_GNSS_ANY001"
    vid = importCVS(file)
    print("Saved pickle")
