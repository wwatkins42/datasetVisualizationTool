import numpy as np
import csv
import utils

types = {'missing':0, 'numerical':1, 'string':2, 'date':3, 'bool':4}
labels= {0:'missing', 1:'numerical', 2:'string', 3:'date', 4:'bool'}
missing_labels = ['','nan','NaN','n/a','N/A','NA']

def loadCSV(filepath, has_header=True):
    with open(filepath, 'rbU') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        has_header = hasHeader(csvfile)
        data = []
        for i, row in  enumerate(reader):
            if i == 0 and has_header: labels = row; continue
            data.append(row)
    data = np.asarray(data)
    if has_header is False:
        labels = ["feature-%d"%i for i in range(data.shape[1])]
    return data, np.asarray(labels)

def hasHeader(csvfile):
    ''' check if csvfile has an header with sniffer. In case it does not
        know, return True
    '''
    try:
        has_header = csv.Sniffer().has_header(csvfile.read(2048))
    except:
        has_header = True
    csvfile.seek(0)
    return has_header

def hasMissingValues(data):
    return np.count_nonzero(np.isin(data, missing_labels)) > 0

def centeredNormalization(data):
    # mean centering + std normalization
    data -= np.mean(data)
    data /= np.std(data)
    return data

def dropMissingData(data, return_indices=False):
    ''' remove the lines containing a missing value from the dataset
        takes a 1 or 2 dimensional array
    '''
    if len(data.shape) == 1:
        new = data[np.isin(data, missing_labels, invert=True)]
        return (new, np.argwhere(np.isin(data, missing_labels))) if return_indices else new
    incomplete = []
    for i in range(len(data)):
        if np.count_nonzero(np.isin(data[i], missing_labels)) > 0:
            incomplete.append(i)
    new = np.delete(data, incomplete, axis=0)
    return (new, incomplete) if return_indices else new

def determineValuesType(data, return_keys=False):
    if return_keys is True:
        mlen = np.max([len(key) for key in types.keys()])
        keys = np.empty_like(data, dtype='|S%d'%mlen)
    valuesType = np.empty_like(data, dtype=int)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i,j] in missing_labels:
                valuesType[i,j] = types['missing']
                if return_keys: keys[i,j] = 'Missing'
            elif utils.isFloat(data[i,j]):
                valuesType[i,j] = types['numerical']
                if return_keys: keys[i,j] = 'Numerical'
            elif data[i,j].lower() in ['true','false']:
                valuesType[i,j] = types['bool']
                if return_keys: keys[i,j] = 'Boolean'
            elif utils.isDate(data[i,j]):
                valuesType[i,j] = types['date']
                if return_keys: keys[i,j] = 'Date'
            else:
                valuesType[i,j] = types['string']
                if return_keys: keys[i,j] = 'String'
    return (valuesType, keys) if return_keys else valuesType

def determineFeaturesType(data):
    ''' return the type of the first value (except missing) for each feature
    '''
    features_type = []
    for j in range(data.shape[1]):
        for i in range(data.shape[0]):
            if data[i,j] in missing_labels:
                if i == data.shape[0] - 1:
                    features_type.append(types['missing'])
                    break
                continue
            elif utils.isFloat(data[i,j]):
                features_type.append(types['numerical'])
                break
            elif data[i,j].lower() in ['true','false']:
                features_type.append(types['bool'])
                break
            elif utils.isDate(data[i,j]):
                features_type.append(types['date'])
                break
            else:
                features_type.append(types['string'])
                break
    return features_type
