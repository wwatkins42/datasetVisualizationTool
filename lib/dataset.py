import numpy as np
import csv
import re

types = {'missing':0, 'numerical':1, 'string':2, 'date':3, 'bool':4}
pattern = {
    'numerical':re.compile(r'[-+]?\d*\.{0,1}\d*\Z'),
    'date':re.compile(r"\d+[-]\d+[-]\d+")
}
missing_labels = ['','nan','NaN','n/a','N/A'] # TODO: should get that from args, with this as default value

def loadCSV(filepath, has_header=True):
    with open(filepath, 'rbU') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = []
        for i, row in  enumerate(reader):
            if i == 0 and has_header: labels = row; continue
            data.append(row)
    data = np.asarray(data)
    if has_header is False:
        labels = ["feature-%d"%i for i in range(data.shape[1])] # TODO: find a way to check if header is missing automatically
    return data, np.asarray(labels)

def hasMissingValues(data):
    return np.count_nonzero(np.isin(data, missing_labels)) > 0

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

def determineValuesType(data):
    tmp = np.empty_like(data, dtype=int)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i,j] in missing_labels:
                tmp[i,j] = types['missing']
            elif pattern['date'].match(data[i,j]):
                tmp[i,j] = types['date']
            elif pattern['numerical'].match(data[i,j]):
                tmp[i,j] = types['numerical']
            elif data[i,j].lower() in ['true','false']:
                tmp[i,j] = types['bool']
            else:
                tmp[i,j] = types['string']
    return tmp

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
            elif pattern['date'].match(data[i,j]):
                features_type.append(types['date'])
                break
            elif pattern['numerical'].match(data[i,j]):
                features_type.append(types['numerical'])
                break
            elif data[i,j].lower() in ['true','false']:
                features_type.append(types['bool'])
                break
            else:
                features_type.append(types['string'])
                break
    return features_type
