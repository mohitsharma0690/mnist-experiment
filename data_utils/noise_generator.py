import numpy as np
import argparse
import csv
import h5py
import os
import pprint
import time
import types

from sklearn.metrics import confusion_matrix


def recursively_save_dict_contents_to_group(h5file, path, dic):
  """
  Take an already open HDF5 file and insert the contents of a dictionary
  at the current path location. Can call itself recursively to fill
  out HDF5 files with the contents of a dictionary.
  """
  assert type(dic) is types.DictionaryType, "must provide a dictionary"
  assert type(path) is types.StringType, "path must be a string"
  assert type(h5file) is h5py._hl.files.File, "must be an open h5py file"
  for key in dic:
    assert type(key) is types.StringType, \
    'dict keys must be strings to save to hdf5'
    if type(dic[key]) in (np.int64, np.float64, types.StringType):
      h5file[path + key] = dic[key]
      assert h5file[path + key].value == dic[key], \
        'The data representation in the HDF5 file does not match the ' \
            'original dict.'
    if type(dic[key]) is np.ndarray:
      h5file[path + key] = dic[key]
      assert np.array_equal(h5file[path + key].value, dic[key]), \
          'The data representation in the HDF5 file does not match the ' \
              'original dict.'
    elif type(dic[key]) is types.DictionaryType:
      recursively_save_dict_contents_to_group(
          h5file, path + key + '/', dic[key])

def create_noise_map(noise_csv, num_labels=10):
  ''' Returns a dictionary with each key(label) mapped to a list of labels(keys)
  with the probability that the source label (key) can be confused as the target
  label(value).
  '''
  noise_map = {}
  with open(noise_csv) as noise_csv_f:
    noise_csv_reader = csv.DictReader(noise_csv_f)
    for i in xrange(num_labels):
      noise_map[i] = []

    # Create the noise map
    for row in noise_csv_reader:
      if int(row['new_label']) > 0:
        old_label, new_label = int(row['org_label']), int(row['new_label'])
        prob = float(row['prob'])
        noise_map[old_label].append((new_label, prob))

  return noise_map

def get_random_noise_prob(noise_csv, num_labels=10):
  noise = [0]*num_labels
  with open(noise_csv) as noise_csv_f:
    noise_csv_reader = csv.DictReader(noise_csv_f)
    for row in noise_csv_reader:
      if int(row['new_label']) < 0:
        noise[int(row['org_label'])] = float(row['prob'])

  return noise

def get_confusion_matrix(y, noisy_y, num_labels=10):
  ''' Get the confusion matrix for y and noisy_y labels.
  '''
  labels = [i for i in xrange(num_labels)]
  conf = confusion_matrix(y, noisy_y, labels)
  return conf

def add_noise(y, noise_csv, num_labels=10):
  noise_map = create_noise_map(noise_csv, num_labels=num_labels)
  noisy_y = np.array(y, dtype=int)
  # Add csv defined confusion noise
  for i in xrange(y.shape[0]):
    org, new_label = y[i][0], -1
    p, total_p = np.random.rand(), 0
    for noise_label, noise_prob in noise_map[org]:
      total_p += p
      if total_p < noise_prob:
        new_label = noise_label
        break
    new_label = org if new_label == -1 else new_label    
    noisy_y[i] = new_label

  # Add random noise 
  random_noise_prob = get_random_noise_prob(noise_csv, num_labels=num_labels)
  for i in xrange(y.shape[0]):
    # Only add random noise to clean samples i.e. for samples which we haven't
    # already added noise to
    if y[i, 0] == noisy_y[i, 0] and \
        np.random.rand() < random_noise_prob[y[i, 0]]:
      noisy_y[i, 0] = np.random.randint(num_labels)

  assert(y.shape == noisy_y.shape)
  return y, noisy_y

def main(data_h5, new_data_h5, noise_csv):
  data_h5f = h5py.File(data_h5, 'r')  
  X = np.array(data_h5f['data'])
  y = np.array(data_h5f['label'], dtype=int)
  data_h5f.close()

  org_y, noisy_y = add_noise(y, noise_csv, num_labels=10)
  new_h5 = os.path.join(os.path.dirname(data_h5), new_data_h5)
  new_h5_dict = {'data': X, 'label': org_y, 'noisy_label': noisy_y}
  new_h5_f = h5py.File(new_h5, 'w')
  recursively_save_dict_contents_to_group(new_h5_f, '/', new_h5_dict)
  new_h5_f.flush()
  new_h5_f.close()
  conf = get_confusion_matrix(org_y, noisy_y, num_labels=10)
  print("Confusion matrix for noisy labels:")
  print(conf)
  print("Did write file to {}".format(new_h5))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Add noise to mnist labels.')
  parser.add_argument('--data_h5', nargs='?', type=str, const=1,
      required=True, default='', help='h5 file containing the original labels')
  parser.add_argument('--new_data_h5', nargs='?', type=str, const=1,
      required=True, default='', help='h5 file containing the new noisy labels')
  parser.add_argument('--noise_csv', nargs='?', type=str, const=1,
      required=True, default='', help='csv containing the noise parameters.')
  args = parser.parse_args()
  pp = pprint.PrettyPrinter(indent=4)
  pp.pprint(args)
  
  np.random.seed(int(time.time()))

  main(args.data_h5, args.new_data_h5, args.noise_csv)

