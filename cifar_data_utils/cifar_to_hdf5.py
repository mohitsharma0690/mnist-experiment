import numpy as np
import h5py
import csv
import pdb
import os
import argparse
import pprint
import types

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

def debug_visualize(d):
  from PIL import Image
  d = np.array(d, dtype=np.int8)
  d = np.swapaxes(d, 0, 1)
  d = np.swapaxes(d, 1, 2)
  img = Image.fromarray(d, 'RGB')
  img.show()
  pdb.set_trace()

def unpickle(f):
    import cPickle
    with open(f, 'rb') as fo:
        d = cPickle.load(fo)
    return d

def convert_data_to_3d(d):
  '''
  d: Nx3072 where 1024 [R,G,B] -- 32*32 [R,G,B] (row major) 
  '''
  d_r, d_g, d_b = np.array(d[:,:1024]), np.array(d[:, 1024:2048]), \
      np.array(d[:,2048:])
  N = d_r.shape[0]
  data = np.zeros((N, 3, 32, 32))
  data[:,0,:,:] = np.reshape(d_r, (N, 32, 32))
  data[:,1,:,:] = np.reshape(d_g, (N, 32, 32))
  data[:,2,:,:] = np.reshape(d_b, (N, 32, 32))
  
  # Visualize random elements to see that we have proper images
  # debug_visualize(data[0])

  return data


def main(dir_pkl, dir_h5):
  '''
  dir_pkl: read data_batch_[1/5], test_batch
  dir_h5: write train_data.h5, test_data.h5
  '''
  train_data = [f for f in os.listdir(dir_pkl) if f.startswith('data_batch')]
  test_data = [f for f in os.listdir(dir_pkl) if f.startswith('test_batch')]
  train_data_h5 = {'data': [], 'label': []}
  for f in train_data:
    f_data = unpickle(os.path.join(dir_pkl, f))
    train_data_h5['data'].append(convert_data_to_3d(f_data['data']))
    train_data_h5['label'].append(np.array(f_data['labels'], dtype=np.int32))

  train_h5_path = os.path.join(dir_h5, 'train_data.h5')
  train_h5 = h5py.File(train_h5_path, 'w')
  train_data_h5['data'] = np.vstack(train_data_h5['data'])
  train_data_h5['label'] = np.hstack(train_data_h5['label'])
  recursively_save_dict_contents_to_group(train_h5, '/', train_data_h5)
  train_h5.flush()
  train_h5.close()
  print('Did write {}'.format(train_h5_path))

  # There is only test pkl file
  assert(len(test_data) == 1)
  test_data_h5 = {}
  test_data = unpickle(os.path.join(dir_pkl, test_data[0]))
  test_data['data'] = convert_data_to_3d(test_data['data'])
  test_data['label'] = np.array(test_data['labels'], dtype=np.int32)
  test_h5_path = os.path.join(dir_h5, 'test_data.h5')
  test_h5  = h5py.File(test_h5_path, 'w')
  recursively_save_dict_contents_to_group(test_h5, '/', test_data)
  test_h5.flush()
  test_h5.close()
  print('Did write {}'.format(test_h5_path))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Create hdf5 files for cifar 10 pickle files.')
  parser.add_argument('--data_pkl', nargs='?', type=str, required=True,
      help='Directory containing cifar10 pkl files.')
  parser.add_argument('--data_h5', type=str, nargs='?', required=True,
      help='Directory where the h5 file should be saved.')

  args = parser.parse_args()
  pp = pprint.PrettyPrinter(indent=4)
  pp.pprint(args)

  main(args.data_pkl, args.data_h5)

