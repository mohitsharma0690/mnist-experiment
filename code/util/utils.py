import csv
import h5py
import itertools
import json
import math
import os
import pdb
import sys
import time
import types

import numpy as np
import cPickle as cp
import pandas as pd
import scipy.io as sio
from copy import deepcopy
from scipy.interpolate import splev, splrep
from scipy.interpolate.rbf import Rbf

NUM_FEATURES = 53

GESTURE_NAMES = ["None", "Nod", "Jerk", "Up", "Down", "Ticks", "Tilt", "Shake",
    "Turn", "Forward", "Backward"]

# NOTE: OpenFace landmark output format
# https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format

# Assuming starting index for facial landmark is 1
OPENFACE_LANDMARKS_TO_USE = [
    28, 28 + 68, # forehead
    34, 34 + 68, # nosetip
    2,   2 + 68, # left side of face
    4,   4 + 68,
    8,   8 + 68, # bottom (right)
    10, 10 + 68,
    14, 14 + 68, # top
    16, 16 + 68
]

for i in range(len(OPENFACE_LANDMARKS_TO_USE)):
  OPENFACE_LANDMARKS_TO_USE[i] = OPENFACE_LANDMARKS_TO_USE[i] + 11

CPM_INDEX_TO_JOINT_TYPE = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
    # 18: "Bkg",  # CPM output doesn't seem to emit this value
]

def get_cpm_joint_type_to_idx(joint):
  for idx, joint_name in enumerate(CPM_INDEX_TO_JOINT_TYPE):
    if joint_name == joint:
      return idx
  assert(False)
  return -1

def get_cpm_joint_type_to_h5_idx(joint):
  return 3*get_cpm_joint_type_to_idx(joint)


def get_gest_index_for_name(name):
  return GESTURE_NAMES.index(name)

def recursively_get_dict_from_group(group_or_data):
  d = {}
  if type(group_or_data) == h5py.Dataset:
    return np.array(group_or_data)

  # Else it's still a group
  for k in group_or_data.keys():
    v = recursively_get_dict_from_group(group_or_data[k])
    d[k] = v
  return d

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

def __get_all_features(openface_h5_path, cpm_h5_path, zface_h5_path):
  openface_h5 = h5py.File(openface_h5_path, 'r')
  X = np.array(openface_h5['features'])
  y = np.array(openface_h5['annotations']).astype(int)
  openface_h5.close()
  cpm_h5 = h5py.File(cpm_h5_path, 'r')
  cpm_X = np.array(cpm_h5['joints'])
  cpm_h5.close()
  zface_X = None
  if zface_h5_path is not None:
    zface_h5 = h5py.File(zface_h5_path)
    zface_X = np.array(zface_h5['features'])
    zface_h5.close()
  return X, y, cpm_X, zface_X

def load_all_face_body_features(openface_dir, cpm_dir, zface_dir):
  '''
  Returns three dicts each for openface, cpm, zface features. This code
  works under the assumption that all the files in each dir has the same
  name.
  '''
  file_to_openface_feat, file_to_cpm_feat, file_to_zface_feat = {}, {}, {}
  file_to_labels = {}
  for f in os.listdir(openface_dir):
    if "mohit" in f: continue
    if "static.mp4" not in f: continue
    assert(os.path.exists(os.path.join(cpm_dir, f)))
    assert(os.path.exists(os.path.join(zface_dir, f)))
    openface_feat, labels, cpm_feat, zface_feat = __get_all_features(
        os.path.join(openface_dir, f),
        os.path.join(cpm_dir, f),
        os.path.join(zface_dir, f),
    )
    file_to_openface_feat[f] = openface_feat
    file_to_cpm_feat[f] = cpm_feat
    file_to_zface_feat[f] = zface_feat
    file_to_labels[f] = labels

  return file_to_openface_feat, file_to_cpm_feat, file_to_zface_feat, \
    file_to_labels


# NOTE: This should be change each time we change the OpenFace features we are
# going to use.
def normalize_openface_feats(X, mean=None, std=None, landmarks=None):
  assert(mean is not None and std is not None and landmarks is not None)
  '''
  for i in range(3):
    assert(std[i+6] >= 0.0001)
    X[:,i] = (X[:,i] - mean[i+6]) / std[i+6]
  '''

  # Normalize landmarks
  for i in range(12, 29):
    assert(std[landmarks[i-12]] >= 0.0001)
    X[:, i-12] = (X[:, i-12] - mean[landmarks[i-12]]) / std[landmarks[i-12]]

  return X

def __normalize_data_each_sensor_signal(X, y, signals=None):
  '''
  Normalize data X and y.
  '''
  if signals is None:
    z_normalize = False
    if not z_normalize:
      X_min = np.min(X, axis=0)
      X_max = np.max(X, axis=0)
      return (X - X_min) / (X_max - X_min), y
    else:
      X_mean = np.mean(X, axis=0)
      X_std = np.std(X, axis=0)
      # norm_X = zero_mean_X / var_X
      norm_X = (X - X_mean) / X_std
      return norm_X, y
  else:
    for i in signals:
      max_signal = float(max(X[:, i]))
      # Some frames are not correctly calculated by openface since the
      # video has no face. These are the initial frames hence we don't want
      # to use such frames
      j = 100
      while abs(X[j, i]) < 0.0001:
        j = j + 1
      X_mean = np.mean(X[j:, i])
      X_std = np.std(X[j:, i])
      X[j:, i] = (X[j:, i] - X_mean) / X_std
      '''
      min_signal = float(min(X[j:, i]))
      X[j:, i] = (X[j:, i] - min_signal) / (max_signal - min_signal)
      '''
    return X, y

def get_face_difference(a, b):
  '''
  Returns difference with respect to time between two face vectors or any
  vectors of same length for that matter.
  '''
  face_vec = np.array(a - b)
  face_vec[1:] = face_vec[1:] - face_vec[:-1]
  face_vec[0] = 0
  return face_vec[:,np.newaxis]

def get_all_aug_in_seq(aug_h5, file_name, gest_type, start, end, win_len=16):
  '''
  Returns a list of all augmentations in a given sequence.
  '''
  aug_h5_f = h5py.File(aug_h5, 'r')
  aug_gest = aug_h5_f[file_name][str(gest_type)]
  aug_gest_keys = aug_gest.keys()
  X_aug = []
  for i in range(start, end+1):
    if str(i) in aug_gest_keys:
      X_aug.append(np.array(aug_gest[str(i)][str(win_len)]))

  aug_h5_f.close()
  return X_aug

def get_middle_3_elements(l):
  assert(len(l) > 0)
  mid =  len(l) // 2
  elements = []
  if mid-1>=0:
    elements.append(l[mid-1])
  elements.append(l[mid])
  if mid+1 < len(l):
    elements.append(l[mid+1])
  return elements

def process_aug_zface_data(X, **kwargs):
  return __process_zface_data(X, **kwargs)

def process_zface_data(X, **kwargs):
  return __process_zface_data(X, **kwargs)

def __process_zface_data(X, **kwargs):
  mean, std = kwargs.get('mean', None), kwargs.get('std', None)
  assert(mean is None)

  transition_vel = np.zeros((X.shape[0], 2))
  transition_vel[1:,:] = X[1:,1:3] - X[:-1,1:3]
  angular_vel = np.zeros((X.shape[0], 3))
  angular_vel[1:, :] = X[1:,3:6] - X[:-1, 3:6]

  X_new = np.hstack([X[:,[0,3,4,5]], transition_vel, angular_vel])
  return X_new

def process_aug_cpm_data(X, **kwargs):
  return __process_cpm_data(X, **dict(kwargs,
    aug=True))

def process_cpm_data(X, **kwargs):
  '''
  Look at conv_lstm/torch/utils.lua for indexes.
  '''
  return __process_cpm_data(X, **kwargs)

def __process_cpm_data(X, aug=False, **kwargs):
  mean, std = kwargs.get('mean', None), kwargs.get('std', None)
  assert(mean is None)
  nose_X = X[:,0:2]  # First index is nose_x, nose_y
  neck_X = X[:,2:4]  # Second index is neck_x, neck_y
  nose_vel, neck_vel = np.zeros(nose_X.shape), np.zeros(neck_X.shape)
  nose_vel[1:,:] = nose_X[1:,:] - nose_X[:-1,:]
  neck_vel[1:,:] = neck_X[1:,:] - neck_X[:-1,:]

  l_shldr_idx = get_cpm_joint_type_to_h5_idx('LShoulder')
  r_shldr_idx = get_cpm_joint_type_to_h5_idx('RShoulder')
  if aug:
    l_shldr_idx, r_shldr_idx = 6, 4

  l_shldr_vel, r_shldr_vel = np.zeros((X.shape[0], 2)), np.zeros((X.shape[0],2))
  l_shldr_vel[1:,:] = X[1:,l_shldr_idx:l_shldr_idx+2] - \
      X[:-1,l_shldr_idx:l_shldr_idx+2]
  r_shldr_vel[1:,:] = X[1:,r_shldr_idx:r_shldr_idx+2] - \
      X[:-1,r_shldr_idx:r_shldr_idx+2]

  return np.hstack([nose_vel, neck_vel, l_shldr_vel, r_shldr_vel])


def process_aug_data(X, y, **kwargs):
  return __process_single_gesture( X, y, **dict(kwargs,
    face_landmarks=[
      1+11, 3+11,
      1+1+11, 3+1+11,
      5+11, 3+11,
      5+1+11, 3+1+11,
      13+11, 3+11,
      13+1+11, 3+1+11]
    ))

def process_single_data(X, y, **kwargs):
  return __process_single_gesture( X, y, **dict(kwargs,
    face_landmarks=[
      28+11, 34+11,
      28+11+68, 34+11+68,
      2+11, 34+11,
      2+11+68, 34+11+68,
      14+11, 34+11,
      14+11+68, 34+11+68]
    ))

def __process_single_gesture(X, y, **kwargs):
  mean, std = kwargs.get('mean', None), kwargs.get('std', None)
  normalize_signals = kwargs.get('normalize_signals', None)
  # Facial landmarks shorthand
  fl = kwargs['face_landmarks']

  # Get pose and gaze
  X_pose = X[:, :12]
  X_pose[:,9:12] = np.rad2deg(X_pose[:, 9:12])

  X_pose_diff = np.zeros((X_pose.shape[0], 6))
  X_pose_diff[1:,:] = X_pose[1:, 6:12] - X_pose[:-1, 6:12]

  landmarks = OPENFACE_LANDMARKS_TO_USE[:]
  X_landmarks = X[:, landmarks]
  # Take landmarks difference for these vectors
  X_landmarks_diff = np.zeros(X_landmarks.shape)
  X_landmarks_diff[1:,:] = X[1:, landmarks] - X[:-1, landmarks]


  # The math here is different since we store the landmarks directly in our
  # augmentations and not the velocities or the difference vectors so the
  # difference has to be taken with respect to a few different vectors.
  X_face_vec_1_x = get_face_difference(X[:,fl[0]], X[:, fl[1]])
  X_face_vec_1_y = get_face_difference(X[:,fl[2]], X[:, fl[3]])

  X_face_vec_2_x = get_face_difference(X[:,fl[4]], X[:,fl[5]])
  X_face_vec_2_y = get_face_difference(X[:,fl[6]], X[:,fl[7]])

  X_face_vec_3_x = get_face_difference(X[:,fl[8]], X[:,fl[9]])
  X_face_vec_3_y = get_face_difference(X[:,fl[10]], X[:,fl[11]])

  #X = np.hstack([X_pose, X_pose_diff, X_landmarks, X_landmarks_diff,
  #  X_face_vec_1_x, X_face_vec_1_y, X_face_vec_2_x, X_face_vec_2_y,
  #  X_face_vec_3_x, X_face_vec_3_y])
  X = np.hstack([X_landmarks, X_landmarks_diff,
    X_face_vec_1_x, X_face_vec_1_y, X_face_vec_2_x, X_face_vec_2_y,
    X_face_vec_3_x, X_face_vec_3_y])

  if mean is None:
    assert('Not implemented')
  else:
    # Normalize landmarks
    X = normalize_openface_feats(X, mean, std, landmarks)

  return X, y

def process_data(X, y, cpm_X=None, **kwargs):

  """
  Process the data set to do normalization and other clean up techniques.
  """
  assert(X.shape[1] > NUM_FEATURES)
  X = X[:, :148]

  # Includes pose as well as gaze.
  X_pose = X[:, :12]
  X_pose[:,9:12] = np.rad2deg(X_pose[:, 9:12])

  X_pose_diff = X_pose[:-1, 6:12] - X_pose[1:, 6:12]
  X_pose_diff = np.vstack((np.zeros(X_pose_diff.shape[1]), X_pose_diff))

  # Add specific landmarks. First landmark is indexed as 1.
  landmarks = [
      28, 28 + 68, # forehead
      34, 34 + 68, # nosetip
      2,   2 + 68, # left side of face
      4,   4 + 68,
      8,   8 + 68, # bottom (right)
      10, 10 + 68,
      14, 14 + 68, # top
      16, 16 + 68]
  # There are 12 axes before 1st landmark but the landmark has been
  # indexed from 1 in OpenFace's notation. Therefore for landmark=1 the
  # correct index here would have been 12 = (11 + 1). Hence below we add
  # 11 to all landmark indexes.
  l = [l1 + 11 for l1 in landmarks]
  X_landmarks = X[:, l]

  # Maybe take a difference for these vectors
  X_landmarks_diff = X[1:, landmarks] - X[:-1, landmarks]
  X_landmarks_diff = np.vstack((np.zeros(16), X_landmarks_diff))

  # TODO(Mohit): Maybe take a difference for these vectors
  # Take 4 direction vectors on face which might change as we move
  X_face_vec_1 = np.array(
     [X[:, 28+11] - X[:, 34+11], X[:, 28+68+11] - X[:, 34+68+11]]).T
  X_face_vec_2 = np.array(
     [X[:, 3+11] - X[:, 34+11], X[:, 3+68+11] - X[:, 34+68+11]]).T
  X_face_vec_3 = np.array(
     [X[:, 15+11] - X[:, 34+11], X[:, 15+68+11] - X[:, 34+68+11]]).T

  dx_nose_ang_vel = np.abs(X_face_vec_1[:, 0])
  dx_nose_ang_vel[dx_nose_ang_vel == 0] = 0.00001  # Avoiding division by 0
  nose_forehead_angle = np.arctan(np.abs(X_face_vec_1[:, 1]) / dx_nose_ang_vel)
  nose_forehead_angle = nose_forehead_angle * 180.0/np.pi
  nose_forehead_angle = nose_forehead_angle[:,np.newaxis]
  nose_forehead_angle[1:,:] = nose_forehead_angle[1:,:] - \
      nose_forehead_angle[0:-1,:]

  if cpm_X is None:
    X = np.hstack(([X_pose, X_pose_diff, X_landmarks, X_landmarks_diff,
      X_face_vec_1, X_face_vec_2, X_face_vec_3, nose_forehead_angle]))
  else:
    final_cpm_X = process_cpm_data(cpm_X)
    final_cpm_X = final_cpm_X[:X_pose.shape[0],:]
    X = np.hstack(([X_pose, X_pose_diff, X_landmarks, X_landmarks_diff,
      X_face_vec_1, X_face_vec_2, X_face_vec_3, nose_forehead_angle,
      final_cpm_X]))

  # Let's only classify ticks
  # We shouldn't do 1-vs-all classification here since we lose the relevant
  # spatial info i.e. we won't know what gesture it really was while
  # grouping things in windows. We should do this as a post processing
  # step.  X, y = do_one_vs_all(X, y, 5)
  # Normalize only the landmarks (not first differences and pose)
  norm_signals = [6,7,8] # + range(15, 31)
  X, y = __normalize_data_each_sensor_signal(X, y, signals=norm_signals)
  return X, y

def normalize_conf(conf):
  '''
  Normalize confusion matrix for better plotting.
  '''
  norm_conf = []
  for i in conf:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
      if float(a):
        tmp_arr.append(float(j)/float(a))
      else:
        tmp_arr.append(float(j))
    norm_conf.append(tmp_arr)
  return norm_conf

def get_f1_score(conf, weights):
  prec = np.zeros(conf.shape[0])
  recall = np.zeros(conf.shape[0])

  f1 = np.zeros(conf.shape[0])
  for i in xrange(conf.shape[0]):
      if np.sum(conf[:, i]) != 0:
          prec[i] = float(conf[i,i])/np.sum(conf[:, i])
      else:
          prec[i] = 0

      if np.sum(conf[i, :]) != 0:
          recall[i] = float(conf[i,i])/np.sum(conf[i,:])
      else:
          recall[i] = 0

      if (prec[i] + recall[i] == 0):
          f1[i] = 0
      else:
          f1[i] = (2.0 * prec[i] * recall[i]) / (prec[i] + recall[i])
  weights = weights / np.sum(weights)
  return np.sum(weights * f1)

def get_sorted_checkpoints(fdir, prefix, sort_cmp):
    targets = []
    for f in os.listdir(fdir):
        if f.startswith(prefix):
            targets.append(f)
    targets.sort(key=sort_cmp)
    return targets

def computeKappa(conf):
    conf = conf.astype(float)
    N = conf.shape[0]
    t = np.arange(1, N+1).astype(float)
    tmp = np.repeat(t[:,np.newaxis], N, axis=1).T
    tmp1 = np.repeat(t[:, np.newaxis], N, axis=1)
    W = ((tmp - tmp1) * (tmp - tmp1)) / (N-1.0)**2
    total_sum = np.sum(conf)
    row_sum = np.sum(conf, 0) / total_sum
    col_sum = np.sum(conf, 1)
    row_sum = np.repeat(row_sum[:, np.newaxis], N, axis=1).T
    col_sum = np.repeat(col_sum[:, np.newaxis], N, axis=1)
    E = row_sum * col_sum
    den = np.sum(W*E)
    kappa = 1.0 - float(np.sum(W*conf))/den
    return kappa

def computeWeightedF1(conf):
    wt = np.zeros(conf.shape[0])
    for i in range(conf.shape[0]):
        wt[i] = np.sum(conf[i,:])
    wt = wt/np.sum(wt)
    return get_f1_score(conf, wt)

def calculate_mean_std(h5_dir, h5_group, save_h5_name, prefix=None, suffix=None):
  save_h5_path = os.path.join(h5_dir, save_h5_name)
  d = {'mean': {}, 'std': {}, 'max': {}, 'min': {}}
  for f in os.listdir(h5_dir):
    if prefix is not None and not f.startswith(prefix):
      continue
    if suffix is not None and not f.endswith(suffix):
      continue
    f_path = os.path.join(h5_dir, f)
    f_h5 = h5py.File(f_path, 'r')
    X = np.array(f_h5[h5_group])
    f_h5.close()
    # Save mean and std for each column of X
    m, s, _max, _min = [], [], [], []
    assert(X.shape[1] > 0)
    for i in range(X.shape[1]):
      curr_X = X[200:,i]
      curr_X, num_outliers = reject_outliers(curr_X, m=20.)
      if num_outliers > 0:
          print("File {}, col: {}, #outliers: {}".format(
              f, i, num_outliers))
      m.append(np.mean(curr_X))
      s.append(np.std(curr_X))
      _max.append(np.max(curr_X))
      _min.append(np.min(curr_X))
    d['mean'][f] = np.array(m)
    d['std'][f] = np.array(s)
    d['max'][f] = np.array(_max)
    d['min'][f] = np.array(_min)

  save_h5 = h5py.File(save_h5_path, 'w')
  recursively_save_dict_contents_to_group(save_h5, '/', d)
  save_h5.flush()
  save_h5.close()
  print('Did write mean/std to {}'.format(save_h5_path))

def get_classif_class_for_gest_5(gest_type):
    if gest_type == 0:
      return 0
    elif gest_type >= 1 and gest_type <= 5:
      return 1
    elif gest_type == 6:
      return 2
    elif gest_type == 7 or gest_type == 8:
      return 3
    elif gest_type == 9 or gest_type == 10:
      return 4
    else:
      raise ValueError

def group_gest_into_classes(gest_by_type, num_classify):
    '''
    Groups gestures into number of classes we want to classify in.
    gest_by_type will contain alltu
    '''
    gest_by_classify = [[] for _ in xrange(num_classify)]
    for i in xrange(len(gest_by_type)):
      new_class = get_classif_class_for_gest_5(i)
      gest_by_classify[new_class] += gest_by_type[i]
    return gest_by_classify

def load_all_features(h5dir):
  '''
  Loads all openface features for all valid h5 files in `h5dir`.
  '''
  X_by_file = {}
  y_by_file = {}

  for npfile in os.listdir(h5dir):
    if npfile.endswith('static.mp4.txt.h5'):
      hf = h5py.File(os.path.join(h5dir, npfile))
      a = np.array(hf.get('annotations')).astype(int)
      f = np.array(hf.get('features'))
      X, y = np.copy(f), np.copy(a)
      X, y = process_data(X, y)
      X_by_file[npfile] = X
      y_by_file[npfile] = y
  return X_by_file, y_by_file

def recursively_get_list_of_all_keys(d, all_keys, curr_keys):
  if type(d) != type({}):
    all_keys.append(deepcopy(curr_keys))
    return
  else:
    for k in d.keys():
      curr_keys.append(k)
      recursively_get_list_of_all_keys(d[k], all_keys, curr_keys)
      curr_keys.pop()

def save_list_as_matlab_cell_array(save_list, mat_file, var_name):
  cell_array = np.empty((1,len(save_list)), dtype=object)
  for i, l in enumerate(save_list):
    cell_array[0, i] = l
  sio.savemat(mat_file, {var_name: cell_array})
  print('Did save {} to {}'.format(var_name, mat_file))


def read_mean_files(openface_mean_h5, cpm_mean_h5):
  '''
  Read Openface and CPM mean files and return all values.
  '''
  openface_h5_f = h5py.File(openface_mean_h5, 'r')
  openface_mean_std = recursively_get_dict_from_group(openface_h5_f)
  openface_h5_f.close()
  cpm_h5_f = h5py.File(cpm_mean_h5, 'r')
  cpm_mean_std = recursively_get_dict_from_group(cpm_h5_f)
  cpm_h5_f.close()
  return openface_mean_std, cpm_mean_std

# Reject outliers, since OpenFace can often lose tracking one of its
# consequences is that it leads to very high values when prob. of finding
# a face is low which skews calculations. Hence we do outlier rejection
# based on median here.
def reject_outliers(data, m = 10.):
    data_copy = np.copy(data)
    d = np.abs(data_copy - np.median(data_copy))
    mdev = np.median(d)
    s = d/mdev if mdev else d
    if mdev == 0:
        print("Median is 0 maybe check manually.")
    data_copy[s>m] = 0
    # print(zip(*(s>m).nonzero()))
    did_find_outlier = True if np.sum(s>m) > 0 else False
    return data_copy, np.sum(s>m)


def get_bin_in_sorted_list(data, val):
    '''
    Returns the index for val in data s.t. data[i] <= val and data[i-1] > val
    '''    
    for i,v in enumerate(data):
        if v >= val:
            return i
    return len(data)

def read_json(json_file_path):
    data = None
    with open(json_file_path, 'r') as fp:
        data = json.load(fp)
    return data
        
