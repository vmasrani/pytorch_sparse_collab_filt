from collections import defaultdict
import csv
import os
try:
    import cPickle as pickle
except:
    import pickle

import numpy as np
from sklearn.feature_extraction import DictVectorizer
import scipy.sparse as sp
from scipy.sparse import rand as sprand

from torch.utils.data import DataLoader


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_ratings(rating_path, dataset):
    """
    Returns a NxM unnormalized ratings matrix. 
    - For the delcious dataset, a user's 'rating' of
    a url is a count of how many times
    a user bookmarks a url.

    - For the lastfm dataset, a user's 'rating' of
    a artist is a count of how many times
    a user listens to an artist.

    Returns: 
    R: an unnormalized ratings matrix,
    """
    ratings = defaultdict(lambda: defaultdict(int))

    # Tally number of tags/bookmarks per product per user
    with open(rating_path, 'rb') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')
        # Skip header
        next(tsvin)
        for row in tsvin:
            if dataset == 'delicious':
                user_id, item_id = int(row[0]), int(row[1])
                weight = 1
            else:
                user_id, item_id, weight = int(
                    row[0]), int(row[1]), int(row[2])
            ratings[user_id][item_id] += weight

    # Make list of dicts to feed to DictVectorizer
    user_ids = sorted(ratings.keys())

    data = []
    for user in user_ids:
        datum = {}
        for product_id in ratings[user].keys():
            datum[product_id] = ratings[user][product_id]
        data += [datum]
    vectorizer = DictVectorizer()
    R = vectorizer.fit_transform(data)

    return R


def load_data(dataset='delicious', clear_cache=False):
    assert dataset in ['delicious',
                       'lastfm'], 'Data must be either "delicious" or "lastfm"'
    print "Loading %s ... " % dataset
    path = "data/pickle/%s.pickle" % dataset
    try:
        if clear_cache:
            os.remove(path)
        R = pickle.load(open(path, "rb"))[0]
        print "Loaded from: %s" % path
    except (OSError, IOError):
        R = _load_data(dataset)
        pickle.dump([R], open(path, "wb"))

    return R


def _load_data(dataset='delicious'):
    if dataset == 'delicious':
        rating_path = 'data/delicious/user_taggedbookmarks.dat'
    else:
        rating_path = 'data/lastfm/user_artists.dat'

    R = get_ratings(rating_path, dataset)

    return R


def train_test_split(R, test_size=0.2):
    # We use our own method because we want to 'punch holes' in the
    # labels rather than remove entire rows like normal train/test splits

    R = R.tocoo()
    nnz = R.nnz
    N, D = R.shape
    triplets = zip(R.data, R.row, R.col)

    # Get train/test idx
    test_idx = sorted(np.random.choice(
        nnz, int(nnz * test_size), replace=False))
    mask = np.ones(nnz, dtype=bool)
    mask[test_idx] = False
    train_idx = np.arange(nnz)[mask]

    R_train = [triplets[i] for i in train_idx]
    R_test = [triplets[i] for i in test_idx]

    def build_sparse(zipped):
        data, row, col = zipped
        return sp.coo_matrix((data, (row, col)), shape=(N, D))

    R_train = build_sparse(zip(*R_train))
    R_test = build_sparse(zip(*R_test))

    assert R_train.shape == R_test.shape == R.shape
    assert R.nnz == R_test.nnz + R_train.nnz

    return R_train, R_test


def load_simulated_data():
    n_users = 1000
    n_items = 1000

    ratings = sprand(n_users, n_items, density=0.01, format='csr')
    data = (np.random.randint(1, 5, size=ratings.nnz).astype(np.float64))
    ratings.data = data

    return ratings


def make_data_loader(R, batch_size):
    row, col = R.nonzero()
    tuple_data = zip(row, col, R.data)
    return DataLoader(tuple_data, batch_size, shuffle=True)
