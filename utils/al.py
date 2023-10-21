# python imports
from timeit import default_timer as timer

# third-party imports
import numpy as np

from sklearn.utils import shuffle

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling


def active_learn_loop(classifier,
                      X_initial, y_initial,
                      X_query, y_query, n_queries):

    # copying arrays
    X_query_cpy = X_query.copy()
    y_query_cpy = y_query.copy()

    # initializing labeled arrays
    X_labeled = np.zeros((n_queries, X_query_cpy.shape[1]), dtype=int)
    y_labeled = np.zeros(n_queries, dtype=int)

    t_learn = timer()

    learner = ActiveLearner(
        estimator=classifier, query_strategy=uncertainty_sampling,
        X_training=X_initial, y_training=y_initial)

    for i in range(n_queries):
        # value selected to be learn
        query_idx, query_inst = learner.query(X_query_cpy)

        # saving labeled data by the specialists
        X_labeled[i, ] = query_inst
        y_labeled[i] = y_query_cpy[query_idx][0]

        # retraining
        learner.teach(query_inst.reshape(1, -1), y_query_cpy[query_idx])

        # remove labeled value from the pool
        X_query_cpy, y_query_cpy =\
            np.delete(X_query_cpy, query_idx, axis=0), np.delete(y_query_cpy, query_idx, axis=0)

    t_learn = timer() - t_learn

    # append entries labeled by specialists to X_initial
    X_train = np.append(X_initial, X_labeled, axis=0)
    y_train = np.append(y_initial, y_labeled)

    # shuffling data
    X_train, y_train = shuffle(X_train, y_train)

    return learner, t_learn, X_train, y_train
