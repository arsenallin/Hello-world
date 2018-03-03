from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')
parser.add_argument('--price_norm_factor', default=1000., type=float,
                    help='price normalization factor')
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

# Order is important for the csv-readers, so we use an OrderedDict here.
COLUMN_TYPES = collections.OrderedDict([
    ("symboling", int),
    ("normalized-losses", float),
    ("make", str),
    ("fuel-type", str),
    ("aspiration", str),
    ("num-of-doors", str),
    ("body-style", str),
    ("drive-wheels", str),
    ("engine-location", str),
    ("wheel-base", float),
    ("length", float),
    ("width", float),
    ("height", float),
    ("curb-weight", float),
    ("engine-type", str),
    ("num-of-cylinders", str),
    ("engine-size", float),
    ("fuel-system", str),
    ("bore", float),
    ("stroke", float),
    ("compression-ratio", float),
    ("horsepower", float),
    ("peak-rpm", float),
    ("city-mpg", float),
    ("highway-mpg", float),
    ("price", float)
])

def raw_dataframe():
    # download and cache the data
    path = tf.keras.utils.get_file(URL.split("/")[-1], URL)
    # load it into pandas dataframe
    df = pd.read_csv(path, names=COLUMN_TYPES.keys(), dtype=COLUMN_TYPES, na_values='?')
    return df
def load_data(y_name='price', train_fraction=0.7, seed=None):
    """Get the import85 data set.
    Args:
            y_name: the column returns as the label.
            train_fraction: the fraction of the dataset to use for training.
            seed: the random seed to use when shuffling the data. None generates
                    a unique shuffle every run.
    Returns:
            a pair of pairs where the first pair is the training data,
            and the second is the test data."""
    # load he raw data columns
    data = raw_dataframe()
    # delete rows with unknowns
    data = data.dropna()
    # shuffle the data
    np.random.seed(seed)

    # split the data into train/set subsets
    x_train = data.sample(frac=train_fraction, random_state=seed)
    x_test = data.drop(x_train.index)

    # extract the label from the features dataframe
    y_train = x_train.pop(y_name)
    y_test = x_test.pop(y_name)
    return (x_train, y_train), (x_test, y_test)

def make_dataset(x, y=None):
    """Create a slice dataset from a pandas Dataframe and labels"""
    # convert the dataframe into a dict
    x = dict(x)
    # convert the pd.series to np.arrays
    for key in x:
        x[key] = np.array(x[key])

    items = [x]
    if y is not None:
        items.append(np.array(y, dtype=np.float32))
    # create a dataset of slices
    return tf.data.Dataset.from_tensor_slices(tuple(items))

def from_dataset(ds):
    return lambda : ds.make_one_shot_iterator().get_next()

def main(argv):
    """Builds, trains and evaluates the model."""
    args = parser.parse_args(argv[1:])
    (train_x, train_y), (test_x, test_y) = load_data()
    train_y /= args.price_norm_factor
    test_y /= args.price_norm_factor

    # build the training dataset
    train = (
        make_dataset(train_x, train_y).shuffle(1000).batch(args.batch_size).repeat()
        # shuffing with a buffer larger than the dataset ensures that the examples are well mixed.
        # repeat forever
    )

    # build the validation dataset
    test = make_dataset(test_x, test_y).batch(args.batch_size)

    feature_columns = [
        # "curb-weight" and "highway-mpg" are numeric columns.
        tf.feature_column.numeric_column(key='curb-weight'),
        tf.feature_column.numeric_column(key='highway-mpg')
    ]
    # Build the estimator
    model = tf.estimator.LinearRegressor(feature_columns=feature_columns)

    # Train the model, by default, the estimators log out put every 100 steps.
    model.train(input_fn=from_dataset(train), steps=args.train_steps)

    # evaluate how the model performs on data it has not seen yet
    eval_result = model.evaluate(input_fn=from_dataset(test))

    # the evaluation returns a python dictionary.
    # the average loss key holds the mean squared error
    average_loss = eval_result['average_loss']

    # convert MSE to root mean square error
    print('\n' + 80 * '*')
    print('\nRMS error for the test set :${:.0f}'.format(args.price_norm_factor * average_loss ** 0.5))

    # Run the model in prediction mode
    input_dict = {
        'curb-weight': np.array([2000, 3000]),
        'highway-mpg': np.array([30,40])
    }
    predict = make_dataset(input_dict).batch(1)
    predict_results = model.predict(input_fn=from_dataset(predict))
    print("\nPrediction results:")
    for i, prediction in enumerate(predict_results):
        msg = ("Curb weight: {: 4d}lbs, "
           "Highway: {: 0d}mpg, "
           "Prediction: ${: 9.2f}")
        msg = msg.format(input_dict['curb-weight'][i], input_dict['highway-mpg'][i],
                         args.price_norm_factor * prediction['predictions'][0])
        print('    ' + msg)
    print()

if __name__ == '__main__':
    # the estimator periodically generates INFO logs, make these logs visible
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
