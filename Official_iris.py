from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pandas as pd
import tensorflow as tf

# tell the program to shut up
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# pack the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')
parser.add_argument('--TRAIN_URL', type=str,
                    default="http://download.tensorflow.org/data/iris_training.csv")
parser.add_argument('--TEST_URL', type=str,
                    default="http://download.tensorflow.org/data/iris_test.csv")
parser.add_argument('--CSV_COLUMN_NAMES', type=list,
                    default=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species'])
parser.add_argument('--SPECIES', type=list,
                    default=['Setosa', 'Versicolor', 'Virginica'])
args = parser.parse_args()

# step 1: get the data
def load_data(y_name='Species'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path = tf.keras.utils.get_file(args.TRAIN_URL.split('/')[-1], args.TRAIN_URL)
    test_path = tf.keras.utils.get_file(args.TEST_URL.split('/')[-1], args.TEST_URL)

    train = pd.read_csv(train_path, names=args.CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=args.CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)

(train_x, train_y), (test_x, test_y) = load_data()

# step 2: define the feature columns
my_feature_columns = []
for key_iris in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key_iris))

# Step 3: build DNN network with 10, 10 units
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[10, 10],
    n_classes=3
)

# step 4: train the model
# step 4.1: define the input function
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

classifier.train(
    input_fn=lambda:train_input_fn(train_x, train_y, args.batch_size),
    steps=args.train_steps
)

# step 5: evaluate the model
# step 5.1: define the input function for evaluation or prediction
def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features. (in prediction)
        inputs = features
    else:
        inputs = (features, labels)
    print(inputs)
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

# step 5.2: Evaluate the model
eval_result = classifier.evaluate(
    input_fn=lambda:eval_input_fn(test_x, test_y, args.batch_size)
)
print(eval_result['accuracy'])

# step 6: predict something with the trained model
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}
predictions = classifier.predict(
    input_fn=lambda:eval_input_fn(predict_x, labels=None, batch_size=args.batch_size)
)

for pred_dict, expec in zip(predictions, expected):
    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print(template.format(args.SPECIES[class_id],
                          100 * probability, expec))


