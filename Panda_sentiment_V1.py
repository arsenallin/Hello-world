import numpy as np
import tensorflow as tf
import random
import pickle
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

pos_file = 'pos.txt'
neg_file = 'neg.txt'

# create the lexicon, the words that shown in the file.
def create_lexicon(pos_file, neg_file):
    lex = []
    # read in the file
    def process_file(f):
        with open(pos_file, 'r') as f:
            lex = []
            lines = f.readlines()
            for line in lines:
                words = word_tokenize(line.lower())
                lex += words
        return lex

    lex += process_file(pos_file)
    lex += process_file(neg_file)

    lemmatizer = WordNetLemmatizer()
    lex = [lemmatizer.lemmatize(word) for word in lex]  # 还原(cats->cat)

    word_count = Counter(lex)
    # print(word_count)
    # {'.': 13944, ',': 10536, 'the': 10120, 'a': 9444, 'and': 7108, 'of': 6624, 'it': 4748, 'to': 3940......}
    # 去掉一些常用词,像the,a and等等，和一些不常用词; 这些词对判断一个评论是正面还是负面没有做任何贡献
    lex = []
    for word in word_count:
        if word_count[word] < 2000 and word_count[word] > 20:
            lex.append(word)
            # 齐普夫定律-使用Python验证文本的Zipf分布 http://blog.topspeedsnail.com/archives/9546
    return lex
lex = create_lexicon(pos_file, neg_file)
# print(lex)
# ['rock', 'be', 'century', 'new', '``', 'he', 'going', 'make', 'even', ...

def normalize_dataset(lex):
    dataset = []
    def string_to_vector(lex, review, clf):
        # lex:词汇表；review:评论；clf:评论对应的分类，[0,1]代表负面评论 [1,0]代表正面评论
        words = word_tokenize(review.lower())
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        features = np.zeros(len(lex))
        for word in words:
            if word in lex:
                features[lex.index(word)] = 1
        return [features, clf]
    with open(pos_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample =string_to_vector(lex, line, [1,0])
            dataset.append(one_sample)
    with open(neg_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample =string_to_vector(lex, line, [0,1])
            dataset.append(one_sample)
    return dataset

dataset = normalize_dataset(lex)
random.shuffle(dataset)

test_size = int(len(dataset) * 0.1)
dataset = np.array(dataset)
train_dataset = dataset[:-test_size]
test_dataset = dataset[-test_size:]

# Feed-Forward Neural Network
n_input_layer = len(lex)

n_layer_1 = 1000
n_layer_2 = 1000
n_output_layer = 2

def neural_network(data):
    layer_1_w_b = {'w_': tf.Variable(tf.random_normal([n_input_layer, n_layer_1])),
                   'b_': tf.Variable(tf.random_normal([n_layer_1]))}
    layer_2_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_1, n_layer_2])),
                   'b_': tf.Variable(tf.random_normal([n_layer_2]))}
    layer_output_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_2, n_output_layer])),
                   'b_': tf.Variable(tf.random_normal([n_output_layer]))}

    layer_1 = tf.nn.relu(tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_']))
    layer_output = tf.add(tf.matmul(layer_2, layer_output_w_b['w_']), layer_output_w_b['b_'])

    return layer_output

batch_size = 50
X = tf.placeholder('float', [None, len(train_dataset[0][0])])
# [None, len(train_x)]代表数据数据的高和宽（矩阵），好处是如果数据不符合宽高，tensorflow会报错，不指定也可以。
Y = tf.placeholder('float')

def train_neural_network(X, Y):
    predict = neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)

    epochs = 13
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        epoch_loss = 0

        i = 0
        random.shuffle(train_dataset)
        train_x = dataset[:, 0]
        train_y = dataset[:, 1]
        for eopch in range(epochs):
            while i < len(train_x):
                start = i
                end = i + batch_size

                batch_x = train_x[start:end]
                batch_y = train_y[start:end]

                _, c = sess.run([optimizer, cost_func], feed_dict={X:list(batch_x), Y:list(batch_y)})
                epoch_loss += c
                i += batch_size
            print(eopch, ' : ', epoch_loss)
        test_x = test_dataset[:, 0]
        test_y = test_dataset[:, 1]
        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({X:list(test_x), Y:list(test_y)}))
train_neural_network(X, Y)
