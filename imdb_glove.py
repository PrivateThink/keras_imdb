import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
import os

max_features = 20000
max_len = 100
batch_size = 32


def load_data_and_labels(positive_data_file, negative_data_file):
    # 读取正向的情感句子
    positive_examples = list(open(positive_data_file,encoding="utf8").readlines())
    # 将句子两边的空格去掉
    positive_examples = [s.strip() for s in positive_examples]
    # 读取负向的情感句子
    negative_examples = list(open(negative_data_file,encoding="utf8").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # 将正向和负向的情感句子合并，
    data = positive_examples + negative_examples
    data = [sent for sent in data]

    # 生成标签，正向的句子标签为[0,1],负向的句子标签为[1, 0]
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    labels = np.concatenate([positive_labels, negative_labels], 0)
    # 打乱数据
    train, test, train_labels, test_labels = train_test_split(data,labels, test_size=0.2,shuffle=True)
    return train, test, train_labels, test_labels


# 获取训练集和测试集
train, test, train_labels, test_labels=load_data_and_labels("data/rt-polarity.pos","data/rt-polarity.neg")

print("训练集的大小：",len(train))
print("测试集的大小：",len(test))
print(train[0])

# 对句子进行序列映射,将文本转化为数值
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train)
word_index=tokenizer.word_index
train = tokenizer.texts_to_sequences(train)
test = tokenizer.texts_to_sequences(test)

# 将句子的长度最大设置为maxlen=100,少于100的补零，超过100的截断
train = sequence.pad_sequences(train, maxlen=max_len)
test = sequence.pad_sequences(test, maxlen=max_len)


embeddings_index = {}
f = open(os.path.join("data/", 'glove.6B.300d.txt'),encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print(('Total %s word vectors.' % len(embeddings_index)))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in list(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# 创建网络结构
model = Sequential()
model.add(Embedding(max_features, 300, input_length=max_len))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))

model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(train, train_labels,
          batch_size=batch_size,
          epochs=3,
          validation_data=[test, test_labels])



