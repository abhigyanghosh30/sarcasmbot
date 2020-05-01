#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pickle
import re
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Embedding, LSTM
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras_self_attention import SeqSelfAttention
import tensorflow
tensorflow.debugging.set_log_device_placement(True)


# In[ ]:


def classification_labels(row):
    if row["label"] == 1:
        return "Negative"
    return "Positive"


def main():
    """Function for classifying sentences into positive and negative"""
    model_dir = "./models"

    # Data reading and pre-processing
    data = pd.read_csv('sentiment_data.csv')
    data.columns = ["text", "label"]
    data['text'] = data['text'].apply(lambda x: x.lower())
    data["sentiment"] = data.apply(classification_labels, axis=1)

    data.drop("label", axis=1, inplace=True)
    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

    maxlen = 30
    max_features = 20000
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(data['text'].values)
    X = tokenizer.texts_to_sequences(data['text'].values)
    X = pad_sequences(X, maxlen=maxlen)

    token_path = os.path.join(model_dir, "token_" + "v1")
    complete_token_path = token_path + ".pickle"
    with open(complete_token_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(X.shape)

    embed_dim = 128
    inputs = Input(shape=(maxlen,))
    emb1 = Embedding(max_features, embed_dim, mask_zero=True)(inputs)
    lstm1 = LSTM(200, return_sequences=True)(emb1)
    lstm_out, att_weights = SeqSelfAttention(attention_activation='sigmoid', return_attention=True)(lstm1)
    lstm2 = LSTM(150, return_sequences=False, trainable=False)(lstm_out)
    outputs = Dense(2, activation='sigmoid')(lstm2)
    model = Model(inputs=[inputs], outputs=outputs)
    print(model.summary())

    Y = pd.get_dummies(data['sentiment']).values
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    batch_size = 32
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model_name = "dummy_model_" + "v1" + "_embed_" + str(embed_dim) + "_batch_size_" + str(32)
    filepath = os.path.join(model_dir, model_name) + "_intermediate.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    epochs = 10
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test), verbose=1, callbacks=callbacks_list)

    print("Training has finished. Model save at ", os.path.join(model_dir, model_name) + "_final.hdf5")
    model.save(os.path.join(model_dir, model_name) + "_final.hdf5")
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

    print('Test score:', score)
    print('Test accuracy:', acc)

main()


# In[ ]:


import numpy as np
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model, Model
from keras_self_attention import SeqSelfAttention
from sklearn.preprocessing import OneHotEncoder

def main():
    """Function for removing sentiment words from a given text"""

    data = pd.read_csv('dataSampleTest.csv')
    data.columns = ["sarcasmText", "text"] # Keeping only the neccessary columns

    stop = stopwords.words('english')
    data['text_without_stopwords'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))
    data['text_without_stopwords'] = data['text_without_stopwords'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply(lambda x: x.lower())

    model_dir = "./models"
    token_path = os.path.join(model_dir, "token_" + "v1")
    complete_token_path = token_path + ".pickle"
    with open(complete_token_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    maxlen =30
    X = tokenizer.texts_to_sequences(data['text'].values)
    X = pad_sequences(X, maxlen=maxlen)
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    print(X.shape)

    max_features = 20000
    enc = OneHotEncoder(handle_unknown='ignore', n_values=max_features, sparse=False)
    
    x_train_one_hot = enc.fit_transform(X)
    x_train_one_hot = np.reshape(x_train_one_hot, (X.shape[0], maxlen, max_features))

    model_name = "dummy_model_" + "v1" + "_embed_" + str(128) + "_batch_size_" + str(32)
    trained_model = os.path.join(model_dir, model_name) + "_final.hdf5"

    model = load_model(trained_model, custom_objects={'SeqSelfAttention': SeqSelfAttention})

    feat_dir = "./features"
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)
    print(model.summary())

    dense_model = Model(inputs=model.input, outputs=model.get_layer('seq_self_attention_1').output)
    dense_feature, attn_weight = dense_model.predict(X)

    new_data_all = np.zeros((attn_weight.shape[0], attn_weight.shape[1]))
    for i in range(0, attn_weight.shape[0]):
        current_max_array = attn_weight[i].max(0)
        temp_list = []
        for k in range(0, current_max_array.shape[0]):
            if current_max_array[k] != 0:
                temp_list.append(current_max_array[k])

        current_mean = np.mean(temp_list)
        current_std = np.std(temp_list)
        num_higher = current_mean + 1*(current_std)
        num_lower = current_mean - 1.5*(current_std)
        high_outlier = (current_max_array <= num_higher).astype(int)
        low_outlier = (current_max_array > num_lower).astype(int)
        context_ones = high_outlier*low_outlier
        new_data = X[i] * context_ones
        new_data_all[i] = new_data

    def sequence_to_text(list_of_indices):
        words = [reverse_word_map.get(letter) for letter in list_of_indices]
        return words

    my_texts = list(map(sequence_to_text, new_data_all))
    correct_sent = list(data["text"])
    all_sentences = []
    new_training_output = []
    for i in range(0, len(my_texts)):
        each_new = [x for x in my_texts[i] if x is not None]
        each_new = " ".join(each_new)
        if each_new != "":
            rem = correct_sent[i]
            new_training_output.append(rem)
            all_sentences.append(each_new)
        else:
            print(i, correct_sent[i])

    print(len(all_sentences), len(new_training_output))
    for i in range(0, len(new_training_output)):
        print(all_sentences[i], " ----------- ", new_training_output[i])

main()

