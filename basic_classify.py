import tensorflow as tf
from tensorflow import keras
import numpy

# getting the data
imdb = keras.datasets.imdb

#splitting into training data and testing data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# print(train_labels[0])



_word_index = imdb.get_word_index()

#get key , value pairs.
#v+3 as for special key value pairs.

word_index = {k:(v+3) for k,v in _word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

#reversing to make integer pointing to the word.
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


#Since review length differ, making max len review 250, padding shorter reviews with 0.
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)



# ? in place of words not found.
# decode_review to decode it to text.
def decode_review(text):
	return " ".join([reverse_word_index.get(i, "?") for i in text])

#see what a review looks like ?
# print(decode_review(train_data[0]))

#defining model

#embedding layer maps similar words closer 88000 word vectors and 16 dimensions
#globalaveragepooling1d basically kind of reduces the dimensions.
# embedding(many dimensions) -> average(few dimensions) -> dense layer(relu) -> dense layer(sigmoid)

model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()  # prints a summary of the model


model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#about 25000 entry in train data.
#splitting train data into train and validation data
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

#batch size is how many movie reviews loading in at once.
fitted_Model = model.fit(x_train,y_train, epochs=40, batch_size=512, validation_data=(x_val,y_val), verbose=1)
results = model.evaluate(test_data,test_labels)
print(results)
