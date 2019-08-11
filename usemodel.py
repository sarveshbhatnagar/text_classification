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

def review_encode(data):
    encoded = [1]

    for word in data:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded

model = keras.models.load_model("model.h5")
with open("review.txt", encoding="utf-8") as f:
	for line in f.readlines():
		nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
		encode = review_encode(nline)
		encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250) # make the data 250 words long
		predict = model.predict(encode)
		print(line)
		print(encode)
		print(predict[0])
