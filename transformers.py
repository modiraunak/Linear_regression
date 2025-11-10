#importing neccessary libraries 
import tensorflow as tf
from keras import layers 
from keras.datasets import imdb
from keras.preprocessing import sequence

#load and preprocess the IMDB datset 
max_features = 10000
max_len = 200

(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train,maxlen=max_len)
x_test = sequence.pad_sequences(x_test,maxlen=max_len)

#define Transformer block
class TransformerBlock(layers.Layer):
    def __init__(self,embed_dim,num_heads,ff_dim,rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim)
        ]) 
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs,inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output,training=training)
        return self.layernorm2(out1 + ffn_output)
    
embed_dim = 32
num_heads = 2
ff_dim = 32

inputs = layers.Input(shape=(max_len,))
embeding_layer = layers.Embedding(input_dim=max_features, output_dim=embed_dim)
x = embeding_layer(inputs)
transformer_block=TransformerBlock(embed_dim, num_heads, ff_dim)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1,activation="sigmoid")(x)
model = tf.keras.Model(inputs=inputs,outputs=outputs)
#compile 
model.compile(optimizer="adam", loss="binary_crossentropy", metrics =['accuracy'])
model.fit(x_train,y_train,batch_size=64,epochs=3,validation_split=0.2)