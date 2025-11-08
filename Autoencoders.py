from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import numpy as np


#sample data
x = np.array([[1,2,3,4,5],
              [2,3,4,5,6],
              [3,4,5,6,7],
              [4,5,6,7,8],
              [5,6,7,8,9],
              [8,9,10,11,12]
              ],dtype ='float')

# Define the size of the input and the encoding dimension
input_dim = x.shape[1]
encoding_dim = 2

#encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)

#decoder
decoded = Dense(input_dim, activation='sigmoid')(encoded)

#autoencoder model
autoencoder = Model(input_layer, decoded)

#compile the model
autoencoder.compile(optimizer='adam', loss='mse')

#train the model
autoencoder.fit(x, x, epochs=100, batch_size=2, shuffle=True)
#encode the data
encoder = Model(input_layer, encoded)
encoded_data = encoder.predict(x)
print("Encoded data:")
print(encoded_data)
#decode the data
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))
decoded_data = decoder.predict(encoded_data)
print("Decoded data:")
print(decoded_data)
