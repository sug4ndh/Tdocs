import pickle

import numpy as np
import keras
import keras.backend as K

# This just needs to be a large numer (it's ok if exercises == 15)
N_EXERCISES = 30

def model():
    n_categories = (N_EXERCISES)*(2) + 1
    embed_dim = 20
    n_lstm_units = 20

    exercise = keras.layers.Input((None, ))
    last_exercise_x_correct = keras.layers.Input((None, ))
    last_correct = keras.layers.Input((None, ))

    embeddings = []
    for inp in [exercise, last_exercise_x_correct, last_correct]:
        # Look up 'embedding vector' for input
        emb = keras.layers.Embedding(
            n_categories,
            embed_dim
        )(exercise)
        embeddings.append(emb)

    # Concatenate all input
    emb = keras.layers.concatenate(embeddings)

    lstm = keras.layers.recurrent.LSTM(
        n_lstm_units, return_sequences=True,
    )

    # Feed input to lstm
    lstm_output = lstm(emb)

    logit = keras.layers.TimeDistributed(
                keras.layers.Dense(1)
            )(lstm_output)

    p = keras.layers.TimeDistributed(
                keras.layers.Activation('sigmoid')
            )(logit)

    # We could use the dot product between current exercise and lstm output
    # logit2 = keras.layers.Lambda(
    #     lambda x: K.sum(x[0]*x[1], axis=2)[..., None]
    # )([lstm_output, embeddings[0]])

    inputs = [exercise, last_exercise_x_correct, last_correct]
    outputs = [p]

    m = keras.models.Model(inputs=inputs, outputs=outputs)

    optimizer = keras.optimizers.Adam()

    m.compile(optimizer, 'binary_crossentropy', metrics=['acc', 'mse'])

    return m

fn = '/Users/anton/Downloads/15items_200steps_history.pkl'

with open(fn, 'rb') as f:
    data = pickle.load(f)

# start with one tutor
tutor_data = data[0]
dev_split = int(len(tutor_data)*0.8)

def process(data):
    x = [[] for _ in range(3)]
    y = []

    for seq in data:
        seq = np.array(seq)

        exercise = seq[:, 0]
        correct = seq[:, 1]

        # Use N_EXERCISES as a placeholder for 'first event'
        last_exercise = np.concatenate([[N_EXERCISES], exercise[:-1]])

        # Use 2 as a placeholder for 'first event'
        last_correct = np.concatenate([[2], correct[:-1]])

        # combine correct and exercise to unique integers:
        exercise_x_correct = correct*(N_EXERCISES) + exercise
        last_exercise_x_correct = np.concatenate([[2*N_EXERCISES + 1], exercise_x_correct[:-1]])
        
        x[0].append(exercise)
        x[1].append(last_exercise_x_correct)
        x[2].append(last_correct)

        y.append(correct)

    x = [np.array(xx) for xx in x]
    # Expand last dimension of y:
    y = np.array(y)[..., None]
    return x, y

x, y = process(tutor_data[:dev_split])
x_dev, y_dev = process(tutor_data[dev_split:])

m = model()

m.fit(x, y, validation_data=[x_dev, y_dev], batch_size=32, epochs=10)

print('Majority class:', y.mean(), y_dev.mean())

