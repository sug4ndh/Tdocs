import pickle

import numpy as np
import keras
import keras.backend as K

N_EXERCISES = 30

def state_model(embed_dim=20, n_lstm_units=20, n_exercises=30):
    n_categories = (n_exercises)*(2) + 1

    exercise = keras.layers.Input((None, ))
    last_exercise_x_correct = keras.layers.Input((None, ))
    last_correct = keras.layers.Input((None, ))

    initial_state_1 = keras.layers.Input((n_lstm_units, ))
    initial_state_2 = keras.layers.Input((n_lstm_units, ))

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
        n_lstm_units, return_sequences=True, return_state=True
    )

    # Feed input to lstm
    lstm_output, s1, s2 = lstm(emb, initial_state=[initial_state_1, initial_state_2])

    logit = keras.layers.TimeDistributed(
                keras.layers.Dense(1)
            )(lstm_output)

    p = keras.layers.TimeDistributed(
                keras.layers.Activation('sigmoid')
            )(logit)

    inputs = [exercise, last_exercise_x_correct, last_correct, initial_state_1, initial_state_2]
    outputs = [p, s1, s2]

    m = keras.models.Model(inputs=inputs, outputs=outputs)

    optimizer = keras.optimizers.Adam()

    m.compile(optimizer, 'binary_crossentropy', metrics=['acc', 'mse'])

    return m


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

    # We could use the dot product between current exercise and lstm output
    # logit2 = keras.layers.Lambda(
    #     lambda x: K.sum(x[0]*x[1], axis=2)[..., None]
    # )([lstm_output, exercise])

    # logit = keras.layers.add([logit, logit2])

    p = keras.layers.TimeDistributed(
                keras.layers.Activation('sigmoid')
            )(logit)

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


path = '/Users/anton/Downloads/0001-0.70.h5'
old_model = model()
old_model.load_weights(path)

model = state_model()

i = 0
for l1 in old_model.layers:
    if str(type(l1)) ==  "<class 'keras.engine.topology.InputLayer'>":
        continue

    while str(type(model.layers[i])) ==  "<class 'keras.engine.topology.InputLayer'>":
        i += 1

    model.layers[i].set_weights(l1.get_weights())
    i += 1

batch_size = 1
s1 = np.zeros([batch_size, 20])
s2 = np.zeros([batch_size, 20])

x1 = [xx[:1] for xx in x]
x2 = [xx[:1] for xx in x] + [s1, s2]

print(old_model.predict(x1))
print(model.predict(x2))

