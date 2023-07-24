import tensorflow as tf
from PythonFiles.DataProcessing import get_data_normal, get_data_hsv, get_data_bw


def get_model():
    # creates model
    new_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 150, 3)),
        tf.keras.layers.MaxPooling2D(3, 3),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 150, 3)),
        tf.keras.layers.MaxPooling2D(3, 3),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(300, 150, 3)),
        tf.keras.layers.MaxPooling2D(3, 3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(16, activation=tf.nn.softmax)])

    # compiles model
    new_model.compile(optimizer=tf.optimizers.Adam(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    new_model.summary()

    # returns model
    return new_model


def run_model(data_id: int):
    # get training and test data
    if data_id == 2:
        training_images, training_labels, test_images, test_labels = get_data_hsv()
    elif data_id == 3:
        training_images, training_labels, test_images, test_labels = get_data_bw()
    else:
        training_images, training_labels, test_images, test_labels = get_data_normal()

    print(f"Training-Data-Size: {len(training_images)} \nTest-Data-Size: {len(test_images)}")

    # turn values from 0-255 in images to a value from 0.0 - 1.0
    # which is easier for the neural network to understand
    training_images = training_images / 255.0
    test_images = test_images / 255.0

    # gets model
    model = get_model()

    # fits the data to the training data
    history = model.fit(training_images, training_labels, epochs=9)

    # evaluates the trained model using the test data
    print("Evaluation:")
    model.evaluate(test_images, test_labels)
