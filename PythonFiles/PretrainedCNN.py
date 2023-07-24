from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import Model
from tensorflow.keras import layers
from PythonFiles.DataProcessing import get_data_normal


def get_model():
    # Get the pretrained model and select an output layer
    pre_trained_model = VGG19(
        input_shape=(300, 150, 3), include_top=False, weights="imagenet"
    )

    # Freeze the layers
    for layer in pre_trained_model.layers:
        layer.trainable = False

    # Get network summary
    pre_trained_model.summary()

    # Choose layer from the network until which it will be used for the training of the base model
    last_layer = pre_trained_model.get_layer("block5_pool")
    last_output = last_layer.output

    # Model definition
    # creates model
    x = layers.Flatten()(last_output)  # last layer of VGG19
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(16, activation="softmax")(x)

    model = Model(pre_trained_model.input, x)
    model.summary()

    # compiles model
    model.compile(
        optimizer="Adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def run_model():
    # get training and test data
    training_images, training_labels, test_images, test_labels = get_data_normal()
    print(
        f"Training-Data-Size: {len(training_images)} \nTest-Data-Size: {len(test_images)}"
    )

    # turn values from 0-255 in images to a value from 0.0 - 1.0
    # which is easier for the neural network to understand
    training_images = training_images / 255.0
    test_images = test_images / 255.0

    # gets model
    model = get_model()

    # fits the data to the training data
    history = model.fit(training_images, training_labels, epochs=5)

    # evaluates the trained model using the test data
    print("Evaluation:")
    model.evaluate(test_images, test_labels)
