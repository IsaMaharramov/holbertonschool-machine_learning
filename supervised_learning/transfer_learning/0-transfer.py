#!/usr/bin/env python3
"""
Transfer Knowledge module for CIFAR-10 classification
"""
from tensorflow import keras as K


def preprocess_data(X, Y):
    """
    Pre-processes the data for the model

    Args:
        X: numpy.ndarray of shape (m, 32, 32, 3) containing CIFAR 10 data
        Y: numpy.ndarray of shape (m,) containing CIFAR 10 labels

    Returns:
        X_p: numpy.ndarray containing preprocessed X
        Y_p: numpy.ndarray containing preprocessed Y
    """
    X_p = K.applications.densenet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    # Load dataset
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # Input layer
    inputs = K.Input(shape=(32, 32, 3))

    # Lambda layer to resize images to 224x224
    # Note: `__import__('tensorflow')` is used to gracefully avoid 'NameError'
    # when the model is later loaded in 0-main.py (which only imports Keras).
    resize_layer = K.layers.Lambda(
        lambda x: __import__('tensorflow').image.resize(x, (224, 224))
    )(inputs)

    # Load DenseNet121 base model initialized with ImageNet weights
    base_model = K.applications.DenseNet121(
        include_top=False,
        weights='imagenet',
        input_tensor=resize_layer
    )
    # Freeze the base model to prevent weights from updating during training
    base_model.trainable = False

    # Add Global Average Pooling to flatten the spatial dimensions
    gap = K.layers.GlobalAveragePooling2D()(base_model.output)

    # Create the Feature Extractor model
    feature_extractor = K.Model(inputs=inputs, outputs=gap)

    # Hint 3: Precompute the output of the frozen layers ONCE.
    # This dramatically cuts down training time since images are only passed
    # through the heavy DenseNet base a single time.
    print("Pre-computing training features...")
    train_features = feature_extractor.predict(
        X_train, batch_size=128, verbose=1
    )
    print("Pre-computing validation features...")
    test_features = feature_extractor.predict(
        X_test, batch_size=128, verbose=1
    )

    # Build an independently trainable Classifier Head
    head_input = K.Input(shape=(1024,))  # Output dimension of DenseNet121 GAP
    y = K.layers.Dense(512, activation='relu')(head_input)
    y = K.layers.Dropout(0.2)(y)
    y = K.layers.Dense(256, activation='relu')(y)
    y = K.layers.Dropout(0.2)(y)
    head_output = K.layers.Dense(10, activation='softmax')(y)

    classifier = K.Model(inputs=head_input, outputs=head_output)
    classifier.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the Classifier strictly on the precomputed feature vectors
    print("Training classifier...")
    classifier.fit(
        train_features, Y_train,
        validation_data=(test_features, Y_test),
        epochs=12,
        batch_size=128,
        verbose=1
    )

    # Re-assemble the Complete Architecture (Feature Extractor + Classifier)
    full_output = classifier(feature_extractor.output)
    full_model = K.Model(inputs=inputs, outputs=full_output)

    # Compile the final assembled model to meet project requirements
    full_model.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Save to current working directory
    full_model.save('cifar10.h5')
    print("Model successfully saved as cifar10.h5")
