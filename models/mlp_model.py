from tensorflow.keras import layers, models

def build_mlp_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(input_shape),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')  # Softmax pour la classification multi-classes
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
