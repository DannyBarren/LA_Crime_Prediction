"""
This version was written for Google Colab and was also changed to allow variable input to avoid errors. 
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

# Build the model with dynamic time dimension
def build_crime_prediction_model():
    # Input 1: Crime Matrix (batch_size, None, 60, 75, 1) - dynamic time steps
    crime_input = layers.Input(shape=(None, 60, 75, 1), name='crime_input')

    # Crime Branch
    x1 = layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='valid')(crime_input)  # (time-2, 58, 73, 32)
    x1 = layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='valid')(x1)  # (time-4, 56, 71, 64)
    x1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(x1)  # (time-4/2, 28, 35, 64)
    x1 = layers.Flatten()(x1)  # Features based on dynamic time
    x1 = layers.Dense(64, activation='relu')(x1)  # 64 units

    # Input 2: Weather & Holiday Matrix (batch_size, None, 60, 75, 3) - dynamic time steps
    weather_input = layers.Input(shape=(None, 60, 75, 3), name='weather_input')

    # Weather & Holiday Branch
    x2 = layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='valid')(weather_input)  # (time-2, 58, 73, 32)
    x2 = layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='valid')(x2)  # (time-4, 56, 71, 64)
    x2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(x2)  # (time-4/2, 28, 35, 64)
    x2 = layers.Flatten()(x2)  # Features based on dynamic time
    x2 = layers.Dense(64, activation='relu')(x2)  # 64 units

    # Concatenation
    combined = layers.Concatenate()([x1, x2])  # 128 units
    # Output path: Dense to flat grid predictions, then reshape to spatial map
    predictions = layers.Dense(4500, activation='relu')(combined)  # 4500 units (60*75 grids)
    predictions = layers.Reshape((60, 75, 1))(predictions)  # (60, 75, 1)
    # Model
    model = Model(inputs=[crime_input, weather_input], outputs=predictions)

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=MeanSquaredError(),
        metrics=[MeanAbsoluteError(name='mae')]
    )
    return model

# Load and prepare datasets (assuming Drive mounted)
from google.colab import drive
drive.mount('/content/drive')

crime_data = pd.read_csv('/content/drive/MyDrive/Crime_Neural_Net/crime_counts_2023.csv', index_col=0).values  # 4500 x 52
weather_data = pd.read_csv('/content/drive/MyDrive/Crime_Neural_Net/weather_holiday_2023.csv', index_col=0).values  # 4500 x 156

def reshape_to_3d(data_2d):
    data_3d = np.zeros((52, 60, 75))
    for t in range(52):
        grid_data = data_2d[:, t]
        data_3d[t, :, :] = grid_data.reshape(60, 75)
    return data_3d

crime_3d = reshape_to_3d(crime_data)
crime_3d = np.expand_dims(crime_3d, axis=-1)
crime_3d = np.expand_dims(crime_3d, axis=0)

weather_3d = np.zeros((52, 60, 75, 3))
for t in range(52):
    temp_data = weather_data[:, t * 3 + 0]
    rain_data = weather_data[:, t * 3 + 1]
    holiday_data = weather_data[:, t * 3 + 2]
    weather_3d[t, :, :, 0] = temp_data.reshape(60, 75)
    weather_3d[t, :, :, 1] = rain_data.reshape(60, 75)
    weather_3d[t, :, :, 2] = holiday_data.reshape(60, 75)
weather_3d = np.expand_dims(weather_3d, axis=0)

# Prepare target data (next week's crime counts)
target_3d = np.roll(crime_3d, shift=-1, axis=1)
target_3d[:, -1, :, :] = 0

# Use full 52 weeks, validate on last few weeks separately
crime_train = crime_3d  # Full 52 weeks
weather_train = weather_3d  # Full 52 weeks
target_train = target_3d[:, :-1, :, :, :]  # All but last week as target (51 weeks)

crime_val = crime_3d[:, -1:, :, :, :]  # Last week as validation input
weather_val = weather_3d[:, -1:, :, :, :]  # Last week as validation input
target_val = target_3d[:, -1:, :, :, :]  # Last week's next (set to 0)

# Build and train the model
model = build_crime_prediction_model()

# Add early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    [crime_train, weather_train],
    target_train,
    validation_data=([crime_val, weather_val], target_val),
    epochs=50,
    batch_size=1,
    verbose=1,
    callbacks=[early_stopping]
)

# Evaluate the model
loss, mae = model.evaluate([crime_val, weather_val], target_val, verbose=0)
print(f"\nValidation Loss: {loss}, Validation MAE: {mae}")

# Save the model to Google Drive
model.save('/content/drive/MyDrive/Crime_Neural_Net/crime_prediction_model.h5')
print("Model saved to Google Drive!")