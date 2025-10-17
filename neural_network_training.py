import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
import pandas as pd
import numpy as np

"""

This first section of code builds the neural network 

"""
# Build the model
def build_crime_prediction_model():
    # Input 1: Crime Matrix (batch_size, 52, 60, 75, 1)
    crime_input = layers.Input(shape=(52, 60, 75, 1), name='crime_input')
   
    # Crime Branch
    x1 = layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='valid')(crime_input)  # (50, 58, 73, 32)
    x1 = layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='valid')(x1)  # (48, 56, 71, 64)
    x1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(x1)  # (24, 28, 35, 64)
    x1 = layers.Flatten()(x1)  # ~1.9M features
    x1 = layers.Dense(64, activation='relu')(x1)  # 64 units
   
    # Input 2: Weather & Holiday Matrix (batch_size, 52, 60, 75, 3)
    weather_input = layers.Input(shape=(52, 60, 75, 3), name='weather_input')
   
    # Weather & Holiday Branch
    x2 = layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='valid')(weather_input)  # (50, 58, 73, 32)
    x2 = layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='valid')(x2)  # (48, 56, 71, 64)
    x2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(x2)  # (24, 28, 35, 64)
    x2 = layers.Flatten()(x2)  # ~1.9M features
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

"""

This section of code loads and prepares the data for importing into the neural network

"""

# Load and prepare datasets
crime_data = pd.read_csv('crime_counts_2023.csv', index_col=0).values  # 4500 x 52
weather_data = pd.read_csv('weather_holiday_2023.csv', index_col=0).values  # 4500 x 156 (52 weeks x 3 features)

# Reshape to (52, 60, 75) with padding/interpolation
def reshape_to_3d(data_2d):
    # Current data is 4500 grids x 52 weeks
    # Target shape is 52 time steps x 60 x 75 grids
    data_3d = np.zeros((52, 60, 75))
    for t in range(52):
        # Distribute 4500 grids across 60x75 = 4500 cells
        grid_data = data_2d[:, t]  # Crime counts for week t
        data_3d[t, :, :] = grid_data.reshape(60, 75)  # Perfect match since 60*75 = 4500
    return data_3d

# Reshape crime data (1 channel)
crime_3d = reshape_to_3d(crime_data)  # Shape: (52, 60, 75)
crime_3d = np.expand_dims(crime_3d, axis=-1)  # Add channel dimension: (52, 60, 75, 1)
crime_3d = np.expand_dims(crime_3d, axis=0)  # Add batch dimension: (1, 52, 60, 75, 1)

# Reshape weather data (3 channels: temp, rain, is_holiday)
weather_3d = np.zeros((52, 60, 75, 3))
for t in range(52):
    temp_col = f'temp_week_{t+1}'
    rain_col = f'rain_week_{t+1}'
    holiday_col = f'is_holiday_week_{t+1}'
    temp_data = weather_data[:, t * 3 + 0]  # Extract temp for week t
    rain_data = weather_data[:, t * 3 + 1]  # Extract rain for week t
    holiday_data = weather_data[:, t * 3 + 2]  # Extract is_holiday for week t
    weather_3d[t, :, :, 0] = temp_data.reshape(60, 75)  # Temp channel
    weather_3d[t, :, :, 1] = rain_data.reshape(60, 75)  # Rain channel
    weather_3d[t, :, :, 2] = holiday_data.reshape(60, 75)  # Holiday channel
weather_3d = np.expand_dims(weather_3d, axis=0)  # Add batch dimension: (1, 52, 60, 75, 3)

# Prepare target data (next week's crime counts)
target_3d = np.roll(crime_3d, shift=-1, axis=1)  # Shift by 1 week
target_3d[:, -1, :, :] = 0  # Last week has no next week, set to 0 for simplicity

"""

This code splits the data into training and testing sets and trains the model, and then evaluates its performance

"""

# Split into training and validation (using time-based split, e.g., first 80% weeks)
train_split = int(0.8 * 52)  # 41 weeks for training
crime_train = crime_3d[:, :train_split, :, :, :]
weather_train = weather_3d[:, :train_split, :, :, :]
target_train = target_3d[:, :train_split, :, :, :]

crime_val = crime_3d[:, train_split:, :, :, :]
weather_val = weather_3d[:, train_split:, :, :, :]
target_val = target_3d[:, train_split:, :, :, :]

# Build and train the model
model = build_crime_prediction_model()

# Train the model
history = model.fit(
    [crime_train, weather_train],  # Input data
    target_train,  # Target data
    validation_data=([crime_val, weather_val], target_val),  # Validation data
    epochs=50,
    batch_size=1,  # Single sample for now, adjust based on memory
    verbose=1
)

# Evaluate the model
loss, mae = model.evaluate([crime_val, weather_val], target_val, verbose=0)
print(f"\nValidation Loss: {loss}, Validation MAE: {mae}")

# Save the model
model.save('crime_prediction_model.h5')
print("Model saved to 'crime_prediction_model.h5'")