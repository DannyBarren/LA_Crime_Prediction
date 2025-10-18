import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
import pandas as pd
import numpy as np

"""
This first section of code builds the neural network and its architecture
"""
#define window size for fixed-length historical sequences
WINDOW_SIZE = 20  #number of past weeks to use for predicting the next week; adjustable, but must be >=6 for the conv/pool layers

#build the model with fixed window size
def build_crime_prediction_model():
    #input 1: Crime Matrix (batch_size, WINDOW_SIZE, 60, 75, 1)
    crime_input = layers.Input(shape=(WINDOW_SIZE, 60, 75, 1), name='crime_input')
  
    #crime branch
    x1 = layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='valid')(crime_input)  #(WINDOW_SIZE-2, 58, 73, 32)
    x1 = layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='valid')(x1)  #(WINDOW_SIZE-4, 56, 71, 64)
    x1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(x1)  #(floor((WINDOW_SIZE-4)/2), 28, 35, 64)
    x1 = layers.Flatten()(x1)  #fixed size since WINDOW_SIZE is fixed
    x1 = layers.Dense(64, activation='relu')(x1)  #64 units
  
    #input 2: Weather & Holiday Matrix (batch_size, WINDOW_SIZE, 60, 75, 3)
    weather_input = layers.Input(shape=(WINDOW_SIZE, 60, 75, 3), name='weather_input')
  
    #weather & holiday branch
    x2 = layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='valid')(weather_input)  #(WINDOW_SIZE-2, 58, 73, 32)
    x2 = layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='valid')(x2)  #(WINDOW_SIZE-4, 56, 71, 64)
    x2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(x2)  #(floor((WINDOW_SIZE-4)/2), 28, 35, 64)
    x2 = layers.Flatten()(x2)  #fixed size
    x2 = layers.Dense(64, activation='relu')(x2)  #64 units
  
    #concatenation
    combined = layers.Concatenate()([x1, x2])  # 128 units
    #output path: Dense to flat grid predictions, then reshape to spatial map
    predictions = layers.Dense(4500, activation='relu')(combined)  # 4500 units (60*75 grids)
    predictions = layers.Reshape((60, 75, 1))(predictions)  # (60, 75, 1)
    #model
    model = Model(inputs=[crime_input, weather_input], outputs=predictions)
  
    #compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=MeanSquaredError(),
        metrics=[MeanAbsoluteError(name='mae')]
    )
    return model

"""
This section of code loads and prepares the data for importing into the neural network

***note*** there are other files of code that were written and executed prior to this stage that filtered, transformed and prepared both the weather and crime datasets for use in this model

these other documents will be shared elsewhere 
"""
# Load and prepare datasets
crime_data = pd.read_csv('crime_counts_2023.csv', index_col=0).values  # 4500 x 52
weather_data = pd.read_csv('weather_holiday_2023.csv', index_col=0).values  # 4500 x 156 (52 weeks x 3 features)

#reshape to (52, 60, 75) with padding/interpolation
def reshape_to_3d(data_2d):
    #current data is 4500 grids x 52 weeks
    #target shape is 52 time steps x 60 x 75 grids
    data_3d = np.zeros((52, 60, 75))
    for t in range(52):
        #distribute 4500 grids across 60x75 = 4500 cells
        grid_data = data_2d[:, t]  #crime counts for week t
        data_3d[t, :, :] = grid_data.reshape(60, 75)  #perfect match since 60*75 = 4500
    return data_3d

#reshape crime data (1 channel)
crime_3d = reshape_to_3d(crime_data)  #shape: (52, 60, 75)
crime_3d = np.expand_dims(crime_3d, axis=-1)  #add channel dimension: (52, 60, 75, 1)

#reshape weather data (3 channels: temp, rain, is_holiday)
weather_3d = np.zeros((52, 60, 75, 3))
for t in range(52):
    temp_data = weather_data[:, t * 3 + 0]  #extract temp for week t
    rain_data = weather_data[:, t * 3 + 1]  #extract rain for week t
    holiday_data = weather_data[:, t * 3 + 2]  #extract is_holiday for week t
    weather_3d[t, :, :, 0] = temp_data.reshape(60, 75)  #temp channel
    weather_3d[t, :, :, 1] = rain_data.reshape(60, 75)  #rain channel
    weather_3d[t, :, :, 2] = holiday_data.reshape(60, 75)  #holiday channel

#create sliding window datasets for multiple training examples
num_windows = 52 - WINDOW_SIZE
crime_inputs = np.zeros((num_windows, WINDOW_SIZE, 60, 75, 1))
weather_inputs = np.zeros((num_windows, WINDOW_SIZE, 60, 75, 3))
targets = np.zeros((num_windows, 60, 75, 1))

for i in range(num_windows):
    crime_inputs[i] = crime_3d[i:i+WINDOW_SIZE, :, :, :]
    weather_inputs[i] = weather_3d[i:i+WINDOW_SIZE, :, :, :]
    targets[i] = crime_3d[i+WINDOW_SIZE, :, :, :]

"""
This code splits the data into training and testing sets and trains the model, and then evaluates its performance
"""
#split into training and validation (time-based, ~80% for train)
train_split = int(0.8 * num_windows)
crime_train = crime_inputs[:train_split]
weather_train = weather_inputs[:train_split]
target_train = targets[:train_split]

crime_val = crime_inputs[train_split:]
weather_val = weather_inputs[train_split:]
target_val = targets[train_split:]

#GPU check and setup
gpus = tf.config.list_physical_devices('GPU')
print(f"Numberm  of GPUs Available: {len(gpus)}")
if gpus:
    print(f"Using GPU: {gpus[0].name}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, False)
    else:
        print("Using GPU")
        
# ***NOTE*** the above section of code for setting up GPU use can be "#'d" out, I was able to run this model on only my GPU without issue, but it will run slower and take more time

#build and train the model
model = build_crime_prediction_model()
#train the model
history = model.fit(
    [crime_train, weather_train],  #input data
    target_train,  #target data
    validation_data=([crime_val, weather_val], target_val),  #validation data
    epochs=50,
    batch_size=8,  #can use larger batch size now with multiple samples
    verbose=1
)
#evaluate the model
loss, mae = model.evaluate([crime_val, weather_val], target_val, verbose=0)
print(f"\nValidation Loss: {loss}, Validation MAE: {mae}")
#save the model
model.save('crime_prediction_model.h5')
print("Model saved to 'crime_prediction_model.h5'")
