import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model('crime_prediction_model.h5')
print("Model loaded successfully!")

# Define reshaping functions (from your original code)
def reshape_to_3d(data_2d):
    data_3d = np.zeros((52, 60, 75))
    for t in range(52):
        grid_data = data_2d[:, t]
        data_3d[t, :, :] = grid_data.reshape(60, 75)
    return data_3d

# Load 2023 data (adjust paths if needed)
crime_data = pd.read_csv('crime_counts_2023.csv', index_col=0).values  # 4500 x 52
weather_data = pd.read_csv('weather_holiday_2023.csv', index_col=0).values  # 4500 x 156

# Reshape crime: (52, 60, 75, 1)
crime_3d = reshape_to_3d(crime_data)
crime_3d = np.expand_dims(crime_3d, axis=-1)

# Reshape weather: (52, 60, 75, 3)
weather_3d = np.zeros((52, 60, 75, 3))
for t in range(52):
    weather_3d[t, :, :, 0] = weather_data[:, t * 3 + 0].reshape(60, 75)  # Temp
    weather_3d[t, :, :, 1] = weather_data[:, t * 3 + 1].reshape(60, 75)  # Rain
    weather_3d[t, :, :, 2] = weather_data[:, t * 3 + 2].reshape(60, 75)  # Holiday

WINDOW_SIZE = 20

# User input for week selection (valid weeks: 21 to 52)
selected_week = int(input("Enter a week number to predict (21 to 52): "))
if selected_week < 21 or selected_week > 52:
    print("Invalid week selected. Please choose between 21 and 52.")
else:
    # Prepare input
    window_idx = selected_week - WINDOW_SIZE - 1  # Start of 20-week window
    crime_input = crime_3d[window_idx:window_idx + WINDOW_SIZE][np.newaxis, ...]  # (1, 20, 60, 75, 1)
    weather_input = weather_3d[window_idx:window_idx + WINDOW_SIZE][np.newaxis, ...]  # (1, 20, 60, 75, 3)
    actual = crime_3d[window_idx + WINDOW_SIZE, :, :, 0]  # Actual for selected week (60, 75)

    # Make prediction
    pred = model.predict([crime_input, weather_input])[0, :, :, 0]  # (60, 75)

    # Print sample output (e.g., first 5x5 grid corner)
    print("Sample Predicted Crime Counts (top-left 5x5 grid):")
    print(pred[:5, :5])

    # Calculate and print metrics
    mae = np.mean(np.abs(pred - actual))
    print(f"\nMAE for this prediction: {mae:.3f} crimes per grid square")

    # Visualize side-by-side heatmaps
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    # Predicted heatmap
    im1 = axs[0].imshow(pred, cmap='hot', interpolation='nearest')
    axs[0].set_title(f'Predicted Crime Heatmap for Week {selected_week} (2023)')
    axs[0].set_xlabel('Grid Columns')
    axs[0].set_ylabel('Grid Rows')
    fig.colorbar(im1, ax=axs[0], label='Predicted Crime Counts')

    # Actual heatmap
    im2 = axs[1].imshow(actual, cmap='hot', interpolation='nearest')
    axs[1].set_title(f'Actual Crime Heatmap for Week {selected_week} (2023)')
    axs[1].set_xlabel('Grid Columns')
    axs[1].set_ylabel('Grid Rows')
    fig.colorbar(im2, ax=axs[1], label='Actual Crime Counts')

    plt.show()

    # Optional: Save prediction
    np.save(f'predicted_week_{selected_week}_2023.npy', pred)
    print("Prediction complete! Check the heatmaps or saved file.")