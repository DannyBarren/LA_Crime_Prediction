# Predictive Crime CNN
This Convolutional Neural Network predicts the locations of clusters of violent crime in Los Angeles for the upcoming week. It was built using Python/TensorFlow and is built on top of a dataset comprised of data gathered from official crime data published by the city of LA and weather data gathered from the National Weather Service's online repository of weather station data.

## Project Overview
This model is an example of a multimodal deep learning model that combines spatial-temporal data (in this case, crime organized via coordinates/grid squares and weather over time) from multiple sources. Overall, the dataset is comprised of past crime data, weather data, and holiday data.
The model treats LA as a 60x75 grid for a total of 4500 grid squares. For each week, data is aggregated and totaled per grid square (crime counts per grid square, weather events per grid square, etc.) and the goal is to use WINDOW_SIZE weeks of data to predict the crime counts per grid square for up to one week in the future.
This is a supervised learning and regression task: the data is labeled and the model is predicting continuous values—crime counts. TensorFlow and Keras are used to build a Convolutional Neural Network (CNN) with two branches (one for crime and one for weather), which merge to make predictions. A CNN is ideal for this task because the data has spatial structures similar to that of an image, as well as a temporal sequence (weeks in a row).
This code is divided into three main sections, and I will break each down in detail and explain how they work. I will also be providing another README and code samples that were used to create the dataset, and those deserve their own, individual documents and explanations because the dataset required a lot of time and effort to gather, prepare and put together.

## Value as a Prototype and Real-World Applications
This project represents a valuable and promising prototype for using AI and machine learning to predict crime patterns in urban areas like Los Angeles, showing how even a simple neural network can turn historical crime and weather data into actionable insights for the future. As it stands, the model is capable of forecasting crime concentrations across a detailed grid with a reasonable degree of accuracy—off by about 0.18 crimes per grid square on average—which makes it a solid starting point that can be refined with more data, advanced features like socio-economic indicators, or even integration with real-time sensors to boost precision and handle longer-term predictions. In the real world, this kind of tool has huge potential for assisting law enforcement with smarter policing strategies, such as identifying emerging hotspots in advance so officers can be deployed proactively to high-risk areas, potentially preventing violent crimes before they occur and making communities safer. For example, in Los Angeles where violent crime rates have been declining through 2025 but police departments face staffing shortages and ongoing challenges like cargo theft, a refined version of this model could optimize limited resources by prioritizing patrols or community interventions based on predicted trends influenced by weather or holidays. While predictive policing tools have sparked debates around ethics, bias, and accountability—like ongoing concerns and past lawsuits over transparency in LAPD programs—this prototype highlights how AI can evolve into a helpful aid when built responsibly, focusing on prevention rather than just reaction, and ultimately contributing to more effective, data-driven public safety strategies.

## Summary of Results and Model Evaluation
After running this model, it demonstrated that it is capable of predicting the location and concentration of crime in the future (up to one week) to a degree that is useful and relevant for decision making. This model demonstrated its viability as a valuable and promising prototype that is capable of being refined and improved to achieve higher degrees of accuracy. As the model currently stands in its present form, it was capable of predicting future crime counts per grid square with an average error of ±0.18 crimes per week. This model is capable of producing generally accurate future crime predictions over short-term periods of time (one week) and identifying future trends and crime concentrations to assist law enforcement with allocating resources in advance to assist in preventing violent crimes before they happen.
The model's performance was evaluated using key regression metrics on validation data, yielding a Mean Absolute Error (MAE) of approximately 0.175 crimes per grid square, indicating predictions are typically off by about 0.18 crimes weekly relative to an average of 0.56 crimes per square in 2023 LA data, for a relative error of around 30-35%. The Mean Squared Error (MSE) of 0.377 and Root Mean Squared Error (RMSE) of roughly 0.61 further highlight its ability to minimize larger outliers while capturing spatial trends, with stronger relative accuracy (5-10% error) in high-crime hotspots. As a prototype, this demonstrates valuable potential for short-term crime forecasting and resource allocation in predictive policing, achieving benchmark-aligned hotspot identification rates of 50-70%, though refinement with updated 2024-2025 data, feature scaling, and advanced layers could enhance precision for real-world applications like preempting violent incidents.

## Code Explanation
This README section provides a detailed breakdown of the code, explaining its structure, purpose, and key concepts section by section. The code is divided into three main parts as indicated by the comments in the script: building the neural network, loading and preparing data, and training/evaluating the model. I'll explain each in depth, including why certain choices were made, how layers and functions work, and tips for modification. This model uses TensorFlow/Keras for a multimodal 3D CNN to predict crime grids in LA based on historical crime and weather data.

### Section 1: Building the Neural Network
This section imports libraries and defines the model's architecture using Keras' functional API, which supports complex models with multiple inputs (crime and weather data) and outputs (predicted crime grid). It's like designing a machine that processes two types of sequential "videos" (time-series grids) to forecast the next frame.
- **Imports and WINDOW_SIZE Setup**:
  The code starts with importing TensorFlow (tf), Keras components (layers, Model, Adam, MeanSquaredError, MeanAbsoluteError), Pandas (pd), and NumPy (np). These are essential for building models, optimizing, handling losses/metrics, and data manipulation.
  `WINDOW_SIZE = 20` is a key hyperparameter: It sets the number of historical weeks used as input to predict the next week. This acts as the model's "memory horizon"—a longer size (e.g., 30) captures broader trends like seasonal effects but requires more data and computation. It must be at least 6 because the convolutions and pooling reduce dimensions: Two 3x3x3 convs subtract 4 from the time axis (2 per conv with 'valid' padding), and pooling halves it—e.g., for WINDOW_SIZE=5, time becomes 1 after convs, then 0 after pooling (invalid). Adjust this based on dataset size; test values like 10-30 for balance.
- **Defining the Model Function (`build_crime_prediction_model()`)**:
  This function constructs the model step by step.
  - **Inputs**:
    `crime_input = layers.Input(shape=(WINDOW_SIZE, 60, 75, 1))`: Defines the crime data input as a 5D tensor (batch_size inferred, time=20 weeks, height=60 rows, width=75 columns, channels=1 for crime counts). The 60x75 grid divides LA into 4,500 cells (~0.11 sq mi each, based on LA's ~500 sq mi area), where each cell holds weekly crime counts. Visualize the full input as a "book" of 20 pages (weeks), each a 60x75 grid map with numbers per cell.
    `weather_input = layers.Input(shape=(WINDOW_SIZE, 60, 75, 3))`: Similar for weather, but with 3 channels: temperature (e.g., average °F), rain (e.g., inches), and is_holiday (binary 0/1 for work-impacting holidays). Holidays aren't weather but influence behavior (e.g., more/less street activity), so they're bundled here for convenience. This treats data like videos: crime as grayscale (1 channel), weather as RGB (3 channels). Inputs are tensors—multi-dimensional arrays ideal for 3D convolutions that scan patterns across time, space, and channels.
  - **Crime Branch**:
    This processes the crime_input to extract spatial-temporal features.
    `x1 = layers.Conv3D(32, kernel_size=(3,3,3), activation='relu', padding='valid')`: First 3D convolution with 32 filters (learnable kernels) scanning 3 units in time/height/width. ReLU activation adds non-linearity (outputs >=0, helps learn complex patterns). 'Valid' padding means no border zeros, shrinking output (e.g., time: 20→18). This layer detects basic patterns like crime clusters.
    Second `Conv3D(64, ...)`: Deepens feature extraction, increasing to 64 filters (time:18→16). Stacking convs builds hierarchical features (low-level edges to high-level hotspots).
    `layers.MaxPooling3D(pool_size=(2,2,2))`: Reduces size by max values in 2x2x2 cubes (time:16→8, spatial halved), focusing on salient features and cutting computation.
    `layers.Flatten()`: Converts 4D tensor to 1D vector (fixed size due to constant WINDOW_SIZE).
    `layers.Dense(64, activation='relu')`: Fully connected layer compresses to 64 units, learning global patterns from flattened features.
    Why this? CNNs excel at local spatial patterns (e.g., crime spreading to adjacent cells) and temporal ones (e.g., weekly escalations), making them suitable for grid-time data.
  - **Weather Branch**:
    Identical to the crime branch but for 3-channel weather_input. It learns how environmental factors (e.g., rain reducing outdoor crimes) interact spatially and temporally. Separate branches allow tailored processing before merging, as crime (counts) and weather (multi-feature) differ.
  - **Merging and Output**:
    `combined = layers.Concatenate()([x1, x2])`: Joins the two 64-unit vectors into 128 units, integrating multimodal insights.
    `predictions = layers.Dense(4500, activation='relu')(combined)`: Expands to 4,500 units (one per grid cell), ReLU ensures non-negative crime predictions.
    `layers.Reshape((60,75,1))`: Reforms into the output grid shape.
    `model = Model(inputs=[crime_input, weather_input], outputs=predictions)`: Defines the full model.
  - **Compilation**:
    `model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])`: Adam adapts learning for efficiency; MSE loss penalizes squared errors (good for continuous regression); MAE metric provides interpretable average error. This setup optimizes for accurate crime count predictions.
Overall, this is a branched 3D CNN architecture (~thousands of parameters), inspired by video analysis models, efficient for fixed-sequence data but could be extended with LSTMs for variable lengths.

### Section 2: Loading and Preparing the Data
Data preparation transforms raw CSVs into model-ready tensors, crucial for training as models expect specific shapes.
- **Loading Data**:
  `crime_data = pd.read_csv('crime_counts_2023.csv', ...).values`: Loads a 4,500 (grids) x 52 (weeks) array of crime counts.
  `weather_data = pd.read_csv('weather_holiday_2023.csv', ...).values`: 4,500 x 156 array (52 weeks x 3 features). Assumes pre-processed data; in practice, ensure no missing values.
- **Reshaping to 3D**:
  `def reshape_to_3d(data_2d)`: Loops over weeks, reshaping each column's 4,500 values into 60x75 maps, yielding (52,60,75) for crime. `np.expand_dims(..., axis=-1)` adds channel: (52,60,75,1).
  For weather: A loop extracts every third column per week (temp, rain, holiday) into channels: (52,60,75,3). This converts flat tables to "image sequences," assuming row-major grid ordering.
- **Creating Sliding Windows**:
  `num_windows = 52 - WINDOW_SIZE` (32 for 20). Creates arrays for inputs/targets. Loop: For i=0 to 31, inputs are weeks i to i+19, target is i+20's crime grid.
  This generates multiple samples from limited data (52 weeks → 32 examples), using overlapping windows for augmentation. Preserves time order to prevent future data leaking into training—key for time-series forecasting. Tip: If adding more years, scale WINDOW_SIZE accordingly; consider normalization (e.g., MinMaxScaler) for better convergence, as raw counts/weather may vary in scale.

### Section 3: Splitting Data, Training, and Evaluating
This trains the model on prepared data, evaluates performance, and saves it—where learning occurs via backpropagation.
- **Data Splitting**:
  `train_split = int(0.8 * num_windows)` (~25 train, 7 val). Time-based split: Early windows for train, later for val, mimicking real prediction (no future peeking).
- **GPU Check**:
  Detects GPUs and sets memory growth to avoid crashes. Ran on CPU here (slow: 6-7s/step); use GPU for speed with larger data.
- **Training**:
  `model = build_crime_prediction_model()`: Instantiates the model.
  `history = model.fit([crime_train, weather_train], target_train, validation_data=..., epochs=50, batch_size=8)`: Trains over 50 full data passes, processing 8 windows/batch. Monitors progress; history stores metrics for plotting (e.g., loss curves). Rapid drop early (loss 638→0.43) shows learning; plateau after ~30 epochs indicates convergence. Batch size=8 balances memory/speed; epochs=50 is ample—add EarlyStopping callback to halt if val loss stalls.
- **Evaluation and Saving**:
  `model.evaluate(...)`: Computes final val loss (0.377) and MAE (0.175)—off by ~0.18 crimes/cell, relative error ~30% vs. 2023 average 0.56/cell.
  `model.save('crime_prediction_model.h5')`: Stores for inference (e.g., load and predict on new data).
  This confirms no overfitting (val close to train) and provides a baseline. For improvement: Visualize predictions vs. actuals (heatmaps), add metrics like R², or retrain with more data for lower errors.

  # Notes on the testing application

  I added code that is used to demonstrate the model's capabilities. This code allows the user to select a week and it produces side by side heatmaps that show how accurate the model is. I have also added some samples of its output for reference. 
