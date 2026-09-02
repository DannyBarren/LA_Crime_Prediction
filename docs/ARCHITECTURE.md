# Architecture

Two Python files. One trains and saves the model, one loads the saved model and draws heatmaps. Nothing else runs.

## File by file

### `neural_network_training.py` — training

Three parts, in order:

1. **Model definition** (`build_crime_prediction_model`). Functional Keras model, two inputs, one output. Compiled with Adam at learning rate 0.001, MSE loss, MAE metric.
2. **Data loading and reshaping.** Reads two CSVs from the working directory, reshapes flat 4,500-row tables into weekly 60×75 grids, then builds sliding windows.
3. **Split, fit, evaluate, save.** Time-based 80/20 split, `fit` for 50 epochs at batch size 8, `evaluate` on the validation windows, then `model.save('crime_prediction_model.h5')`.

The script also probes for a GPU with `tf.config.list_physical_devices('GPU')` and sets memory growth. It runs on CPU without it, just slower.

### `model_test_application.py` — demo

Loads `crime_prediction_model.h5`, re-reads the same two CSVs, and reapplies the same reshaping logic (it is duplicated, not imported). It prompts for a week number between 21 and 52, builds the single 20-week window ending just before that week, predicts, prints the top-left 5×5 corner of the prediction and the MAE for that week, and shows predicted and actual heatmaps side by side with `matplotlib`. It also writes `predicted_week_<N>_2023.npy`.

Week 21 is the first predictable week: you need 20 weeks of history before it. Week 52 is the last week in the data.

## Tensor shapes

| Stage | Shape |
| --- | --- |
| `crime_counts_2023.csv` | 4500 × 52 (cells × weeks) |
| `weather_holiday_2023.csv` | 4500 × 156 (cells × 52 weeks × 3 features) |
| Crime after reshape | (52, 60, 75, 1) |
| Weather after reshape | (52, 60, 75, 3) — temp, rain, is_holiday |
| Crime model input | (batch, 20, 60, 75, 1) |
| Weather model input | (batch, 20, 60, 75, 3) |
| Target | (batch, 60, 75, 1) |

60 × 75 = 4,500, which is exactly the number of rows in each CSV, so each week's column reshapes into the grid with no padding or interpolation. Weather columns are interleaved per week: column `t*3+0` is temperature, `t*3+1` is rain, `t*3+2` is the holiday flag.

Each branch is `Conv3D(32, 3×3×3, valid)` → `Conv3D(64, 3×3×3, valid)` → `MaxPooling3D(2×2×2)` → `Flatten` → `Dense(64)`. Spatially: 60×75 → 58×73 → 56×71 → 28×35. The two 64-unit vectors concatenate into 128, then `Dense(4500, relu)` and `Reshape((60, 75, 1))`. ReLU on the output keeps predicted counts non-negative.

## Why `WINDOW_SIZE` must be ≥ 6

The time axis shrinks the same way the spatial axes do. Each `Conv3D` with a kernel of 3 and `padding='valid'` removes 2 frames, so two of them take `W` to `W - 4`. Then `MaxPooling3D(pool_size=(2,2,2))` halves it to `floor((W - 4) / 2)`.

That result has to be at least 1. So `W - 4 >= 2`, which means `W >= 6`. At `W = 5` the time axis is 1 after the convolutions and 0 after pooling, and the graph fails to build. At `W = 20` the time axis goes 20 → 18 → 16 → 8.

`WINDOW_SIZE` is a module-level constant in the training script and a separate literal in the demo script. If you change one, change both, and retrain — the `Flatten` output size depends on it, so an old `.h5` will not load into a new window size.

## Time-based split

`num_windows = 52 - WINDOW_SIZE` = 32 windows. Window `i` uses weeks `i` through `i+19` as input and week `i+20` as the target.

`train_split = int(0.8 * 32)` = 25. Windows 0–24 train, windows 25–31 validate. The split is by index, not shuffled, so every validation target falls later in the year than every training target. No future week is ever visible during training. The cost is a small validation set — 7 windows — so the reported metrics are noisy.

## What is not in git

- `crime_counts_2023.csv` and `weather_holiday_2023.csv`. Both scripts read them from the working directory and will fail immediately without them.
- `crime_prediction_model.h5`. Produced by the training script, required by the demo script.
- `predicted_week_<N>_2023.npy` files written by the demo script.
- The aggregation code that turned raw LAPD incident records and NWS station observations into the two grid CSVs. It lives outside this repo.

`.gitignore` covers `*.h5` and `*.npy` so training and demo artifacts do not get committed by accident.
