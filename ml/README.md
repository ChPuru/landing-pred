# Project 2: The Professional-Grade Ensemble Predictor

This is our most advanced and intelligent landing prediction system. It represents the final evolution of our work, combining the best of physics, data science, and multiple AI models to achieve the highest possible accuracy and robustness.

The core idea is that no single prediction method is perfect. This system acts like a "board of expert advisors" who all analyze the flight data simultaneously, with a final "CEO" model that intelligently fuses their wisdom into a single, high-confidence prediction.

## Key Features & Innovations

- **Advanced Physics Engine:** Utilizes an iterative, second-by-second simulation with a layered wind model and dynamic air density calculations to provide a highly accurate physics-based baseline.

- **Diverse Machine Learning Ensemble:** Employs two distinct and powerful ML models:
    1. **XGBoost (Extreme Gradient Boosting):** A state-of-the-art model that excels at analyzing a snapshot of the current flight data with high accuracy.
    2. **LSTM (Long Short-Term Memory) Neural Network:** A deep learning model that excels at understanding patterns, trends, and momentum over the last 30 seconds of flight.

- **Stacked Generalization (The "CEO" Model):** The predictions from all three "experts" (Physics, XGBoost, LSTM) are fed into a final XGBoost "meta-model." This model has been trained to learn the complex, non-linear rules for how to best combine the predictions under different flight conditions.

- **Professional Data Pipeline:** The entire system is built on a robust data pipeline that prevents common ML errors. This includes:
  - **Group-Aware Data Splitting:** Prevents "data leakage" by ensuring data from any single flight does not appear in both the training and testing sets.
  - **Target Shaping:** The models are trained to predict a more stable "displacement" (meters North/East) rather than absolute GPS coordinates.
  - **Feature Scaling:** All data is normalized before being fed to the neural network, significantly improving its performance.

## How It Works

For every telemetry packet received, the system performs a multi-stage analysis:

1. **The Ensemble:** The Physics, XGBoost, and LSTM models all independently calculate a predicted landing displacement.
2. **The Fusion:** The three predictions are fed into the XGBoost Meta-Model, which outputs a single, fused, high-confidence displacement.
3. **The Final Projection:** The fused displacement is converted back into a final, absolute GPS coordinate for display.

## How to Run This Project

1. **Install Libraries:**

    ```bash
    pip install -r requirements.txt
    ```

2. **Generate the Rich Dataset:**

    ```bash
    python generate_rich_dataset.py
    ```

3. **Prepare the Data for ML (Crucial Step):**

    ```bash
    python data_preparation.py
    ```

4. **Train the Ensemble of Models (This will take a few minutes):**

    ```bash
    python train_final_ensemble.py
    ```

5. **Run the Ultimate Predictor:**

    ```bash
    python main_app_final.py
    ```
