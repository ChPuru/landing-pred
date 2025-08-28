# Project 1: Advanced Physics-Based Landing Predictor

This project is the most powerful landing zone predictor that can be built using **only physics and algorithms**, with no machine learning. It's designed to be a highly accurate and realistic simulation tool for our CanSat's descent.

Instead of a single, simple calculation, this system runs a high-speed, second-by-second simulation of the entire remaining flight every time it gets new data from the probe.

## Key Features & Innovations

- **Iterative Simulation Engine:** For every telemetry packet received, the ground station simulates the *entire rest of the flight*, one second at a time. This makes the prediction incredibly dynamic and responsive to the latest data.

- **Layered Wind Model:** We can define different wind speeds and directions for various altitude bands (e.g., strong winds high up, calm winds near the ground). The simulation uses the correct wind as the CanSat descends through each layer, dramatically improving accuracy.

- **Dynamic Air Density & Terminal Velocity:** The model calculates the changing air density as the CanSat falls. This allows it to compute a realistic terminal velocity, correctly simulating that the probe falls faster in the thin upper atmosphere and slower near the ground.

- **Parachute Oscillation:** The simulation includes a small, random "wobble" factor at each step. This mimics the natural, chaotic swinging of the parachute, leading to a more organic and believable predicted flight path.

## How It Works

The core of this project is the **Forward Simulation Engine**. When a live telemetry packet arrives, the engine:

1. Initializes a "virtual CanSat" at the live probe's exact position and altitude.
2. Enters a high-speed loop, simulating the descent one second at a time until the virtual CanSat reaches the ground.
3. In each step of the loop, it looks up the correct wind for the current altitude, calculates the new terminal velocity based on air density, and updates the virtual CanSat's position.
4. The final coordinate of the virtual CanSat is the predicted landing spot. This entire process takes a fraction of a second.

## File Structure

- `config_advanced.py`: A detailed configuration file to set the CanSat's physical properties (mass, parachute size) and define the layered wind model.
- `physics_engine_advanced.py`: The heart of the project. Contains all the advanced physics logic for the iterative simulation.
- `data_sources/`: Contains the script to replay our test flight data.
- `generate_dataset.py`: A utility to create a large flight log for testing.
- `main_app_advanced.py`: The main application that connects the data source to the simulation engine and displays the final prediction.

## How to Run This Project

1. **Install Libraries:**

    ```bash
    pip install pandas numpy
    ```

2. **Generate Test Data (If you don't have it yet):**

    ```bash
    python generate_dataset.py
    ```

3. **Run the Predictor:**

    ```bash
    python main_app_advanced.py
    ```
