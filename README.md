# Neural Network Visualization

This repository demonstrates a small Flask web application that visualizes how a
multilayer perceptron (MLP) learns a toy classification problem. The app trains
a network, captures the intermediate decision boundaries with Matplotlib and
saves them as an animated GIF.

## Project layout

```
Neural-Network-Visualization/
├── app.py               # Flask routes and server
├── neural_networks.py   # MLP implementation and visualization logic
├── templates/
│   └── index.html       # Web form for running experiments
├── static/
│   ├── script.js        # Client-side logic and validation
│   └── style.css        # Basic page styling
├── results/
│   └── visualize.gif    # Example output
├── Makefile             # Helper commands for setup and running
└── requirements.txt     # Python dependencies
```

### Running the demo

1. Create a virtual environment and install the required packages:

```bash
make install
```

2. Start the development server:

```bash
make run
```

The application listens on `http://localhost:3000`. Open this address in your
browser, choose an activation function (`relu`, `tanh` or `sigmoid`), specify the
learning rate and number of training steps, then click **Train and Visualize**.
The resulting animation is saved to `results/visualize.gif` and displayed on the
page.

### How it works

- **neural_networks.py** creates an `MLP` class with `forward` and `backward`
  methods and defines `visualize()`, which iterates training and uses
  `matplotlib.animation.FuncAnimation` to build the GIF.
- **app.py** exposes two routes: `/` renders the form and `/run_experiment`
  triggers `visualize()` using parameters from the client.
- **templates/** and **static/** hold the front-end code that submits the form
  via `fetch`, handles validation and displays the generated image.

The generated GIF is overwritten each time a new experiment runs. Feel free to
edit `visualize()` if you want to save multiple outputs.

### Next steps

Explore `neural_networks.py` to tweak the network architecture or the
visualization. You can also enhance the front end with additional controls or
progress indicators.

### Resources

- [YouTube demo](https://youtu.be/5rNrOVM7CA8)
