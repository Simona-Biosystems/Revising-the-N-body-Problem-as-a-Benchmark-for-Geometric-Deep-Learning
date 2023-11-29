# Revising the N-body Problem as a Benchmark for Geometric Deep Learning
This repository contains the code for the paper "Revising the N-body Problem as a Benchmark for Geometric Deep Learning".

This README provides instructions on how to set up, train, and benchmark the Ponita model on N-body gravitational simulations in self-feed mode.

## Installation
The easiest and most convenient way to install the project is to use Docker.
### Building the Docker image
```bash
docker build -t nbody-cuda . 
```

### Running the Docker container
```bash
docker run --runtime=nvidia --gpus all -it -v $(pwd):/Revising-the-N-body-Problem-as-a-Benchmark-for-Geometric-Deep-Learning nbody-cuda
```

We don't recommend using Conda, as it has problems with conflicting dependencies. However, a `environment.yml` file is provided for reference.


## Training the Model
To train like we did for the paper, just run the training script. Default parameters will be used. Here is an example command:

   ```bash
   python -m 'train'
   ```
   (It is important that you run Python with the `-m` flag to avoid import issues.)

   See `train.py` for parameters and how to change them.

## Running Inference and Plotting Macros-Properties
The models are tested periodically during training. This entails generating trajectories (`batch_size` of them by default) and plotting their macroproperties against the ground truth trajectories' macroproperties. You can these in the respective checkpoint directories (for example, `runs/ponita/2024-10-17_16-54-12/checkpoints/1024`). You can change how often the model is tested by changing the `test_macros_every` argument to the training script. 
(For now, this was optimized for convenience, not efficiency. If you don't want to test periodically, edit the code and do the inference and plotting manually afterwards as needed.)

To run inference (and plot macros-properties of the generated trajectories) manually, just run the `infer_self_feed.py` script:
```bash
python -m 'helper_scripts.infer_self_feed' --model-path=runs/ponita/2024-10-17_16-54-12/model.pth
```

## Running the KS Tests
Once you have trained a model and generated some macro-properties, you can create the KS-test-related plots from the paper with:
```bash
python -m 'ks_test.ks_test_model_checkpoints' --run-path=runs/ponita/2024-10-17_16-54-12
```


