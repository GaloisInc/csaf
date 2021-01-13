### Introduction

This file includes explanations and guidance on how to detect anomalies in a CSAF environment.

### Step 1: Generating data

In order to train predictors, it is required to first gather and store data. Before generating data, one should generate 
a set of initial conditions for the system to begin with. To do so, `generate_f16_ic.py` can be used. Following shows an 
example command on how to run `generate_f16_ic.py`:

```commandline
python generate_f16_ic.py --ic-file ic.json --n-samples 100
```

***In general, to get the meaning and purpose of each input argument, simply run:***

```commandline
python file.py --help
```

Now that the initial conditions for 100 samples are stored, they can be used to generate trajectory data using 
`generate_dataset.py` script with the proper input arguments. Following is an example command to generate data:

```commandline
python generate_dataset.py --ic-file=ic.json --config-file=../examples/f16/f16_simple_airspeed_config.toml --data-format openai --output-file data_airspeed.json
```

So far, we have generated 100 trajectories with different length using the "airspeed" autopilot. These data can then be
used to train the predictors.

### Step 2: Train predictors

Predictors, or recurrent general value distributions, are very close in principle to value functions in common 
reinforcement learning literature. They estimate the expected distribution of returns given a state and horizon. To 
train them, following command can be used:

```commandline
python main.py --update_gvds --env_name CSAF_airspeed --lr 1e-4 --batch_size 64 --horizon 1 --target_return_type time_avg --data_path ../src/data_airspeed.json --gvd_name 0
```
During the training, a plot containing training loss and test loss is stored so that the training process can be monitored.

### Step 3: Test predictors

When predictors are trained, we can simply run them over a nominal *unseen* trajectory to see how they perform in terms 
of predicting the features throughout the trajectory. It can be done by:

```commandline
python main.py --test_gvds --env_name CSAF_airspeed --lr 1e-4 --batch_size 64 --target_return_type time_avg --data_path ../src/data_airspeed.json --test_data_path ../src/data_airspeed_test.json --horizon 1
```

Once test is done, a plot containing all predictors' estimations and actual returns are stored. This plot is very useful 
to see how they perform. If the predictors have a high accuracy, we can proceed to the next step.

### Step 4: Anomaly detection

In this step, using the predictors, we want to detect anomalies happening in an anomalous environment. But before that, 
we need to have trajectories representing the anomalous systems. This can be done simply by just going to step 1 and 
instead of using the nominal system configuration file, use the noisy system configuration. Once the noisy data are 
generated and stored, anomaly detection can be started. Following command starts the anomaly detection:

```commandline
python main.py --pair_anomaly_detection --env_name CSAF_airspeed --batch_size 1 --horizon 1 --target_return_type time_avg --score_calc_method knn --merge_type avg --data_path ../src/data_airspeed.json --noisy_data_path ../src/ddata_airspeed_noisy.json --test_data_path ../src/data_airspeed_test.json
```