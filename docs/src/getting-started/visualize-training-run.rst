.. visualize-training:

Visualize your training run
===========================

For each training run a new output directory in the format outputs/YYYY-MM-DD/HH-MM-SS based on
the current date and time is created. There you will find intermediate and final model checkpoints,
the restart options_restart.yaml file as well as the log file train.log and train.csv,
where various training metrics are saved.

Per default, the training metrics will contain the `training loss` and `validation loss`,
RMSE and MAE for requested targets and derivatives (commonly energies and forces)





Assuming your training metrics is in 