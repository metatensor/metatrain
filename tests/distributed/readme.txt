This sub-folder contains a simple test to check whether distributed training
works as expected. The test consists of a training exercise using PET on a
small ethanol dataset. Distributed training is enabled automatically when the
job runs under more than one SLURM task, so "options.yaml" is used both for
the single-GPU run (submit.sh) and for the distributed run
(submit-distributed.sh); the logs obtained from the two should be the same.

Moreover, we have:
- options-distributed-multitarget.yaml, testing multi-target training
- options-distributed-finetuning.yaml, testing fine-tuning on a new target
