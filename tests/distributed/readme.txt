This sub-folder contains a simple test to check whether distributed training
works as expected. The test consists of a training exercise using SOAP-BPNN
on a small ethanol dataset. The logs obtained by using "options.yaml" and
"options-distributed.yaml" should be the same.

Moreover, we have:
- options-distributed-multitarget.yaml, testing multi-target training
- options-distributed-finetuning.yaml, testing fine-tuning on a new target
