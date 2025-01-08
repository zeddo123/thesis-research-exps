## Anomaly Classification (semi-supervised vs fewshot/zeroshot)
This repository holds the code for the models used in the thesis. In each model's directory, there's
`easy.py` scripts that provide easy to set parameters such as which dataset to test, number of supporting
sets to used, etc.

By default runs are upload to a weights and biases project for experiment tracking.

```sh
usage: easy-inCTRL-runner [-h] [-dn DATASET_NAME] [-dp DATASET_PATH] [-mt MODEL_TYPE] [-mp MODEL_PATH]
                          [-om OUTPUT_METRICS] [-op OUTPUT_PREDICTIONS] [-fs FEW_SHOT_PATH]

options:
  -h, --help            show this help message and exit
  -dn DATASET_NAME, --dataset_name DATASET_NAME
  -dp DATASET_PATH, --dataset_path DATASET_PATH
  -mt MODEL_TYPE, --model_type MODEL_TYPE
  -mp MODEL_PATH, --model_path MODEL_PATH
  -om OUTPUT_METRICS, --output_metrics OUTPUT_METRICS
  -op OUTPUT_PREDICTIONS, --output_predictions OUTPUT_PREDICTIONS
  -fs FEW_SHOT_PATH, --few_shot_path FEW_SHOT_PATH

```

