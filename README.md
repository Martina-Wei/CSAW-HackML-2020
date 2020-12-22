# CSAW-HackML-2020

This repo is for repairing BD which is a challeng from CSAW-HackML-2021 [here] (https://github.com/Martina-Wei/CSAW-HackML-2021)

```bash
├── data 
    └── clean_validation_data.h5 // this is clean data used to evaluate the BadNet and design the backdoor defense
    └── clean_test_data.h5
    └── sunglasses_poisoned_data.h5
├── models
    └── anonymous_bd_net.h5
    └── anonymous_bd_weights.h5
    └── sunglasses_bd_net.h5
    └── sunglasses_bd_weights.h5
├── architecture.py
├── eval.py // this is the evaluation script
└── bd1.py // this shows the filter for db1
```
    

## Dependencies
   1. Python 3.6.9
   2. Keras 2.3.1
   3. Numpy 1.16.3
   4. Matplotlib 2.2.2
   5. H5py 2.9.0
   6. TensorFlow-gpu 1.15.2
   
## Evaluating the Backdoored Model
   1. To simply evaluate the original backdoored model, execute `eval.py` by running:  
      `python3 eval.py <clean validation data directory> <model directory>`.
      
      E.g., `python3 eval.py data/clean_validation_data.h5  models/sunglasses_bd_net.h5`.
      
      
   2. For the detet poisoned data, we construct a filter use validation data.
       `python3 bd1.py <model path> -m <mode==pca/mi> -v <clean validation data directory> -t <clean test data directory> -n <lower bound> -p <poisoned data>`

      E.g., `python3 repair_generator.py models/multi_trigger_multi_target_bd_net.h5 -v data/clean_validation_data.h5  -t data/lipstick_poisoned_data.h5 -m pca > result`
   
