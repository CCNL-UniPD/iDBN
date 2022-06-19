# A developmental approach for training deep belief networks

### User guide

A `dbn-env.yml` file contains the useful dependencies and libraries needed. 

The code needs three `json` configuration files. 
* `cparams.json` contains configuration parameters: the algorithm to use, the dataset, whether to assess classification performance on the hidden activities, number of runs and whether to assess number acuity (this latter only for the SZ dataset);
* `lparams-mnist.json` contains the hyper-parameters used to fit the DBN for the MNIST dataset;
* `lparams-sz.json` contains hyper-parameters to fit the DBN for the SZ dataset and the instructions for the Weber fraction assessment: delta-rule classifier hyper-parameters, number of classifiers to fit, the number acuity reference quantities.

Examples of the configuration files used in the simulations are listed below.

##### `cparams.json`
````json
{
	"DATASET_ID"       : "SZ",
	"READOUT"          : false,
	"RUNS"             : 1,
	"LAYERS"           : 2,
	"ALG_NAME"         : "i",
	"NUM_DISCR"        : false
}
````

##### `lparams-mnist.json`
```json
{
	"BATCH_SIZE"       : 128,
	"EPOCHS"           : 50,
	"INIT_MOMENTUM"    : 0.5,
	"FINAL_MOMENTUM"   : 0.9,
	"LEARNING_RATE"    : 0.01,
	"WEIGHT_PENALTY"   : 1e-4
}
````

##### `lparams-sz.json`
```json
{
	"BATCH_SIZE"        : 128,
	"EPOCHS"            : 100,
	"INIT_MOMENTUM"     : 0.5,
	"FINAL_MOMENTUM"    : 0.9,
	"LEARNING_RATE"     : 0.01,
	"WEIGHT_PENALTY"    : 1e-4,
	
	"EPOCHS_NDISCR"     : [0, 1, 2, 3, 4, 5],
	"NDISCR_RANGES"     : {"8"  : [5, 6, 7, 8, 9, 10, 11, 12],
	                      "16" : [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]},
	"NUM_LCLASSIFIERS"  : 1,
	"LC_EPOCHS"         : 500,
	"LC_LEARNING_RATE"  : 0.000001,
	"LC_WEIGHT_PENALTY" : 0.000001,
	"LC_INIT_MOMENTUM"  : 0.5,
	"LC_FINAL_MOMENTUM" : 0.9	
}
````
