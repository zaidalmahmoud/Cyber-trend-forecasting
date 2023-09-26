# B-MTGNN
This is a PyTorch implementation of the Bayesian model proposed in the paper: "Forecasting Cyber Threats and Pertinent Technologies".
The model forecasts the graph of cyber attacks and pertinent technologies 3 years in advance, by extending the [MTGNN](https://dl.acm.org/doi/abs/10.1145/3394486.3403118) model proposed by Wu et al.

In this model, we employ the Bayesian approach to capture epistemic uncertainty. Specifically, we employ the Monte Carlo dropout method where the use of dropout neurons during inference provides a Bayesian approximation of the deep Gaussian processes. Therefore, during the prediction phase, the trained model runs multiple times, which results in a distribution of prediction (representing the uncertainty) rather than a single point.

## Requirements
The model is implemented using Python3 with dependencies specified in requirements.txt

## Data Smoothing
All data files can be found in the directory called **data**.

The file **smoothing.py** performs double exponential smoothing on the data (**data.txt**) and produces the file **sm_data.csv**. These data files including the graph adjacency file (**graph.csv**) can be found in the directory called **data**.

## Hyper-parameter Optimisation
The hyper-parameter optimisation is performed in the file **train_test.py**. This script performs random search to produce the optimal set of hyper-parameters. These hyper-parameters are finally saved as an output in the file called **hp.txt**, which is in the directory **model/Bayesian**. The output also includes validation and testing results when using the optimal set of hyper-parameters. These results include plots for the predicted curves against the actual curves. These are saved in the directories called **Validation** and **Testing** in the directory **model/Bayesian**. For the evaluation, 2 metrics are used namely the Root Relative Squared Error (RSE) and the Relative Absolute Error (RAE). These metrics are saved in the same directories (Validation and Testing), and the average value of these metrics across 142 nodes are also displayed on the console as a final output. 

# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5

```

## Model Training

### Single-step

* Solar-Energy

```
python train_single_step.py --save ./model-solar-3.pt --data ./data/solar_AL.txt --num_nodes 137 --batch_size 4 --epochs 30 --horizon 3
#sampling
python train_single_step.py --num_split 3 --save ./model-solar-sampling-3.pt --data ./data/solar_AL.txt --num_nodes 137 --batch_size 16 --epochs 30 --horizon 3
```
* Traffic 

```
python train_single_step.py --save ./model-traffic3.pt --data ./data/traffic.txt --num_nodes 862 --batch_size 16 --epochs 30 --horizon 3
#sampling
python train_single_step.py --num_split 3 --save ./model-traffic-sampling-3.pt --data ./data/traffic --num_nodes 321 --batch_size 16 --epochs 30 --horizon 3
```

* Electricity

```
python train_single_step.py --save ./model-electricity-3.pt --data ./data/electricity.txt --num_nodes 321 --batch_size 4 --epochs 30 --horizon 3
#sampling 
python train_single_step.py --num_split 3 --save ./model-electricity-sampling-3.pt --data ./data/electricity.txt --num_nodes 321 --batch_size 16 --epochs 30 --horizon 3
```

* Exchange-Rate

```
python train_single_step.py --save ./model/model-exchange-3.pt --data ./data/exchange_rate.txt --num_nodes 8 --subgraph_size 8  --batch_size 4 --epochs 30 --horizon 3
#sampling
python train_single_step.py --num_split 3 --save ./model-exchange-3.pt --data ./data/exchange_rate.txt --num_nodes 8 --subgraph_size 2  --batch_size 16 --epochs 30 --horizon 3
```
### Multi-step
* METR-LA

```
python train_multi_step.py --adj_data ./data/sensor_graph/adj_mx.pkl --data ./data/METR-LA --num_nodes 207
```
* PEMS-BAY

```
python train_multi_step.py --adj_data ./data/sensor_graph/adj_mx_bay.pkl --data ./data/PEMS-BAY/ --num_nodes 325
```

## Citation

```
@inproceedings{wu2020connecting,
  title={Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks},
  author={Wu, Zonghan and Pan, Shirui and Long, Guodong and Jiang, Jing and Chang, Xiaojun and Zhang, Chengqi},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  year={2020}
}
```
