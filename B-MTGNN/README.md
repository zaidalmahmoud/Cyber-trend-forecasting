# B-MTGNN
This is a PyTorch implementation of the model in the paper: "Forecasting Cyber Threats and Pertinent Technologies".
The model forecasts the graph of cyber attacks and pertinent technologies 3 years in advance, while extending the [MTGNN](https://dl.acm.org/doi/abs/10.1145/3394486.3403118) model proposed by Wu et al.

## Requirements
The model is implemented using Python3 with dependencies specified in requirements.txt

## Data Smoothing

The file **smoothing.py** performs double exponential smoothing on the data (data.txt) and produces the file **sm_data.csv**. These data files including the graph adjacency file (graph.txt) can be found in the directory called **data**.

### Traffic datasets
Download the METR-LA and PEMS-BAY dataset from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) provided by [Li et al.](https://github.com/liyaguang/DCRNN.git) . Move them into the data folder. 

```

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
