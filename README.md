# Scaleformer
**[Scaleformer: Iterative Multi-scale Refining Transformers for Time Series Forecasting](https://arxiv.org/abs/2206.04038), ICLR 2023**

This repo is a modified version of the public implementation of [Autoformer paper](https://arxiv.org/abs/2106.13008) which can be find in this [repository](https://github.com/thuml/Autoformer). We also use the related parts of [FEDformer](https://github.com/MAZiqing/FEDformer), [FiLM](https://github.com/tianzhou2011/FiLM/), and [NHits](https://github.com/Nixtla/neuralforecast).

## Why Scaleformer?

Using iteratively refining a forecasted time series at multiple scales with shared weights, architecture adaptations and a specially-designed normalization scheme, we are able to achieve significant performance improvements with minimal additional computational overhead.

<p align="center">
<img src="figs\teaser.png" width=90% alt="" align=center />
<br><br>
<b>Figure 1.</b> Overview of the proposed framework. (<b>Left</b>) Representation of a single scaling block. In each step, we pass the normalized upsampled version of the output from previous step along with the normalized downsampled version of encoder as the input. (<b>Right</b>) Representation of the full architecture. We process the input in a multi-scale manner iteratively from the smallest scale to the original scale.
</p>

Our experiments on various public datasets demonstrate that the proposed method outperforms the corresponding baselines. Depending on the choice of transformer architecture, our mutli-scale framework results in mean squared error reductions ranging from 5.5% to 38.5%.

<p align="center">
<img src="figs\table.png" width=90% alt="" align=center />
<br><br>
<b>Table 1.</b> 
Comparison of the MSE and MAE results for our proposed multi-scale framework version of different methods (<b>-MSA</b>) with respective baselines. Results are given in the multi-variate setting, for different lenghts of the horizon window. The best results are shown in <b>Bold</b>. Our method outperforms vanilla version of the baselines over almost all datasets and settings. The average improvement (error reduction) is shown in Green numbers at the bottom with respect the base models.
</p>

## Installation

**1. Clone our repo and install the requirements:**
```
git clone https://github.com/BorealisAI/scaleformer.git
cd scaleformer
pip install -r requirements.txt
```

**2. Supported Data Sources**

Scaleformer now supports multiple data sources with a unified interface.

### CSV Files (Legacy)

Download datasets from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/e1ccfff39ad541908bae/) or [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing).

Place them in `./dataset/`:
```
scaleformer
├── dataset
│   ├── exchange_rate
│   │   └── exchange_rate.csv
│   ├── traffic
│   │   └── traffic.csv
│   ├── electricity
│   │   └── electricity.csv
│   └── ...
```

### PostgreSQL / MySQL / SQLite

Connect directly to your SQL database:

```bash
# PostgreSQL
python -u run.py \
  --data_source postgresql \
  --root_path "postgresql://username:password@localhost:5432/timeseries_db" \
  --data_path "time_series_table" \
  --time_column "timestamp" \
  --model InformerMS \
  --pred_len 192

# MySQL
python -u run.py \
  --data_source mysql \
  --root_path "mysql://username:password@localhost:3306/timeseries" \
  --data_path "sensor_data" \
  --model AutoformerMS \
  --pred_len 96

# SQLite
python -u run.py \
  --data_source sqlite \
  --root_path "sqlite:///./data/timeseries.db" \
  --data_path "measurements" \
  --model TransformerMS \
  --pred_len 336
```

### MongoDB

Connect to MongoDB collections:

```bash
python -u run.py \
  --data_source mongodb \
  --root_path "mongodb://localhost:27017" \
  --database "timeseries" \
  --collection "metrics" \
  --time_field "timestamp" \
  --value_fields "[\"temperature\", \"humidity\", \"pressure\"]" \
  --model FEDformerMS \
  --pred_len 720
```

For MongoDB Atlas (cloud):
```bash
python -u run.py \
  --data_source mongodb \
  --root_path "mongodb+srv://username:password@cluster.mongodb.net/timeseries" \
  --collection "sensor_readings" \
  --model PerformerMS \
  --pred_len 48
```

### Redis

Support for both Redis Streams and RedisTimeSeries module:

```bash
# Redis Streams
python -u run.py \
  --data_source redis \
  --root_path "redis://localhost:6379/0" \
  --key_pattern "ts:*" \
  --start_time "2024-01-01 00:00:00" \
  --model NHitsMS \
  --pred_len 24

# RedisTimeSeries with aggregation
python -u run.py \
  --data_source redis \
  --root_path "rediss://localhost:6379/0" \
  --use_timeseries_module True \
  --key_pattern "metrics:*" \
  --aggregation "avg" \
  --bucket_size_msec 60000 \
  --model FiLMMS \
  --pred_len 96
```

### Parquet Files

High-performance columnar storage:

```bash
# Single parquet file
python -u run.py \
  --data_source parquet \
  --root_path "./data/timeseries.parquet" \
  --time_column "timestamp" \
  --model ReformerMS \
  --pred_len 192

# Partitioned dataset with filters
python -u run.py \
  --data_source parquet \
  --root_path "./data/partitioned/" \
  --time_column "timestamp" \
  --filters "[[('year', '=', 2023), ('month', '=', 1)]]" \
  --model FEDformerMS \
  --pred_len 96
```

### Synthetic Data

Generate synthetic Mackey-Glass series:

```bash
python -u run.py \
  --data synthetic \
  --model TransformerMS \
  --pred_len 96
```

## Running the code

### 1. Running a single experiment

You can run a single experiment using the following command:

```bash
python -u run.py --data_path {DATASET} --model {MODEL} --pred_len {L} --loss {LOSS_FUNC}
```

For example, using **Informer-MSA** model for **traffic** dataset with output length **192** and **adaptive** loss function:

```bash
python -u run.py --data_path traffic.csv --model InformerMS --pred_len 192 --loss adaptive
```

**Note**: For new data sources, use `--data_source` instead of `--data`:

```bash
python -u run.py \
  --data_source postgresql \
  --root_path "postgresql://localhost/timeseries" \
  --data_path "measurements" \
  --model InformerMS \
  --pred_len 192
```

### 2. Running all experiments

To run all experiments using Slurm, use `run_all.sh` which uses `run_single.sh` to submit jobs with different parameters. The final errors will be available in `results.txt` and you can check the `slurm` directory for logs of each experiment.

### 3. Available Models

- **AutoformerMS**: Multi-scale Autoformer
- **InformerMS**: Multi-scale Informer
- **TransformerMS**: Multi-scale Transformer
- **ReformerMS**: Multi-scale Reformer
- **FEDformerMS**: Multi-scale FEDformer
- **PerformerMS**: Multi-scale Performer
- **NHitsMS**: Multi-scale NHits
- **FiLMMS**: Multi-scale FiLM

### 4. Configuration Parameters

Common parameters:
- `--seq_len`: Input sequence length (default: 96)
- `--label_len`: Label sequence length (default: 48)
- `--pred_len`: Prediction horizon length
- `--batch_size`: Batch size (default: 32)
- `--features`: Feature mode - 'S' (single), 'M' (multi), 'MS' (multi-single)
- `--target`: Target column name (default: 'OT')
- `--loss`: Loss function - 'mse' or 'adaptive'
- `--embed`: Time encoding - 'timeF' or other
- `--freq`: Frequency - 'h' (hourly), 't' (minutely), 'd' (daily), etc.

Data source specific parameters:
- SQL: `--time_column`, `--value_columns`
- MongoDB: `--database`, `--collection`, `--time_field`, `--value_fields`, `--query`
- Redis: `--key_pattern`, `--start_time`, `--end_time`, `--use_timeseries_module`, `--aggregation`
- Parquet: `--time_column`, `--filters`, `--columns`

## Advanced Usage

### Custom Database Queries

For SQL databases, you can use custom queries instead of table names:

```bash
python -u run.py \
  --data_source postgresql \
  --root_path "postgresql://localhost/db" \
  --query "SELECT timestamp, temperature, humidity FROM sensor_data WHERE date >= '2023-01-01' ORDER BY timestamp" \
  --model InformerMS \
  --pred_len 96
```

### Redis with Time Range

```bash
python -u run.py \
  --data_source redis \
  --root_path "redis://localhost:6379/0" \
  --key_pattern "sensor:*" \
  --start_time "2024-01-01 00:00:00" \
  --end_time "2024-01-31 23:59:59" \
  --model AutoformerMS \
  --pred_len 336
```

### Feature Modes

- `--features S`: Single target variable (univariate)
- `--features M`: Multi-variable (all columns as features)
- `--features MS`: Mix of single target with other features

## Contact

If you have any question regarding the ScaleFormer, please contact aminshabaany@gmail.com.

## Citation

```
@article{shabani2022scaleformer,
  title={Scaleformer: Iterative Multi-scale Refining Transformers for Time Series Forecasting},
  author={Shabani, Amin and Abdi, Amir and Meng, Lili and Sylvain, Tristan},
  journal={arXiv preprint arXiv:2206.04038},
  year={2022}
}
```

## Acknowledgement

We acknowledge the following github repositories that made the base of our work:

https://github.com/thuml/Autoformer

https://github.com/MAZiqing/FEDformer

https://github.com/jonbarron/robust_loss_pytorch.git

https://github.com/Nixtla/neuralforecast

https://github.com/tianzhou2011/FiLM/
