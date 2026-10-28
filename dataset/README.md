# Dataset preparation

The datasets are not included in this repository.

Download the seven public forecasting benchmarks and organize them as follows:

```text
dataset/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── weather/
│   └── weather.csv
├── electricity/
│   └── electricity.csv
└── traffic/
    └── traffic.csv
```

## Public sources

### ETT-small

The ETTh1, ETTh2, ETTm1, and ETTm2 datasets are available from:

[https://www.kaggle.com/datasets/alaaelmor/ettsmall](https://www.kaggle.com/datasets/alaaelmor/ettsmall)

Place the four CSV files under:

```text
dataset/ETT-small/
```

### Weather

The Weather dataset is available from:

[https://www.bgc-jena.mpg.de/wetter/weather_data.html](https://www.bgc-jena.mpg.de/wetter/weather_data.html)

Place the CSV file at:

```text
dataset/weather/weather.csv
```

### Electricity

The Electricity dataset is available from:

[https://zenodo.org/records/4656140](https://zenodo.org/records/4656140)

Place the CSV file at:

```text
dataset/electricity/electricity.csv
```

### Traffic

The Traffic dataset is available from:

[https://zenodo.org/records/4656132](https://zenodo.org/records/4656132)

Place the CSV file at:

```text
dataset/traffic/traffic.csv
```

## Dataset summary

| Dataset | Variables | Sampling resolution |
|:--------|----------:|:--------------------|
| ETTh1 | 7 | 1 hour |
| ETTh2 | 7 | 1 hour |
| ETTm1 | 7 | 15 minutes |
| ETTm2 | 7 | 15 minutes |
| Weather | 21 | 10 minutes |
| Electricity | 321 | 1 hour |
| Traffic | 862 | 1 hour |