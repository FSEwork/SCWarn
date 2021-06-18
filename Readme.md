# SCWarn

SCWarn can identify bad software changes in online service systems.

## Dependencies

numpy >= 1.17.4  
matplotlib >= 3.0.3  
pandas >= 0.25.3  
tensorflow >= 1.13.1  
torch >= 1.3.1  
PyYAML >= 5.3.1  
scikit_learn >= 0.23.2  
statsmodels >= 0.12.1  
tqdm >= 4.52.0  

## Usage

**Step1: prepare for your data**

You need to transform your own data to csv format. And there must be a timestamp column in your training and testing data. 

**Step 2:  chagne the "config.yml"**

You may need to change the "train_path", "test_path" and "output_path" for your task.

And you can change the parameters in config.yml to confiure the algorithms  used by SCWarn. 

The meaning of the parameters is as follows:

- **train_path**: the path of your training dataset. In SCWarn, you need to provide with dataset in CSV format.

- **test_path**: the path of your testing dataset. In SCWarn, you need to provide with dataset in CSV format.

- **output_path**: the path of SCWarn output. SCWarn gives the anomaly score of testing data stored in a CSV format file.

- **scaler**: 'standard' or 'minmax'. Choose a scaler to preprocess the data. The 'standard' scaler will normalize the data. And the 'minmax' scaler will scale the data to the range between zero to one.

- **dim(optional)**: dimensions of the data fed SCWarn. For example, "from 1 to 7" means that SCWarn will use the 1st to 7th columns of training and testing CSV data. (PS: the 0th column is timestamp). If 'config.yml' does not include the 'dim' parameter, SCWarn will use all the columns in CSV data.

- **algorithms**: Here, you can configure the algorithms used by SCWarn.

- - **LSTM**:

  - - **epoch**: total training epoch number.
    - **batch_size**: training batch size.
    - **learning_rate**: learning rate of the optimizer in SCWarn.
    - **model_path(optional)**: the path of an existing model. If 'LSTM' includes model_path, SCWarn will use the current model to test and not train a new model. 
    - **seq_len**: The sliding window size.

  - **ISST**: ISST can only handle one single metric.

  - - **dim_pos**: ISST can only handle one metric, so we need to select a column to feed ISST. Setting dim_pos to n means that ISST uses the nth column to train and test.

  - **AE**: Autoencoder, the configuration is the same as LSTM.

  - **VAE**: Variational Autoencoder, the configuration is the same as LSTM.

  - **GRU**: the configuration is the same as LSTM.

  - **MLSTM**: Multimodal-LSTM

  - - **modal**: The modal is a list that has two int elements. The first element is the number of the first modal(begin from the 1st column). The second element is the number of the 2nd modal.
    - The rest parameters are the same as LSTM.

  - **MMAE**: Multimodal-AE, the configuration is the same as LSTM.

```yml
train_path: "data/sample/train.csv"
test_path: "data/sample/abnormal.csv"
output_path: "result/sample.csv"

scaler: 'standard'

# select dimensions used to train and test
dim:
   from: 1
   to: 4

algorithms:
  LSTM:
      epoch: 10
      batch_size: 32
      learning_rate: 0.01 
      seq_len: 10

 ISST:
     dim_pos: 3

 AE:
     epoch: 50
     batch_size: 32
     learning_rate: 0.01
     seq_len: 10

 VAE:
     epoch: 10
     batch_size: 32
     learning_rate: 0.01
     seq_len: 10

 MLSTM:
     epoch: 10
     batch_size: 32
     learning_rate: 0.01
     seq_len: 10
     modal:
       - 4 
       - 3

 MMAE:
     epoch: 10
     batch_size: 32
     learning_rate: 0.01
     seq_len: 15
     modal:
       - 4
       - 3

```

**Step 3: run SCWarn**

Run the command below.

```shell
python main.py
```

And then check the result_path, you will get the anomlay scores for each sample in your testing data.

