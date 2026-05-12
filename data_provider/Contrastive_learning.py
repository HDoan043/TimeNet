# Dataset for contrastive learning
# the method __get_item__ should return: 
#   _ a sample which is an original normal window A
#   _ positive sample which is a version of window A but containing noises, jitter, ....
#   _ negative sample which is a version of window A but containing anomalies
import numpy as np
import pandas as pd
import torch.nn as nn

class Dataset_Contrastive(Dataset):
    def __init__(self, args, root_path, flag='train', task_name="long_term_forecasting",
                 size=None, features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='5min', 
                 seasonal_patterns=None, train_ratio = 0.7, test_ratio = 0.2, step = 1):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            if task_name.lower() == "long_term_forecasting":
                self.seq_len = 24 * 4 * 4
                self.label_len = 24 * 4
                self.pred_len = 24 * 4
            else:
                self.win_size = 500
                self.step = step
        else:
            if task_name.lower() == "long_term_forecasting":
                self.seq_len = size[0]
                self.label_len = size[1]
                self.pred_len = size[2]
            else:
                self.win_size = size
                self.step = step
                
        # init
        assert flag in ['train', 'test', 'val', 'full']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'full': 3}
        self.task_name = task_name.lower()
        self.flag = flag
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,self.data_path))

        # =============== REORDER COLUMNS =====================
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove('date')
        if self.target in cols:
            cols.remove(self.target)
            df_raw = df_raw[['date'] + cols + [self.target]]
        else:
            df_raw = df_raw[['date'] + cols]

        if "Unnamed: 0" in cols:
            df_raw = df_raw.drop(["Unnamed: 0"], axis=1)
        # ================ TRAIN TEST SPLIT =====================
        num_train = int(len(df_raw) * self.train_ratio)
        num_test = int(len(df_raw) * self.test_ratio)
        num_vali = len(df_raw) - num_train - num_test
        if self.task_name == "long_term_forecasting":
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len, 0]
            border2s = [num_train, num_train + num_vali, len(df_raw), len(df_raw)]
        else:
            border1s = [0, num_train, len(df_raw) - num_test , 0]
            border2s = [num_train, num_train + num_vali, len(df_raw), len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # ========= DETERMINE FEATURE COLUMNS & TARGET COLUMN =========
        if self.features == 'M' or self.features == 'MS':
            col = list(df_raw.columns)
            col.remove('date')
            if self.target in col:
                col.remove(self.target)
            df_data = df_raw[col]
        elif self.features == 'S':
            df_data = df_raw[[self.target]] 

        if self.features == 'M':
            y = df_data.copy()
        else:
            y = df_raw[self.target]

        # ===================== NORMALIZATION ==========================
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)            # fit in train set
            data = self.scaler.transform(df_data.values)  # transform in both train and test set
            y = y.values
        else:
            data = df_data.values
            y = y.values

        
        df_stamp = df_raw[['date']][border1:border2]
        timestamps = list(pd.to_datetime(df_stamp['date']))
        df_stamp['date'] = timestamps 

        # ==================== ENSURE CONTINUALNESS ========================
        # Inspect freq
        print("[ℹ️] Number of timestamps in {} set: {}".format(self.flag, len(timestamps)))
        df_freq = pd.DataFrame({"timestamp": timestamps})
        df_freq["previous_timestamp"] = df_freq["timestamp"].shift(1)
        mod_freq_series = (df_freq["timestamp"] - df_freq["previous_timestamp"]).mode()/np.timedelta64(1, "m")
        mod_freq = str(mod_freq_series.iloc[0]) + "min"

        # Inspect interupt
        full_timestamp_range = pd.date_range(start=timestamps[0], end=timestamps[-1], freq=mod_freq)
        full_timestamp_df = pd.DataFrame({"timestamps": full_timestamp_range})
        full_timestamp_df["missing_timestamps"] = ~full_timestamp_df["timestamps"].isin(timestamps)*1
        timestamps_df_with_index = pd.DataFrame({"index": df_stamp.index, "timestamps": timestamps})
        full_timestamp_df_with_index = pd.merge(
            full_timestamp_df,
            timestamps_df_with_index,
            on="timestamps",
            how="right"
        )
        interrupted_timestamp = full_timestamp_df["missing_timestamps"].values
        interrupted_timestamp = np.append(interrupted_timestamp, -1)
        index_col = full_timestamp_df_with_index["index"].values
        index_col = np.append(index_col, 1)
        
        interrupted_index = []
        stack = []
        for i, (index, is_interrupted) in enumerate(list(zip(index_col, interrupted_timestamp))):
            if not is_interrupted:
                stack.append(index)
            elif len(stack) :
                interrupted_index.append(stack.copy())
                stack.clear()
                
        if len(interrupted_index)>1:
            print("[ℹ️] The sequence for {} is interupted at {} indexes: {}".format(self.flag, 
                                                                                      len(interrupted_index), 
                                                                                      [each[0] for each in interrupted_index]))
        else:
            print("[ℹ️] The sequence for {} is contiguous".format(self.flag))

        # Get the possible indexes: 
        #     a training sample is a window of timestamps, 
        #     the index of training sample is the index of the beginning timestamp in window
        
        possible_index = []
        sample_length = self.seq_len + self.pred_len if self.task_name == "long_term_forecasting" else self.win_size
        for continous_indexes in interrupted_index:
            if len(continous_indexes) >= sample_length:
                max_start_index = len(continous_indexes) - sample_length + 1
                step_index_index = range(0, max_start_index, self.step)
                step_index = [continous_indexes[index] for index in step_index_index]
                possible_index.extend(step_index)
        self.possible_index = possible_index
        self.possible_timestamps = pd.DataFrame([timestamps[index:index+sample_length] for index in possible_index])
        print("[ℹ️] Number of {} samples: {}".format(self.flag, len(possible_index)))

        # ========================= ENCODE TIMELABEL ===============================
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            if self.freq == "5min":
                df_stamp['5minute'] = df_stamp.date.dt.minute.map(lambda x: x // 5)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # ========================== GET FINAL DATA ====================================
        # Get feature data and label data
        self.data_x = data[border1:border2]
        self.data_y = y[border1:border2]
        # Reset index
        self.possible_index = np.array(possible_index) - border1
        
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        index = self.possible_index[index]
        x_index_start = index
        x_index_end = index + self.win_size
        y_index_start = index 
        y_index_end = index + self.win_size

        seq_x = self.data_x[x_index_start: x_index_end]
        seq_y = self.data_y[y_index_start: y_index_end]
        seq_x_mark = self.data_stamp[x_index_start:x_index_end]
        seq_y_mark = self.data_stamp[y_index_start:y_index_end]
        
        if self.flag == "train" or self.set_type == 0:
            positive_sample = run_augment_single(seq_x, seq_y)
            negative_sample = inject_anomaly(seq_x)
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        # return len(self.data_x) - self.seq_len - self.pred_len + 1
        return len(self.possible_index)

    def get_timestamps(self):
        return self.possible_timestamps

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_corr_matrix(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        col = list(df_raw.columns)
        col.remove("date")
        col.remove(self.target)
        num_train = int(len(df_raw) * self.train_ratio)
        df_raw = df_raw[0: num_train]
        df = df_raw[col]
        
        corr_matrix = df.corr().values

        return corr_matrix