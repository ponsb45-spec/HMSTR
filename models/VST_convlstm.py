import ast
import datetime
import json
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from evaluation.uv_evaluate import GridEvaluate
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input, layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.layers import MultiHeadAttention

from dotenv import load_dotenv
load_dotenv()
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"
os.environ.pop("TF_GPU_ALLOCATOR", None)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf.config.optimizer.set_experimental_options({"layout_optimizer": False})
tf.config.optimizer.set_jit(False)
tf.get_logger().setLevel("ERROR")

plt.rcParams["image.cmap"] = "viridis"
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU配置警告: {e}")


class WindDataLoad:
    def __init__(self, input_steps, pred_steps, normalize=True, val_rate=0.1, test_rate=0.2, ele_path=None):
        self.y_test = None
        self.y_val = None
        self.y_train = None
        self.X_test = None
        self.X_val = None
        self.X_train = None
        self.C = None
        self.normalize = normalize
        self.center = None
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.val_rate = val_rate
        self.test_rate = test_rate
        self.B = None
        self.K = None
        self.H = None
        self.W = None
        self.scaler_y = StandardScaler()
        self.scalers_x = []
        self.ele_path = ele_path
        self.scaler_elevation = StandardScaler()

    def load_elevation_data(self):
        ele_data = np.load(self.ele_path).astype(np.float32)
        ele_data = np.where((ele_data < 0) | (np.isnan(ele_data)), 0.0, ele_data)
        if self.normalize:
            ele_data = self.scaler_elevation.fit_transform(ele_data)

        return ele_data.astype(np.float32)


    def build_sliding_window_dataset(self, data_original, data_vmd):

        data_original = np.transpose(data_original, (0, 2, 3, 1)).astype(np.float32)
        data_vmd = data_vmd.astype(np.float32)

        self.B, self.K, self.H, self.W, self.C = data_vmd.shape

        data_num = self.B - self.input_steps - self.pred_steps + 1
        if data_num <= 0:
            raise ValueError("时间步不足，无法构建滑动窗口样本。")

        y = np.zeros((data_num, self.pred_steps, self.H, self.W, 2), dtype=np.float32)
        x_vmd = np.zeros((self.K, data_num, self.input_steps, self.H, self.W, self.C), dtype=np.float32)
        sample_starts = np.arange(data_num, dtype=np.int32)

        for idx in range(data_num):
            input_end = idx + self.input_steps
            pred_end = input_end + self.pred_steps
            y[idx] = data_original[input_end:pred_end]

        for modal_idx in range(self.K):
            modal_series = data_vmd[:, modal_idx, :, :, :]
            for idx in range(data_num):
                x_vmd[modal_idx, idx] = modal_series[idx:idx + self.input_steps]

        return y, x_vmd, sample_starts

    def _build_sample_masks(self, sample_starts):
        train_boundary = int(self.B * (1 - self.val_rate-self.test_rate))
        val_boundary = int(self.B * (1 - self.test_rate))
        target_end = sample_starts + self.input_steps + self.pred_steps

        train_mask = target_end <= train_boundary
        val_mask = (target_end > train_boundary) & (target_end <= val_boundary)
        test_mask = target_end > val_boundary

        if not np.any(train_mask):
            raise ValueError("训练集为空，请调整 input_steps、pred_steps 或数据划分比例。")
        if not np.any(val_mask):
            raise ValueError("验证集为空，请减小 val_rate 或缩短预测窗口。")
        if not np.any(test_mask):
            raise ValueError("测试集为空，请减小 test_rate 或缩短预测窗口。")

        return train_mask, val_mask, test_mask

    def _transform_y(self, arr):
        if not self.normalize:
            return arr.astype(np.float32)
        shape = arr.shape
        arr_flat = arr.reshape(-1,2)
        return self.scaler_y.transform(arr_flat.reshape(-1, 2)).reshape(shape).astype(np.float32)
    def _transform_x(self,arr,scaler):
        if not self.normalize or scaler is None:
            return arr.astype(np.float32)
        shape = arr.shape
        arr_flat = arr.reshape(-1,self.C)
        transformed = scaler.transform(arr_flat)
        return transformed.reshape(shape).astype(np.float32)

    def generate(self, data_original, data_vmd):
        original_y, original_x_vmd, sample_starts = self.build_sliding_window_dataset(
            data_original, data_vmd
        )

        if self.normalize:
            self.scaler_y = StandardScaler()
            y_flat = original_y.reshape(-1,2)
            self.scaler_y.fit(y_flat)
            original_y = self._transform_y(original_y)

            self.scalers_x = []
            for modal_idx in range(self.K):
                scaler_x = StandardScaler()
                modal_x_flat = original_x_vmd[modal_idx].reshape(-1, self.C)
                scaler_x.fit(modal_x_flat)
                self.scalers_x.append(scaler_x)
                original_x_vmd[modal_idx] = self._transform_x(original_x_vmd[modal_idx], scaler_x)

        train_mask, val_mask, test_mask = self._build_sample_masks(sample_starts)

        self.y_train = original_y[train_mask].astype(np.float32)
        self.y_val = original_y[val_mask].astype(np.float32)
        self.y_test = original_y[test_mask].astype(np.float32)

        self.X_train = []
        self.X_val = []
        self.X_test = []
        for modal_idx in range(self.K):
            modal_x = original_x_vmd[modal_idx]
            self.X_train.append(modal_x[train_mask].astype(np.float32))
            self.X_val.append(modal_x[val_mask].astype(np.float32))
            self.X_test.append(modal_x[test_mask].astype(np.float32))

        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} 数据划分完成: "
            f"train={len(self.y_train)}, val={len(self.y_val)}, test={len(self.y_test)}"
        )
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test


class CNNLSTMModel:
    def __init__(self, input_dim, dropout):
        self.input_dim = input_dim
        self.dropout = dropout

    def build(self):
        input_layer = Input(shape=self.input_dim)
        x = layers.TimeDistributed(
            layers.Conv2D(32, (3, 3), padding="same", activation="swish")
        )(input_layer)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.TimeDistributed(
            layers.Conv2D(64, (3, 3), padding="same", activation="swish")
        )(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.TimeDistributed(
            layers.Conv2D(32, (3, 3), padding="same", activation="swish")
        )(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.Dropout(self.dropout)(x)
        # x = layers.Conv3D(64, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
        # x = layers.TimeDistributed(layers.BatchNormalization())(x)
        # x = layers.Dropout(self.dropout)(x)
        return input_layer, x

# 时间位置编码
class TemporalPositionalEncoding(layers.Layer):
    def __init__(self, seq_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.pos_embedding = layers.Embedding(input_dim=seq_len, output_dim=embed_dim,name="temporal_pos_embedding")

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        batch_size  = tf.shape(inputs)[0]
        positions = tf.range(start=0, limit=self.seq_len, delta=1)
        pos_emb = self.pos_embedding(positions)
        pos_emb = tf.cast(pos_emb, inputs.dtype)
        pos_emb = tf.expand_dims(pos_emb, axis=0)
        pos_emb = tf.tile(pos_emb, [batch_size, 1, 1])
        pos_emb = tf.reshape(pos_emb, (batch_size, self.seq_len, 1, 1, self.embed_dim))
        return inputs + pos_emb

    def get_config(self):
        config = super().get_config()
        config.update({
            "seq_len": self.seq_len,
            "embed_dim": self.embed_dim,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

# 空间位置编码
class SpatialPositionalEncoding(layers.Layer):
    def __init__(self, height, width, embed_dim, **kwargs):
        super().__init__(**kwargs)
        # 使用独立的 h 和 w embedding，最后相加，这样无论 h, w 如何变，只需调整输入维度
        self.row_embed = self.add_weight(shape=(1, height, 1, embed_dim), initializer="random_normal",name='row_embed')
        self.col_embed = self.add_weight(shape=(1, 1, width, embed_dim), initializer="random_normal",name='col_embed')

    def call(self, inputs):
        # inputs: [Batch, Time, H, W, C]
        return inputs + self.row_embed + self.col_embed
    def get_config(self):
        config = super().get_config()
        config.update({
            "height": self.height,
            "width": self.width,
            "embed_dim": self.embed_dim,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class SAM(layers.Layer):

    def __init__(self):
        super().__init__()
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=5,
            padding='same',
            activation='sigmoid'
        )
    def compute_output_shape(self, input_shape):
        return input_shape
    def call(self, x):
        avg_pool = tf.reduce_mean(x,axis=-1,keepdims=True)
        max_pool = tf.reduce_max(x,axis=-1,keepdims=True)
        concat = tf.concat([avg_pool, max_pool],axis=-1)
        attention = self.conv(concat)
        return x * attention


class CBAM(layers.Layer):

    def __init__(self,
                 channels,
                 reduction=8):
        super().__init__()

        self.channels = channels

        # Channel Attention
        self.mlp = tf.keras.Sequential([
            layers.Dense(
                channels // reduction,
                activation='relu'
            ),
            layers.Dense(channels)
        ])

        # Spatial Attention
        self.spatial_conv = layers.Conv2D(
            filters=1,
            kernel_size=7,
            padding='same',
            activation='sigmoid'
        )

    def call(self, x):

        avg_pool = tf.reduce_mean(x,axis=[1,2],keepdims=False)
        max_pool = tf.reduce_max(x,axis=[1,2],keepdims=False)
        avg_attn = self.mlp(avg_pool)
        max_attn = self.mlp(max_pool)
        channel_attn = tf.nn.sigmoid(avg_attn + max_attn)
        channel_attn = tf.reshape(channel_attn,(-1,1,1,self.channels))
        x = x * channel_attn

        avg_pool = tf.reduce_mean(x,axis=-1,keepdims=True)
        max_pool = tf.reduce_max(x,axis=-1,keepdims=True)
        spatial = tf.concat([avg_pool,max_pool],axis=-1)
        spatial_attn = self.spatial_conv(spatial)

        x = x * spatial_attn
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

class TransformerBlockBuilder():
    def __init__(self, num_heads, ff_dim, dropout):
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

    def build(self, x):
        t, h, w, c = x.shape[1], x.shape[2], x.shape[3],x.shape[4]
        identity = x # 保存残差
        x = SpatialPositionalEncoding(height=h,width=w,embed_dim=c)(x)
        sapatial_attn = layers.TimeDistributed(CBAM(c))(x)
        spatial_attn = layers.Dropout(self.dropout)(sapatial_attn)
        output1 = layers.LayerNormalization(epsilon=1e-6)(x + spatial_attn)
        output1 = layers.Reshape((t,h,w,c))(output1)

        time_attn = MultiHeadAttention(
            num_heads=min(self.num_heads,t),
            key_dim=c // self.num_heads,
            attention_axes=(1,)
        )(output1,output1)
        time_attn = layers.Dropout(self.dropout)(time_attn)
        output2 = layers.LayerNormalization(epsilon=1e-6)(output1 + time_attn)
        x = layers.Reshape((t, h, w, c))(output2)

        ffn_output = layers.Dense(self.ff_dim, activation="swish")(x)
        ffn_output = layers.Dropout(self.dropout)(ffn_output)
        ffn_output = layers.Dense(c)(ffn_output)
        ffn_output = layers.Dropout(self.dropout)(ffn_output)
        output3 = layers.LayerNormalization(epsilon=1e-6)(identity + ffn_output)
        return output3
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CombinedModel:
    def __init__(
        self,
        num_submodels,
        num_transformer_blocks,
        dropout,
        ff_dim,
        num_heads,
        output_units,
        input_shape,
        input_steps,
        ele_data
    ):
        self.history = None
        self.model = None
        self.output_units = output_units
        self.dropout = dropout
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_submodels = num_submodels
        self.submodel_builder_cls = CNNLSTMModel
        self.num_transformer_blocks = num_transformer_blocks
        self.input_shape = input_shape
        self.input_steps = input_steps
        self.ele_data = ele_data

    def build_model(self):
        inputs = []
        sub_outputs = []
        ele_shape = self.ele_data.shape

        for modal_idx in range(self.num_submodels):
            builder = self.submodel_builder_cls(self.input_shape, dropout=self.dropout)
            inp, out = builder.build()
            inputs.append(inp)
            sub_outputs.append(out)

        # 地形地势特征编码
        ele_input = Input(shape=(*ele_shape,1))
        ele = layers.Conv2D(16,(1,1),padding='same',activation='swish')(ele_input)
        ele = layers.Conv2D(32,(3,3),padding='same',activation='swish')(ele)
        ele = layers.BatchNormalization()(ele)
        ele = layers.Lambda(
            lambda z: tf.tile(tf.expand_dims(z, axis=1), [1, self.input_shape[0], 1, 1, 1])
        )(ele)
        combine = layers.Concatenate(axis=-1)(sub_outputs + [ele])

        combine = layers.Conv3D(64,(1,1,1), (1,1,1),activation='swish', padding='same')(combine)

        x = layers.BatchNormalization()(combine)

        for idx in range(self.num_transformer_blocks):
            x = TransformerBlockBuilder(
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout,
            ).build(x)

        x = layers.BatchNormalization()(x)

        encoder_output, state_h, state_c = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            return_state=True,
            recurrent_activation="sigmoid",
            dropout=self.dropout,
            recurrent_dropout = self.dropout,
            recurrent_regularizer=tf.keras.regularizers.l2(1e-4),
            kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )(x)

        # last_observed_feature = layers.Lambda(
        #     lambda tensor: tensor[:, -1:, :, :, :]  # 取最后一个时间步，保留维度 (Batch, 1, H, W, 64)
        # )(x)
        # decoder_inputs = layers.Lambda(
        #     lambda t: tf.tile(t, multiples=[1, self.output_units, 1, 1, 1])
        # )(last_observed_feature)
        #
        # decoder_inputs = TemporalPositionalEncoding(
        #     seq_len=self.output_units,
        #     embed_dim=64
        # )(decoder_inputs)
        decoder_inputs = layers.TimeDistributed(
            layers.Conv2D(64, kernel_size=(1, 1), activation='swish')
        )(encoder_output)
        decoder_inputs = (layers.Conv3D(64, kernel_size=(3, 1, 1), strides=(2, 1, 1), padding='same')
                          (decoder_inputs))

        decoder_inputs = TemporalPositionalEncoding(
            seq_len=self.output_units,
            embed_dim=64
        )(decoder_inputs)

        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            dropout=self.dropout,
            recurrent_dropout=self.dropout,
            recurrent_regularizer=tf.keras.regularizers.l2(1e-4),
            kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )(decoder_inputs,initial_state=[state_h,state_c])
        x = layers.TimeDistributed(
            layers.Conv2D(32, (3, 3), padding="same", activation="swish"),
        )(x)
        final_output = layers.TimeDistributed(
            layers.Conv2D(2, (1, 1), padding="same", activation=None, dtype='float32'),
        )(x)

        self.model = models.Model(inputs=inputs + [ele_input], outputs=final_output)
        return self.model

    def get_callbacks(self):
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=1,
            mode="min",
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_delta=1e-4,
            verbose=1,
            min_lr=1e-6,
        )
        return [early_stopping, reduce_lr, TerminateOnNaN()]

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        if self.model is None:
            self.build_model()
        ele_data_expanded = self.ele_data[np.newaxis, ..., np.newaxis]
        ele_train = np.tile(ele_data_expanded, (len(X_train[0]), 1, 1, 1))
        ele_val = np.tile(ele_data_expanded, (len(X_val[0]), 1, 1, 1))

        X_train_with_ele = list(X_train) + [ele_train]
        X_val_with_ele = list(X_val) + [ele_val]
        X_train = [np.asarray(x, dtype=np.float32) for x in X_train_with_ele]
        X_val = [np.asarray(x, dtype=np.float32) for x in X_val_with_ele]
        y_train = np.asarray(y_train, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)

        for name, arrays in (("X_train", X_train), ("X_val", X_val)):
            for idx, arr in enumerate(arrays):
                if not np.isfinite(arr).all():
                    raise ValueError(f"{name}[{idx}] contains NaN or Inf.")
        for name, arr in (("y_train", y_train), ("y_val", y_val)):
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} contains NaN or Inf.")

        def build_dataset(inputs, targets, bs, shuffle=False):
            def gen():
                for idx in range(len(targets)):
                    yield tuple(x[idx] for x in inputs), targets[idx]

            input_signature = tuple(tf.TensorSpec(shape=x.shape[1:], dtype=tf.float32) for x in inputs)
            target_signature = tf.TensorSpec(shape=targets.shape[1:], dtype=tf.float32)
            ds = tf.data.Dataset.from_generator(
                gen,
                output_signature=(input_signature, target_signature),
            )
            if shuffle:
                ds = ds.shuffle(buffer_size=min(len(targets), 1024))
            return ds.batch(bs, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

        train_ds = build_dataset(X_train, y_train, batch_size, shuffle=True)
        val_ds = build_dataset(X_val, y_val, batch_size, shuffle=False)

        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=1,
            callbacks=self.get_callbacks(),
        )
        return self.history


class Evaluate:
    def __init__(self, model, start_lat,end_lat,preprocessor, result_path, x_true, y_true):
        self.start_lat = start_lat
        self.end_lat = end_lat
        self.x_true = x_true
        self.y_true = y_true
        self.preprocessor = preprocessor
        self.scaler_y = preprocessor.scaler_y
        self.y_pred = None
        self.model = model
        self.result_path = result_path
        self.result = None

    def _inverse_transform(self, arr):
        arr_2d = arr.reshape(-1, 2)
        restored = self.scaler_y.inverse_transform(arr_2d).reshape(arr.shape)
        return np.transpose(restored, (0, 1, 4, 2, 3))

    def predict(self, batch_size=256):
        num_samples = len(self.y_true)
        y_pred_list = []
        if len(self.x_true) == self.preprocessor.K:
            ele_data_expanded = np.tile(
                self.preprocessor.load_elevation_data()[np.newaxis, ..., np.newaxis],
                (num_samples, 1, 1, 1)
            )
            self.x_true = list(self.x_true) + [ele_data_expanded]

        for idx in range(0, num_samples, batch_size):
            batch_end = min(idx + batch_size, num_samples)
            batch_x = [x[idx:batch_end] for x in self.x_true]
            try:
                batch_pred = self.model.predict(batch_x, batch_size=batch_size, verbose=1)
                y_pred_list.append(batch_pred)
            except Exception as e:
                print(f"批次预测失败: {e}")
                if batch_size > 1:
                    return self.predict(batch_size=max(1, batch_size // 2))
                raise

        y_pred = np.concatenate(y_pred_list, axis=0)
        self.y_pred = self._inverse_transform(y_pred)
        self.y_true = self._inverse_transform(self.y_true)
        return self.y_pred, self.y_true

    def evaluate(self):
        lat_dim = 11
        lon_dim = 11
        evaluation = GridEvaluate(self.result_path, lat_dim, lon_dim, self.start_lat, self.end_lat)
        rmse, mae, acc = evaluation.evaluate()
        global_rmse = rmse.mean()
        global_mae = mae.mean()
        global_acc = acc.mean()

        print("=" * 50)
        print("全局评估指标:")
        print("=" * 50)
        print(f"Global RMSE: {global_rmse:.4f}")
        print(f"Global MAE:  {global_mae:.4f}")
        print(f"Global ACC:  {global_acc:.4f}")

        print("\n" + "=" * 50)
        print("各预测时次的详细指标:")
        print("=" * 50)
        print("\nRMSE per step:")
        print(", ".join(f"{value:.4f}" for value in rmse))
        print("\nMAE per step:")
        print(", ".join(f"{value:.4f}" for value in mae))
        print("\nACC per step:")
        print(", ".join(f"{value:.4f}" for value in acc))
        self.result = {
            "global_rmse": float(global_rmse),
            "global_mae": float(global_mae),
            "global_acc": float(global_acc),
            "rmse": rmse.tolist(),
            "mae": mae.tolist(),
            "acc": acc.tolist(),
        }
        return rmse, mae, acc


def save_model(model, preprocessor, evaluation, model_save_path, history, test=False):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = model_save_path + ("/" if not test else "/test/") + current_time
    os.makedirs(model_save_path, exist_ok=True)
    weights_path = os.path.join(model_save_path,"checkpoint.ckpt")
    model.save_weights(weights_path,save_format='tf')
    print(f"✓ Model weights saved to {weights_path}")

    model_info = {
        "input_shape": [str(s) for s in model.input_shape] if model.input_shape else None,
        "output_shape": [str(s) for s in model.output_shape] if model.output_shape else None,
        "num_layers": len(model.layers),
        "layer_names": [layer.name for layer in model.layers],
        "layer_types": [layer.__class__.__name__ for layer in model.layers],
    }

    with open(os.path.join(model_save_path, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=4)
    print(f"✓ Model info saved")

     # 保存模型摘要（便于查看结构）
    with open(os.path.join(model_save_path, "model_summary.txt"), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"✓ Model summary saved")
    with open(model_save_path + "/scaler_y.pkl", "wb") as f:
        pickle.dump(preprocessor.scaler_y, f)
    with open(model_save_path + "/scalers_x.pkl", "wb") as f:
        pickle.dump(preprocessor.scalers_x, f)
    with open(model_save_path + "/training_history.pkl", "wb") as f:
        pickle.dump(history.history, f)
    with open(model_save_path + "/evaluation.json", "w", encoding="utf-8") as f:
        json.dump(evaluation.result, f, ensure_ascii=False, indent=4)


def save_evaluate_result_to_txt(save_path, train_metrics, test_metrics):
    txt_path = os.path.join(save_path, "VMD-CNN-LSTM-评估结果.txt")
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H-%M-%S %A')}\n")
        f.write("训练集评估结果:\n")
        for metric, value in train_metrics.items():
            f.write(f"{metric}: {value}\n")
        f.write("测试集评估结果:\n")
        for metric, value in test_metrics.items():
            f.write(f"{metric}: {value}\n")
        f.write("=" * 50 + "\n")
    print("评估结果已保存。")



