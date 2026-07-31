import numpy as np
import pandas as pd
import ast
import time

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from models.VST_convlstm import WindDataLoad, CombinedModel, Evaluate, save_model
from evaluation.loss import TotalWindLoss

if __name__ == "__main__":
    vmd_data_path = "data/vmd_data_small/vmd_uv_combined_data.pkl"
    orginal_data_path = "data/processed_combined_small/combined_uv.csv"
    model_save_path = "output/model/HMSTR"

    input_steps = 48
    pred_steps = 24
    val_rate = 0.1
    test_rate = 0.2

    vmd_data_processor = pd.read_pickle(vmd_data_path)
    original_data_processor = pd.read_csv(orginal_data_path)

    vmd_combined_data = np.stack(vmd_data_processor["combined"].values)
    u_data = np.stack(original_data_processor["u100"].apply(lambda x: np.array(ast.literal_eval(x))))
    v_data = np.stack(original_data_processor["v100"].apply(lambda x: np.array(ast.literal_eval(x))))
    original_data = np.stack([u_data, v_data], axis=1)

    wind_speed_loader = WindDataLoad(
        input_steps=input_steps,
        pred_steps=pred_steps,
        normalize=True,
        val_rate=val_rate,
        test_rate=test_rate,
        ele_path="data/processed_ele/elevation.npy"
    )
    X_train, X_val, X_test, y_train, y_val, y_test = wind_speed_loader.generate(
        data_original=original_data,
        data_vmd=vmd_combined_data,
    )
    ele_data = wind_speed_loader.load_elevation_data()
    print(X_test[0][0].shape)

    cross_device_ops = tf.distribute.HierarchicalCopyAllReduce()
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)
    print(f"多GPU策略初始化完成，可用设备数: {strategy.num_replicas_in_sync}")

    base_batch_size = 16
    global_batch_size = base_batch_size * strategy.num_replicas_in_sync

    with strategy.scope():
        combined_model = CombinedModel(
            num_submodels=vmd_combined_data.shape[1],
            num_transformer_blocks=1,
            dropout=0.3,
            ff_dim=64,
            num_heads=1,
            output_units=pred_steps,
            input_shape=X_test[0][0].shape,
            input_steps=input_steps,
            ele_data=ele_data
        )
        model = combined_model.build_model()
        total_loss = TotalWindLoss(
            y_mean=wind_speed_loader.scaler_y.mean_,
            y_scale=wind_speed_loader.scaler_y.scale_,
            calm_threshold=1.0
        )
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4, clipnorm=1.0, clipvalue=0.5),
            loss=total_loss,
            metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
        )

    model.summary()
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} 模型训练开始")

    history = combined_model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=100,
        batch_size=global_batch_size,
    )
    start_lat = 44.75
    end_lat = 47.5
    save_path_train = "outputs/hmstr_train.npz"
    evaluate_train = Evaluate(model,start_lat,end_lat, wind_speed_loader, save_path_train, X_train, y_train)
    pred, true = evaluate_train.predict()
    np.savez(save_path_train, pred=np.array(pred), true=np.array(true))
    train_rmse,train_mae,train_acc=evaluate_train.evaluate()
    train_metrics = {'rmse': train_rmse, 'mae': train_mae, 'acc': train_acc}

    save_path_test = "outputs/hmstr_test.npz"
    evaluate_test = Evaluate(model, start_lat, end_lat, wind_speed_loader, save_path_test, X_test, y_test)
    pred, true = evaluate_test.predict()
    np.savez(save_path_test, pred=np.array(pred), true=np.array(true))
    test_rmse, test_mae, test_acc = evaluate_test.evaluate()
    test_metrics = {"rmse": test_rmse, "mae": test_mae, "acc": test_acc}

    save_model(model, wind_speed_loader, evaluate_test, model_save_path, history, test=False)