import numpy as np
import tensorflow as tf

def calculate_WDFA(result_path,
                   calm_thresholds=(0.5,1.0,2.0),
                   alpha=(22.5,45,90),
                   output_step=24,
                   ):

    result = np.load(result_path)

    pred = result["pred"]
    true = result["true"]

    results={}
    for threshold in calm_thresholds:
        results[threshold] = {
            f"WDFA_{a}": [] 
            for a in alpha
        }

        results[threshold]["circular_MAE"] = []
        results[threshold]["valid_count"] = []

    for i in range(output_step):
        pred_u = pred[:, i, 0]
        pred_v = pred[:, i, 1]
        true_u = true[:, i, 0]
        true_v = true[:, i, 1]

        # 计算风向
        pred_dir = tf.atan2(pred_v, pred_u)
        true_dir = tf.atan2(true_v, true_u)

        pred_dir_deg = pred_dir * 180.0 / np.pi
        true_dir_deg = true_dir * 180.0 / np.pi
        pred_dir_deg = tf.cast(pred_dir_deg, tf.float32)
        true_dir_deg = tf.cast(true_dir_deg, tf.float32)
        # 角度误差计算
        wind_dir_diff = tf.abs(pred_dir_deg - true_dir_deg)
        wind_dir_diff = tf.where(wind_dir_diff > 180.0, 360.0 - wind_dir_diff, wind_dir_diff)
        wind_dir_diff = wind_dir_diff.numpy()

        true_speed = np.sqrt(true_u ** 2 + true_v ** 2)

        for threshold in calm_thresholds:

            # 只保留真实风速大于阈值区域
            mask = true_speed >= threshold


            valid_count = np.sum(mask)

            results[threshold]["valid_count"].append(
                valid_count / mask.size * 100
            )

            for a in alpha:
                
                # 计算WDFA
                count = np.count_nonzero(wind_dir_diff[mask] < a)
                WDFA = (count / valid_count) * 100 if valid_count else np.nan
                results[threshold][f"WDFA_{a}"].append(WDFA)

            circular_mae = (
                np.mean(wind_dir_diff[mask]) if valid_count else np.nan
            )

            results[threshold]["circular_MAE"].append(
                circular_mae
            )

    return results

if __name__=="__main__":
    result_path = r"D:/LWQ/wind-forecast-main//output/vst-conv_test.npz"
    results = calculate_WDFA(result_path,
                          calm_thresholds=(0.5,1.0,2.0),
                          alpha=(22.5,45,90),
                          output_step=24)
    print(
        "\n===== WDFA hourly results ====="
    )

    threshold = 1.0


    for hour in range(24):

        print(
            f"Hour {hour+1:02d}: "
            f"WDFA22.5={results[threshold]['WDFA_22.5'][hour]:.2f}%  "
            f"WDFA45={results[threshold]['WDFA_45'][hour]:.2f}%  "
            f"WDFA90={results[threshold]['WDFA_90'][hour]:.2f}%  "
            f"MAE={results[threshold]['circular_MAE'][hour]:.2f}°"
        )
