"""
Script that runs visual-odometer with different configurations such as different image-registration algorithms.
"""

# Default packages:
import time
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Internal modules:
from visual_odometer import VisualOdometer
from utils import create_datasets, get_available_datasets

# Path where dataset is located:
DATA_ROOT = "../datasets/"

# Benchmark configs:
PLOT_RESULTS = True
VERBOSE = True

# Create data-structure which the results will be kept on:
results = {
    ("Dataset ID", "", ""): [],
    ("Model ID", "", ""): [],
    ("Method", "", ""): [],

    ("Elapsed time / (ms)", "mean", ""): [],
    ("Elapsed time / (ms)", "max", ""): [],
    ("Elapsed time / (ms)", "min", ""): [],

    ("Point wise error / (pixels)", "x-axis", "mean"): [],
    ("Point wise error / (pixels)", "x-axis", "max"): [],
    ("Point wise error / (pixels)", "x-axis", "min"): [],

    ("Point wise error / (pixels)", "y-axis", "mean"): [],
    ("Point wise error / (pixels)", "y-axis", "max"): [],
    ("Point wise error / (pixels)", "y-axis", "min"): [],

    ("Point wise error / (pixels)", "combined", "mean"): [],
    ("Point wise error / (pixels)", "combined", "max"): [],
    ("Point wise error / (pixels)", "combined", "min"): [],

    ("Cumulative error / (pixels)", "x-axis", ""): [],
    ("Cumulative error / (pixels)", "y-axis", ""): [],
    ("Cumulative error / (pixels)", "comb", ""): [],
}

# Hard-coded calibration results:
mm_per_px = [
    21.02,
    21.50,
    21.89,
    22.32,
    22.87,
    23.50,
    23.94,
    24.56,
    25.06,
    25.73,
    26.42,
    27.15,
    27.90
]

# Result tempalte:
template_estimates = {
    "ID": [],
    "Method": [],
    "x-axis": [],
    "y-axis": [],
}

if __name__ == "__main__":
    methods = ["svd", "projection-svd", "phase-correlation", "phase-amplified-correlation"]

    # Create the dataset:
    create_datasets(DATA_ROOT, overwrite=False)
    valid_datasets = get_available_datasets(DATA_ROOT, verbose=VERBOSE)

    for jj, dataset in enumerate(valid_datasets):
        print(f"Currently processing {dataset} dataset. Progress: {jj}/{len(dataset)} ({jj/len(dataset):.1%})", end='\r')
        data = pd.read_pickle(DATA_ROOT + dataset + "/dataset.pkl")
        estimates = copy.deepcopy(template_estimates)
        for ii, method in enumerate(methods):

            odometer = VisualOdometer(img_shape=data["img"][0].shape, xres=-1, yres=-1)
            odometer.config_displacement_estimation(method)

            t0 = time.time()
            elapsed_time = []

            nrows = len(data)
            delta_x_pred, delta_y_pred = [], []
            x_pred, y_pred = [0], [0]
            x_true, y_true = [0], [0]

            # Run displacement estimation loop:
            for i in range(nrows):
                order = data.iloc[i]['order']
                img = data.iloc[i]['img']

                ti0 = time.time()
                odometer.feed_image(img)

                delta_xi, delta_yi = odometer.get_displacement()
                ti1 = time.time()

                delta_xi, delta_yi = delta_xi / mm_per_px[jj], delta_yi / mm_per_px[jj]  # Shift in millimeter [px] / [px/mm] -> [mm]

                elapsed_time.append((ti1 - ti0) * 1e3)
                delta_x_pred.append(delta_xi)
                delta_y_pred.append(delta_yi)
                x_pred.append(delta_xi + x_pred[-1])
                y_pred.append(delta_yi + y_pred[-1])
                x_true.append(data["delta_x"][i] + x_true[-1])
                y_true.append(data["delta_y"][i] + y_true[-1])

            x_true, y_true = np.array(x_true), np.array(y_true)
            x_pred, y_pred = np.array(x_pred), np.array(y_pred)

            # Compute metrics:
            pointwise_x_error = x_true - x_pred
            pointwise_y_error = y_true - y_pred
            pointwise_comb_error = np.sqrt(pointwise_x_error ** 2 + pointwise_y_error ** 2)

            cumulative_x_error = x_pred[-1] - x_true[-1]
            cumulative_y_error = y_pred[-1] - y_true[-1]
            cumulative_comb_error = np.sqrt(cumulative_x_error ** 2 + cumulative_y_error ** 2)

            # Convert results into dataframe:
            ii += 1
            results[("Dataset ID", "", "")].append(jj + 1)
            results[("Model ID", "", "")].append(ii)
            results[("Method", "", "")].append(method)

            # Elapsed time
            results[("Elapsed time / (ms)", "mean", "")].append(np.mean(elapsed_time))
            results[("Elapsed time / (ms)", "max", "")].append(np.max(elapsed_time))
            results[("Elapsed time / (ms)", "min", "")].append(np.min(elapsed_time))

            # Pointwise error (x-axis)
            results[("Point wise error / (pixels)", "x-axis", "mean")].append(np.mean(pointwise_x_error))
            results[("Point wise error / (pixels)", "x-axis", "max")].append(np.max(pointwise_x_error))
            results[("Point wise error / (pixels)", "x-axis", "min")].append(np.min(pointwise_x_error))

            # Pointwise error (y-axis)
            results[("Point wise error / (pixels)", "y-axis", "mean")].append(np.mean(pointwise_y_error))
            results[("Point wise error / (pixels)", "y-axis", "max")].append(np.max(pointwise_y_error))
            results[("Point wise error / (pixels)", "y-axis", "min")].append(np.min(pointwise_y_error))

            # Pointwise error (combined)
            results[("Point wise error / (pixels)", "combined", "mean")].append(np.mean(pointwise_comb_error))
            results[("Point wise error / (pixels)", "combined", "max")].append(np.max(pointwise_comb_error))
            results[("Point wise error / (pixels)", "combined", "min")].append(np.min(pointwise_comb_error))

            # Cumulative errors
            results[("Cumulative error / (pixels)", "x-axis", "")].append(cumulative_x_error)
            results[("Cumulative error / (pixels)", "y-axis", "")].append(cumulative_y_error)
            results[("Cumulative error / (pixels)", "comb", "")].append(cumulative_comb_error)

            #
            estimates["ID"].append(ii)
            estimates["Method"].append(method)
            estimates["x-axis"].append(x_pred)
            estimates["y-axis"].append(y_pred)

        #
        if PLOT_RESULTS:
            import matplotlib
            matplotlib.use("TkAgg")

            fig = plt.figure(figsize=(8, 8))
            plt.title(f"Dataset: {dataset}")

            colors = ['k', 'g', 'purple', 'cyan']
            for i, method in enumerate(methods):
                plt.plot(estimates['x-axis'][i], estimates['y-axis'][i], '-o', color=colors[i], label=method + f"\n elapsed-time: {results[("Elapsed time / (ms)", "mean", "")][len(methods) * jj + i]:.2f} s", markersize=3)
            plt.plot([80], [80], 'or', markersize=3)
            plt.plot(x_true, y_true, '-o', markersize=3, alpha=.5, color='r', label='Ideal')
            plt.xlabel("x-axis / (pixels)")
            plt.ylabel("x-axis / (pixels)")
            plt.axis("equal")
            plt.xlim(-5, 100)
            plt.ylim(-5, 100)
            plt.legend()
            plt.savefig(DATA_ROOT + dataset + "_trajectory.png", dpi=300)
            plt.show()

    #%% Convert dict to DataFrame (single row per run if lists are same length)


    if VERBOSE:
        df = pd.DataFrame(results)

        # Convert columns to MultiIndex
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Category", "Axis", "Metric"])

        print(
            df
            .round(4)
            .to_string(index=False)
        )