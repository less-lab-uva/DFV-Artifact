import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import onnxruntime
import pandas as pd
import sys
import torch

from collections import defaultdict

from dnnv.nn import parse as dnnv_parse
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import save_image


def measure_rerr(vae, img):
    vae_input_name = vae.get_inputs()[0].name
    errs = []
    for i in range(100):
        recons_img = vae.run(None, {vae_input_name: img})[0]
        # err = ssim(recons_img[0, 0], img[0, 0])
        err = ((recons_img[0, 0] - img[0, 0]) ** 2).mean()
        errs.append(err)
    return np.mean(errs), np.std(errs)


def parse(path: Path):
    def model(x, path=path):
        dnn = onnxruntime.InferenceSession(str(path))
        input_name = dnn.get_inputs()[0].name
        return dnn.run(None, {input_name: x})[0]

    return model


def parse_logs(path: Path):
    results = {"model": [], "property_id": [], "seed": [], "result": [], "time": []}
    for log in path.iterdir():
        with open(log) as logfile:
            model = property_id = seed = result = time = None
            for line in logfile:
                line = line.strip()
                if "PGD" in line:
                    problem_params = line.split()
                    model = problem_params[0].lower()
                    property_id = int(problem_params[2])
                    seed = int(problem_params[3].split("=")[-1])
                elif line.startswith("result"):
                    result = line.split(":")[-1].strip()
                elif line.startswith("falsification time"):
                    time = float(line.split(":")[-1].strip())
                if time is not None:
                    print(model, property_id, seed, result, time)
                    results["model"].append(model)
                    results["property_id"].append(property_id)
                    results["seed"].append(seed)
                    results["result"].append(result)
                    results["time"].append(time)
                    model = property_id = seed = result = time = None
    df = pd.DataFrame(results)
    print(df)
    return df


def setBoxColors(bp):
    plt.setp(bp["boxes"][:1], color="darkred", lw=1.5)
    plt.setp(bp["caps"][:2], color="darkred", lw=1.5)
    plt.setp(bp["whiskers"][:2], color="darkred", lw=1.5)
    plt.setp(bp["medians"][:1], color="darkred", lw=2)

    plt.setp(bp["boxes"][1:], color="blue", lw=1.5)
    plt.setp(bp["caps"][2:], color="blue", lw=1.5)
    plt.setp(bp["whiskers"][2:], color="blue", lw=1.5)
    plt.setp(bp["medians"][1:], color="blue", lw=2)


def main():
    cex_dir = Path(sys.argv[1])
    (cex_dir / "png").mkdir(exist_ok=True, parents=True)

    log_dir = Path(sys.argv[2])
    results_df = parse_logs(log_dir)
    results_df.to_csv(cex_dir / "results.csv", index=False)

    times = [
        results_df[results_df["model"] == "dnn"].time.values,
        results_df[results_df["model"] == "vae+dnn"].time.values,
        results_df[results_df["model"] == "gan+dnn"].time.values,
    ]
    counts = [
        results_df[results_df["model"] == "dnn"].time.count(),
        results_df[results_df["model"] == "vae+dnn"].time.count(),
        results_df[results_df["model"] == "gan+dnn"].time.count(),
    ]
    plt.figure(figsize=(6, 4))
    time_ax = plt.gca()
    count_ax = time_ax.twinx()
    boxes = time_ax.boxplot(
        times, labels=["DroNet", "FC-$VAE_{DroNet}$", "$GAN_{DroNet}$"]
    )
    setBoxColors(boxes)
    x = [b.get_xdata().mean().round() for b in boxes["boxes"]]
    count_ax.bar(
        x, counts, color=["darksalmon", "skyblue", "skyblue"], alpha=0.5, width=0.35
    )
    plt.xlabel("Model")
    time_ax.set_ylabel("Time (seconds)")
    time_ax.set_yticks(np.arange(0, 3601, 300))
    count_ax.set_ylabel("Number of Counter-Examples")
    count_ax.set_yticks(np.arange(0, 51, 10))
    without_dfv_legend_line = mlines.Line2D(
        [], [], color="darkred", label="Without DFV"
    )
    with_dfv_legend_line = mlines.Line2D([], [], color="blue", label="With DFV")
    plt.legend(handles=[with_dfv_legend_line, without_dfv_legend_line])
    plt.tight_layout()
    plt.savefig(cex_dir / "time_plot_with_counts.png")
    plt.close()

    generative_models = {
        "vae+dnn": parse(Path("models/dronet_fc_vae_decoder.onnx")),
        "gan+dnn": parse(Path("models/dronet_dcgan_generator.onnx")),
    }
    # vae_model_path = "models/dronet_fc_vae.onnx"
    vae_model_path = "models/dronet_vae_vae.onnx"
    dronet_path = Path("benchmark/onnx/dronet.onnx")

    regenerate = False
    if regenerate:
        data = {
            "model": [],
            "falsifier": [],
            "seed": [],
            "property": [],
            "mean_reconstruction_error": [],
            "stddev_reconstruction_error": [],
        }
        falsifiers = ["pgd"]
        nfalsifiers = len(falsifiers)
        nprops = 10
        nseeds = 5
        image_grid = defaultdict(
            lambda: torch.zeros(nseeds * nfalsifiers * nprops, 1, 200, 200)
        )
        image_grid_by_seed = [
            defaultdict(lambda: torch.zeros(nfalsifiers * nprops, 1, 200, 200))
            for i in range(nseeds)
        ]
        vae = onnxruntime.InferenceSession(vae_model_path)
        dnn = dnnv_parse(dronet_path)[2:]
        for filename in sorted((cex_dir / "npy").iterdir()):
            model, falsifier, prop, seed = filename.stem.split("_")
            if (
                results_df[
                    (results_df["model"] == model)
                    & (results_df["property_id"] == int(prop))
                    & (results_df["seed"] == int(seed))
                ]
                .count()["model"]
                .item()
                == 0
            ):
                print(
                    model,
                    falsifier,
                    prop,
                    seed,
                    "generated a counter example after timeout",
                )
                continue
            identifier = (
                falsifiers.index(falsifier) * nseeds * nprops
                + int(prop) * nseeds
                + int(seed)
            )
            img = np.load(filename)
            if not filename.stem.startswith("dnn"):
                img = generative_models[model](img)
            mean_re, stddev_re = measure_rerr(vae, img)
            y = dnn(img)
            print(filename, ":", y, y[0, 0] * 180 / np.pi, (mean_re, stddev_re))
            img = torch.from_numpy(img)
            image_grid[model][identifier] = img[0]
            image_grid_by_seed[int(seed)][model][
                falsifiers.index(falsifier) * nprops + int(prop)
            ] = img[0]
            save_image(img, cex_dir / "png" / f"{filename.stem}.png")
            data["model"].append(model)
            data["falsifier"].append(falsifier)
            data["seed"].append(int(seed))
            data["property"].append(int(prop))
            data["mean_reconstruction_error"].append(mean_re)
            data["stddev_reconstruction_error"].append(stddev_re)
        for key, value in image_grid.items():
            save_image(value, cex_dir / f"{key}_image_grid.png", nrow=5)
        for seed, image_grid_i in enumerate(image_grid_by_seed):
            for key, value in image_grid_i.items():
                save_image(value, cex_dir / f"{key}_image_grid_{seed}.png", nrow=1)
                save_image(
                    value, cex_dir / f"{key}_image_grid_{seed}_horizontal.png", nrow=10
                )
        df = pd.DataFrame(data)
        df.to_csv(cex_dir / "reconstruction_error.csv", index=False)

    df = pd.read_csv(cex_dir / "reconstruction_error.csv")

    mses = [
        df[df["model"] == "dnn"].mean_reconstruction_error.values,
        df[df["model"] == "vae+dnn"].mean_reconstruction_error.values,
        df[df["model"] == "gan+dnn"].mean_reconstruction_error.values,
    ]
    plt.figure(figsize=(6, 4))
    mse_ax = plt.gca()
    count_ax = mse_ax.twinx()
    boxes = mse_ax.boxplot(
        mses, labels=["DroNet", "FC-$VAE_{DroNet}$", "$GAN_{DroNet}$"]
    )
    setBoxColors(boxes)
    x = [b.get_xdata().mean().round() for b in boxes["boxes"]]
    count_ax.bar(
        x, counts, color=["darksalmon", "skyblue", "skyblue"], alpha=0.5, width=0.35
    )
    plt.xlabel("Model")
    mse_ax.set_ylabel("Mean Reconstruction Error")
    # mse_ax.set_yticks(np.arange(0, 11) / 10)
    count_ax.set_ylabel("Number of Counter-Examples")
    count_ax.set_yticks(np.arange(0, 51, 10))
    without_dfv_legend_line = mlines.Line2D(
        [], [], color="darkred", label="Without DFV"
    )
    with_dfv_legend_line = mlines.Line2D([], [], color="blue", label="With DFV")
    plt.legend(handles=[with_dfv_legend_line, without_dfv_legend_line])
    plt.tight_layout()
    plt.savefig(cex_dir / "ce_mse_with_counts.png")
    plt.close()


if __name__ == "__main__":
    main()
