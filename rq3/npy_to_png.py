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


def measure_rsim(vae, img):
    vae_input_name = vae.get_inputs()[0].name
    errs = []
    for i in range(100):
        recons_img = vae.run(None, {vae_input_name: img})[0]
        err = ssim(recons_img[0, 0], img[0, 0])
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
        "vae+dnn": parse(Path("dronet_fc_vae_decoder.onnx")),
        "gan+dnn": parse(Path("dronet_dcgan_generator.onnx")),
    }
    # vae_model_path = "dronet_fc_vae.onnx"
    vae_model_path = "dronet_vae_vae.onnx"
    dronet_path = Path("benchmark/onnx/dronet.onnx")

    regenerate = False
    if regenerate:
        data = {
            "model": [],
            "falsifier": [],
            "seed": [],
            "property": [],
            "mean_reconstruction_sim": [],
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
            mean_re, stddev_re = measure_rsim(vae, img)
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
            data["mean_reconstruction_sim"].append(mean_re)
            data["stddev_reconstruction_error"].append(stddev_re)
        for key, value in image_grid.items():
            save_image(value, cex_dir / f"{key}_image_grid.png", nrow=5)
        for seed, image_grid_i in enumerate(image_grid_by_seed):
            for key, value in image_grid_i.items():
                save_image(value, cex_dir / f"{key}_image_grid_{seed}.png", nrow=1)
        df = pd.DataFrame(data)
        df.to_csv(cex_dir / "reconstruction_similarity.csv", index=False)

    df = pd.read_csv(cex_dir / "reconstruction_similarity.csv")

    ssims = [
        df[df["model"] == "dnn"].mean_reconstruction_sim.values,
        df[df["model"] == "vae+dnn"].mean_reconstruction_sim.values,
        df[df["model"] == "gan+dnn"].mean_reconstruction_sim.values,
    ]
    plt.figure(figsize=(6, 4))
    sim_ax = plt.gca()
    count_ax = sim_ax.twinx()
    boxes = sim_ax.boxplot(
        ssims, labels=["DroNet", "FC-$VAE_{DroNet}$", "$GAN_{DroNet}$"]
    )
    setBoxColors(boxes)
    x = [b.get_xdata().mean().round() for b in boxes["boxes"]]
    count_ax.bar(
        x, counts, color=["darksalmon", "skyblue", "skyblue"], alpha=0.5, width=0.35
    )
    plt.xlabel("Model")
    sim_ax.set_ylabel("Mean Reconstruction Similarity")
    sim_ax.set_yticks(np.arange(0, 11) / 10)
    count_ax.set_ylabel("Number of Counter-Examples")
    count_ax.set_yticks(np.arange(0, 51, 10))
    without_dfv_legend_line = mlines.Line2D(
        [], [], color="darkred", label="Without DFV"
    )
    with_dfv_legend_line = mlines.Line2D([], [], color="blue", label="With DFV")
    plt.legend(handles=[with_dfv_legend_line, without_dfv_legend_line])
    plt.tight_layout()
    plt.savefig(cex_dir / "ce_ssim_with_counts.png")
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(7, 4))
    sim_boxes = ax[0].boxplot(
        ssims, labels=["DroNet", "FC-$VAE_{DroNet}$", "$GAN_{DroNet}$"]
    )
    setBoxColors(sim_boxes)
    time_boxes = ax[1].boxplot(
        times, labels=["DroNet", "FC-$VAE_{DroNet}$", "$GAN_{DroNet}$"]
    )
    setBoxColors(time_boxes)
    count_ax_0 = ax[0].twinx()
    count_ax_1 = ax[1].twinx()
    x = [b.get_xdata().mean().round() for b in boxes["boxes"]]
    count_ax_0.bar(x, counts, color=["darksalmon", "skyblue", "skyblue"], alpha=0.5)
    count_ax_1.bar(x, counts, color=["darksalmon", "skyblue", "skyblue"], alpha=0.5)
    ax[0].set_ylabel("Mean Reconstruction Similarity")
    ax[0].set_yticks(np.arange(0, 11) / 10)
    ax[1].set_ylabel("Time (seconds)")
    ax[1].set_yticks(np.arange(0, 3601, 300))
    count_ax_1.set_ylabel("Number of Counter-Examples")
    count_ax_1.set_yticks(np.arange(0, 51, 10))
    without_dfv_legend_line = mlines.Line2D(
        [], [], color="darkred", label="Without DFV"
    )
    with_dfv_legend_line = mlines.Line2D([], [], color="blue", label="With DFV")
    plt.legend(
        handles=[with_dfv_legend_line, without_dfv_legend_line], loc="upper right"
    )
    plt.tight_layout()
    plt.savefig(cex_dir / "ce_ssim_time_counts.png")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    sim_ax = ax
    time_ax = sim_ax.twinx()
    count_ax = sim_ax.twinx()
    sim_boxes = sim_ax.boxplot(
        ssims,
        # labels=["DroNet", "FC-$VAE_{DroNet}$", "$GAN_{DroNet}$"],
        positions=[0, 3, 6],
    )
    setBoxColors(sim_boxes)
    time_boxes = time_ax.boxplot(
        times,
        # labels=["DroNet", "FC-$VAE_{DroNet}$", "$GAN_{DroNet}$"],
        positions=[1, 4, 7],
    )
    setBoxColors(time_boxes)
    plt.setp(time_boxes["boxes"], ls="--")
    # x = [b.get_xdata().mean().round() for b in boxes["boxes"]]
    bars = count_ax.bar(
        [2, 5, 8], counts, color=["darksalmon", "skyblue", "skyblue"], alpha=0.5
    )
    for bar in bars:
        x = bar.get_x() + bar.get_width() / 2
        count_ax.text(
            x,
            bar.get_height(),
            f"{bar.get_height()}",
            horizontalalignment="center",
            verticalalignment="bottom",
        )
    ax.set_xticks([1, 4, 7])
    ax.set_xticklabels(["DroNet", "FC-$VAE_{DroNet}$", "$GAN_{DroNet}$"])
    sim_ax.set_ylabel("Mean Reconstruction Similarity")
    sim_ax.set_yticks(np.arange(0, 11) / 10)
    time_ax.set_ylabel("Time (seconds)")
    time_ax.set_yticks(np.arange(0, 3601, 300))
    # count_ax.set_ylabel("Number of Counter-Examples")
    count_ax.set_yticks([])
    without_dfv_legend_line = mlines.Line2D(
        [], [], color="darkred", label="Without DFV"
    )
    with_dfv_legend_line = mlines.Line2D([], [], color="blue", label="With DFV")
    plt.legend(
        handles=[with_dfv_legend_line, without_dfv_legend_line], loc="upper right"
    )
    plt.tight_layout()
    plt.savefig(cex_dir / "ce_ssim_with_time_with_counts.png")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    sim_ax = ax
    time_ax = sim_ax.twinx()
    count_ax = sim_ax.twinx()
    bars = count_ax.bar(
        [0.5, 3.5, 6.5], counts, color=["darksalmon", "skyblue", "skyblue"], alpha=0.5,
    )
    for bar in bars:
        x = bar.get_x() + bar.get_width() / 2
        count_ax.text(
            x,
            bar.get_height(),
            f"{bar.get_height()}",
            horizontalalignment="center",
            verticalalignment="bottom",
        )
    sim_boxes = sim_ax.boxplot(
        ssims,
        # labels=["DroNet", "FC-$VAE_{DroNet}$", "$GAN_{DroNet}$"],
        positions=[0, 3, 6],
    )
    setBoxColors(sim_boxes)
    time_boxes = time_ax.boxplot(
        times,
        # labels=["DroNet", "FC-$VAE_{DroNet}$", "$GAN_{DroNet}$"],
        positions=[1, 4, 7],
    )
    setBoxColors(time_boxes)
    ax.set_xticks([0.5, 3.5, 6.5])
    ax.set_xticklabels(["DroNet", "FC-$VAE_{DroNet}$", "$GAN_{DroNet}$"])
    sim_ax.set_ylabel("Mean Reconstruction Similarity")
    sim_ax.set_yticks(np.arange(0, 11) / 10)
    time_ax.set_ylabel("Time (seconds)")
    time_ax.set_yticks(np.arange(0, 3601, 300))
    # count_ax.set_ylabel("Number of Counter-Examples")
    count_ax.set_yticks([])
    without_dfv_legend_line = mlines.Line2D(
        [], [], color="darkred", label="Without DFV"
    )
    with_dfv_legend_line = mlines.Line2D([], [], color="blue", label="With DFV")
    plt.legend(
        handles=[with_dfv_legend_line, without_dfv_legend_line], loc="upper right"
    )
    plt.tight_layout()
    plt.savefig(cex_dir / "ce_ssim_with_time_with_counts_2.png")
    plt.close()


if __name__ == "__main__":
    main()
