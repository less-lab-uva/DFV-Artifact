import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os


def setBoxColors(bp):
    plt.setp(bp['boxes'], color='blue', lw=1.5)
    plt.setp(bp['caps'], color='blue', lw=1.5)
    plt.setp(bp['whiskers'], color='blue', lw=1.5)
    plt.setp(bp['medians'], color='blue', lw=2)


def plot_ssim_full(ssim_list, title=False, ls=8, ax=None):

    if ax == None:
        fig,ax = plt.subplots()

    bp = ax.boxplot(ssim_list)
    setBoxColors(bp)

    ce_count_x, ce_count_y = list(range(1,16)), list()
    for i in range(15):
        ce_count_y.append(len(ssim_list[i]))
    
    ### Ce count plot
    axb = ax.twinx()
    axb.bar(ce_count_x, ce_count_y, color='skyblue', alpha=0.5);
    axb.set_yticks(list(range(0,110,10)))
    axb.set_ylabel("Number of Counter-Examples")
    axb.set_ylim(0,100)

    ax.set_xticks(list(range(1,16)))

    diff_vae = ['$VAE_{'+str(ls)+',1,16}$','$VAE_{'+str(ls)+',1,32}$','$VAE_{'+str(ls)+',1,64}$','$VAE_{'+str(ls)+',1,128}$','$VAE_{'+str(ls)+',1,256}$', '$VAE_{'+str(ls)+',2,16}$','$VAE_{'+str(ls)+',2,32}$','$VAE_{'+str(ls)+',2,64}$','$VAE_{'+str(ls)+',2,128}$','$VAE_{'+str(ls)+',2,256}$','$VAE_{'+str(ls)+',4,16}$','$VAE_{'+str(ls)+',4,32}$','$VAE_{'+str(ls)+',4,64}$','$VAE_{'+str(ls)+',4,128}$','$VAE_{'+str(ls)+',4,256}$']

    ax.set_xticklabels(diff_vae, rotation=65)
    ax.set_ylabel('Mean Reconstruction Similarity')
    ax.set_ylim(0,1)
    ax.set_yticks(np.arange(0,1.1,0.1))
    if title:
        ax.set_title(title)


def plot_mse_full(mse_list, title=True, ls=8, ax=None):

    if ax == None:
        fig,ax = plt.subplots()

    bp = ax.boxplot(mse_list)
    setBoxColors(bp)

    ce_count_x, ce_count_y = list(range(1,16)), list()
    for i in range(15):
        ce_count_y.append(len(mse_list[i]))
    
    ### Ce count plot
    axb = ax.twinx()
    axb.bar(ce_count_x, ce_count_y, color='skyblue', alpha=0.5);
    axb.set_yticks(list(range(0,110,10)))
    axb.set_ylabel("Number of Counter-Examples")
    axb.set_ylim(0,100)

    diff_vae = ['$VAE_{'+str(ls)+',1,16}$','$VAE_{'+str(ls)+',1,32}$','$VAE_{'+str(ls)+',1,64}$','$VAE_{'+str(ls)+',1,128}$','$VAE_{'+str(ls)+',1,256}$', '$VAE_{'+str(ls)+',2,16}$','$VAE_{'+str(ls)+',2,32}$','$VAE_{'+str(ls)+',2,64}$','$VAE_{'+str(ls)+',2,128}$','$VAE_{'+str(ls)+',2,256}$','$VAE_{'+str(ls)+',4,16}$','$VAE_{'+str(ls)+',4,32}$','$VAE_{'+str(ls)+',4,64}$','$VAE_{'+str(ls)+',4,128}$','$VAE_{'+str(ls)+',4,256}$']

    ax.set_xticks(list(range(1,16)))
    ax.set_xticklabels(diff_vae, rotation=65)
    ax.set_ylabel('Mean Reconstruction Error')
    if title:
        ax.set_title(title)
    ax.set_ylim(0,0.13)


def plot_times_full(counter_examples_times, title=False, ls=8, ax=None):

    if ax == None:
        fig,ax = plt.subplots()

    times_list = list()
    for number_layer in [1,2,4]:
        for number_neuron in [16,32,64,128,256]:
            times_list.append(counter_examples_times[number_layer][number_neuron])

    bp = ax.boxplot(times_list)
    setBoxColors(bp)

    ce_count_x, ce_count_y = list(range(1,16)), list()
    for i in range(15):
        ce_count_y.append(len(times_list[i]))
    
    ### Ce count plot
    axb = ax.twinx()
    axb.bar(ce_count_x, ce_count_y, color='skyblue', alpha=0.5);
    axb.set_yticks(list(range(0,110,10)))
    axb.set_ylabel("Number of Counter-Examples")
    axb.set_ylim(0,100)

    ax.set_xticks(list(range(1,16)))

    diff_vae = ['$VAE_{'+str(ls)+',1,16}$','$VAE_{'+str(ls)+',1,32}$','$VAE_{'+str(ls)+',1,64}$','$VAE_{'+str(ls)+',1,128}$','$VAE_{'+str(ls)+',1,256}$', '$VAE_{'+str(ls)+',2,16}$','$VAE_{'+str(ls)+',2,32}$','$VAE_{'+str(ls)+',2,64}$','$VAE_{'+str(ls)+',2,128}$','$VAE_{'+str(ls)+',2,256}$','$VAE_{'+str(ls)+',4,16}$','$VAE_{'+str(ls)+',4,32}$','$VAE_{'+str(ls)+',4,64}$','$VAE_{'+str(ls)+',4,128}$','$VAE_{'+str(ls)+',4,256}$']

    ax.set_xticklabels(diff_vae, rotation=65)
    ax.set_ylabel('Time (seconds)')
    if title:
        ax.set_title(title)

    ax.set_yscale('log')


def plot_ssim_per_ls_full(SSIMs, output_path, ax=None):

    if ax == None:
        fig,ax = plt.subplots()

    all_ce_ssim = list()
    for l, latent_space in enumerate([1,2,4,8,16,32]):
        all_ce_ssim.append(SSIMs[latent_space]['PGD'][0])
        for i in range(1,15):
            all_ce_ssim[l] = np.append(all_ce_ssim[l], SSIMs[latent_space]['PGD'][i])

    bp = ax.boxplot(all_ce_ssim)
    setBoxColors(bp)

    ce_count_x, ce_count_y = list(range(1,7)), list()
    for i in range(6):
        ce_count_y.append(len(all_ce_ssim[i]))
    
    ### Ce count plot
    axb = ax.twinx()
    axb.bar(ce_count_x, ce_count_y, color='skyblue', alpha=0.5);
    axb.set_yticks(list(range(0,1600,100)))
    axb.set_ylabel("Number of Counter-Examples")
    axb.set_ylim(0,1500)

    ax.set_xticks(list(range(1,7)))
    ax.set_xticklabels(['1','2','4','8','16','32'])
    ax.set_xlabel('Latent space')
    ax.set_ylabel('Mean Reconstruction Similarity')
    ax.set_yticks(np.arange(0,1.1,0.1))
    ax.set_ylim(0,1)

    plt.savefig(output_path,bbox_inches='tight')


def plot_std_vs_ssim_full(SSIMs, output_path, ax=None):
    if ax == None:
        fig,ax = plt.subplots()

    stds = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]
    ce_ssim = list()

    for std in stds:
        ce_ssim.append(SSIMs[std])

    bp = ax.boxplot(ce_ssim)
    setBoxColors(bp)

    ce_count_x, ce_count_y = list(range(1,17)), list()
    for index, i in enumerate(range(1,17)):
        ce_count_y.append(len(ce_ssim[index]))
    
    ### Ce count plot
    axb = ax.twinx()
    axb.bar(ce_count_x, ce_count_y, color='skyblue', alpha=0.5);
    axb.set_yticks(list(range(0,110,10)))
    axb.set_ylabel("Number of Counter-Examples")
    axb.set_ylim(-3,100)

    ax.set_xlabel('Radius')
    ax.set_xticks(list(range(1,17)));
    ax.set_xticklabels(stds)
    ax.tick_params(axis='x',labelrotation=45);
    ax.set_yticks(np.arange(0,1.1,0.1))
    ax.set_ylabel('Mean Reconstruction Similarity')
    ax.set_ylim(-0.03,1)

    plt.savefig(output_path,bbox_inches='tight')


def plot_std_vs_times_full(ce_times, output_path, ax=None):
    if ax == None:
        fig,ax = plt.subplots()

    stds = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]
    times = list()

    for std in stds:
        times.append(ce_times[std])

    bp = ax.boxplot(times);
    setBoxColors(bp)

    ce_count_x, ce_count_y = list(range(1,17)), list()
    for i in range(1,17):
        ce_count_y.append(len(ce_times[stds[i-1]]))
    
    ### Ce count plot
    axb = ax.twinx()
    axb.bar(ce_count_x, ce_count_y, color='skyblue', alpha=0.5);
    axb.set_yticks(list(range(0,110,10)))
    axb.set_ylabel("Number of Counter-Examples")
    axb.set_ylim(0,100)

    ax.set_yscale('log')
    ax.set_xlabel('Radius')
    ax.set_xticks(list(range(1,17)));
    ax.set_xticklabels(stds)
    ax.tick_params(axis='x',labelrotation=45);
    ax.set_ylabel('Time (seconds)')

    plt.savefig(output_path,bbox_inches='tight')


def main():

    print("Load data")
    multidim_SSIMs = pickle.load(open("./processed_data/multidim_SSIMs_100_final.pkl", "rb"))
    multidim_MSEs = pickle.load(open("./processed_data/multidim_MSEs_1000_final.pkl", "rb"))
    multidim_counter_examples_times = pickle.load(open("./processed_data/multidim_ce_times_pgd.pkl", "rb"))
    ls_SSIMs = pickle.load(open("./processed_data/ls_SSIMs_100.pkl", "rb"))
    ls_counter_examples_times = pickle.load(open("./processed_data/ls_ce_times.pkl", "rb"))
    

    if not os.path.exists('./images'):
        os.makedirs('./images')

    fig, axs = plt.subplots(6,1)
    fig.set_figheight(24)
    fig.set_figwidth(6)
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    for i, latent_space in enumerate([1,2,4,8,16,32]):
        plot_ssim_full(multidim_SSIMs[latent_space]['PGD'], 'Latent space ' + str(latent_space), ls=latent_space, ax=axs[i])
    plt.savefig('./images/multidim_PGD_ce_ssim.png',bbox_inches='tight')


    fig, axs = plt.subplots(6,1)
    fig.set_figheight(24)
    fig.set_figwidth(6)
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    for i, latent_space in enumerate([1,2,4,8,16,32]):
        plot_mse_full(multidim_MSEs[latent_space]['PGD'], 'Latent space ' + str(latent_space), ls=latent_space, ax=axs[i])
    plt.savefig('./images/multidim_PGD_ce_mse.png',bbox_inches='tight')


    fig, axs = plt.subplots(6,1)
    fig.set_figheight(24)
    fig.set_figwidth(6)
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    for i, latent_space in enumerate([1,2,4,8,16,32]):
        plot_times_full(multidim_counter_examples_times[latent_space], 'Latent space ' + str(latent_space), ls=latent_space, ax=axs[i])
    plt.savefig('./images/multidim_times_PGD.png',bbox_inches='tight')


    plot_ssim_full(multidim_SSIMs[8]['PGD'], ls=8)
    plt.savefig('./images/multidim_models_vs_ssim_ls8.png',bbox_inches='tight')


    plot_std_vs_ssim_full(ls_SSIMs, './images/ls_std_vs_ssim_8_2_256.png')
    plot_std_vs_times_full(ls_counter_examples_times, './images/ls_times_std_logScale.png')
    plot_ssim_per_ls_full(multidim_SSIMs, './images/multidim_ls_vs_ssim_boxplot')


if __name__ == '__main__':
    main()