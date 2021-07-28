import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os

def setBoxColors(bp):
    plt.setp(bp['boxes'][0], color='blue', linewidth=1.5)
    plt.setp(bp['caps'][0], color='blue', linewidth=1.5)
    plt.setp(bp['caps'][1], color='blue', linewidth=1.5)
    plt.setp(bp['whiskers'][0], color='blue', linewidth=1.5)
    plt.setp(bp['whiskers'][1], color='blue', linewidth=1.5)
    plt.setp(bp['medians'][0], color='blue', linewidth=2)

    plt.setp(bp['boxes'][1], color='darkred', linewidth=1.5)
    plt.setp(bp['caps'][2], color='darkred', linewidth=1.5)
    plt.setp(bp['caps'][3], color='darkred', linewidth=1.5)
    plt.setp(bp['whiskers'][2], color='darkred', linewidth=1.5)
    plt.setp(bp['whiskers'][3], color='darkred', linewidth=1.5)
    plt.setp(bp['medians'][1], color='darkred', linewidth=2)


def plot_img_quality_full(SSIMs, vae_ssim, output_path, ax=None):

    if ax == None:
        fig,ax = plt.subplots()
    
    ax.margins(0,0.1)

    # Quartiles
    ax.axhline(y=np.quantile(vae_ssim, 0.5), ls='-', lw=1, c='black')

    ce_count_x, ce_count_y = list(), list()
    for i, tool in enumerate(['DeepFool','BIM','FGSM','PGD','Neurify','Nnenum','Verinet']):
        group = [SSIMs[tool]['with_decoder'], SSIMs[tool]['without_decoder']]
        if i == 0:
            bp = ax.boxplot(group, positions = [1,2], widths = 0.5)
            ce_count_x.extend([1.0,2.0])
            ce_count_y.extend([len(SSIMs[tool]['with_decoder']), len(SSIMs[tool]['without_decoder'])])
            setBoxColors(bp)
        else:
            bp = ax.boxplot(group, positions = [2*i+i+1,2*i+i+2], widths = 0.5)
            ce_count_x.extend([(1.5+3*i)-0.5, (1.5+3*i)+0.5])
            ce_count_y.extend([len(SSIMs[tool]['with_decoder']), len(SSIMs[tool]['without_decoder'])])
            setBoxColors(bp)

    ### Ce count plot
    axb = ax.twinx()
    color_list = list()
    for i in range(7):
        color_list.extend(['skyblue',  'darksalmon'])
    axb.bar(ce_count_x, ce_count_y, color=color_list, alpha=0.5);
    axb.set_yticks(list(range(0,110,10)))
    axb.set_ylabel("Number of Counter-Examples")
    axb.set_ylim(0,130)

    ax.set_xticks([1.5,4.5,7.5,10.5,13.5,16.5,19.5])
    ax.set_xticklabels(['DeepFool','BIM','FGSM','PGD','Neurify','nnenum','VeriNet'])
    ax.set_ylabel('Mean Reconstruction Similarity')
    ax.set_ylim(0,1.3)
    ax.set_yticks(np.arange(0,1.1,0.1))

    custom_lines = [plt.Line2D([0], [0], color='blue', lw=2),
                    plt.Line2D([0], [0], color='darkred', lw=2)]
    ax.legend(custom_lines, ['With DFV', 'Without DFV'], loc='upper right')
    plt.savefig(output_path,bbox_inches='tight')


def plot_counter_examples_images_rotated(ces_with_metric, output_path):
    fig, axs = plt.subplots(7,10)
    fig.set_figheight(7)
    fig.set_figwidth(10)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for i in range(5):
        for j in range(7):
            if j == 0:
                axs[j,i].set_title("A-"+str(i))
            if i == 0:
                axs[j,i].set_ylabel(['DeepFool', 'BIM', 'FGSM', 'PGD', 'Neurify', 'nnenum', 'VeriNet'][j] + '                ', rotation=0)

            axs[j][i].axes.get_xaxis().set_visible(False)
            axs[j][i].set_yticks([])
            axs[j,i].imshow(ces_with_metric[i][j].reshape(28,28), cmap='gray')

    for i in range(5,10):
        for j in range(7):
            if j == 0:
                axs[j,i].set_title("B-"+str(i-5))

            axs[j][i].axes.get_xaxis().set_visible(False)
            axs[j][i].set_yticks([])
            axs[j,i].imshow(ces_with_metric[i+5][j].reshape(28,28), cmap='gray')

    plt.savefig(output_path,bbox_inches='tight')


def plot_counter_examples_images(ces_with_metric, output_path):
    fig, axs = plt.subplots(10,7)
    fig.set_figheight(10)
    fig.set_figwidth(7)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    for i in range(5):
        for j in range(7):
            if i == 0:
                axs[i,j].set_title(['DeepFool', 'BIM', 'FGSM', 'PGD', 'Neurify', 'Nnenum', 'Verinet'][j])
            if j == 0:
                axs[i,j].set_ylabel("A-"+str(i)+'      ', rotation=0)

            axs[i][j].axes.get_xaxis().set_visible(False)
            axs[i][j].set_yticks([])
            axs[i,j].imshow(ces_with_metric[i][j].reshape(28,28), cmap='gray')

    for i in range(5,10):
        for j in range(7):
            if j == 0:
                axs[i,j].set_ylabel("B-"+str(i-5)+'      ', rotation=0)

            axs[i][j].axes.get_xaxis().set_visible(False)
            axs[i][j].set_yticks([])
            axs[i,j].imshow(ces_with_metric[i+5][j].reshape(28,28), cmap='gray')

    plt.savefig(output_path,bbox_inches='tight')


def plot_ce_times_full(counter_examples_times, output_path, ax=None):

    if ax == None:
        fig,ax = plt.subplots()
    
    tool_vector = ['DeepFool','BIM','FGSM','PGD','Neurify','Nnenum','Verinet']

    ce_count_x, ce_count_y = list(), list()
    for i in range(7):
        if i <= 3:
            group = [counter_examples_times['Falsifiers'][tool_vector[i]]['with_decoder'], counter_examples_times['Falsifiers'][tool_vector[i]]['without_decoder']]
            ce_with_decoder = len(counter_examples_times['Falsifiers'][tool_vector[i]]['with_decoder'])
            ce_without_decoder = len(counter_examples_times['Falsifiers'][tool_vector[i]]['without_decoder'])
            
            ce_count_y.extend([len(counter_examples_times['Falsifiers'][tool_vector[i]]['with_decoder']), len(counter_examples_times['Falsifiers'][tool_vector[i]]['without_decoder'])])
        else:
            group = [counter_examples_times['Verifiers'][tool_vector[i]]['with_decoder'], counter_examples_times['Verifiers'][tool_vector[i]]['without_decoder']]
            ce_with_decoder = len(counter_examples_times['Verifiers'][tool_vector[i]]['with_decoder'])
            ce_without_decoder = len(counter_examples_times['Verifiers'][tool_vector[i]]['without_decoder'])
            
            ce_count_y.extend([len(counter_examples_times['Verifiers'][tool_vector[i]]['with_decoder']), len(counter_examples_times['Verifiers'][tool_vector[i]]['without_decoder'])])

        if i == 0:
            bp = ax.boxplot(group, positions = [1,2], widths = 0.6)
            ce_count_x.extend([1.0,2.0])
            setBoxColors(bp)
        else:
            bp = ax.boxplot(group, positions = [2*i+i+1,2*i+i+2], widths = 0.6)
            ce_count_x.extend([(1.5+3*i)-0.5, (1.5+3*i)+0.5])
            setBoxColors(bp)

    ### Ce count plot
    axb = ax.twinx()
    color_list = list()
    for i in range(7):
        color_list.extend(['skyblue',  'darksalmon'])
    axb.bar(ce_count_x, ce_count_y, color=color_list, alpha=0.5);
    axb.set_yticks(list(range(0,110,10)))
    axb.set_ylabel("Number of Counter-Examples")
    axb.set_ylim(0,100)

    ax.set_yscale('log');

    ax.set_xticks([1.5,4.5,7.5,10.5,13.5,16.5,19.5])
    ax.set_xticklabels(['DeepFool','BIM','FGSM','PGD','Neurify','nnenum','VeriNet'])
    ax.set_ylabel('Time (seconds)')

    custom_lines = [plt.Line2D([0], [0], color='blue', lw=2),
                    plt.Line2D([0], [0], color='darkred', lw=2)]
    ax.legend(custom_lines, ['With DFV', 'Without DFV'], loc='upper left')

    plt.savefig(output_path,bbox_inches='tight')


def main():

    print("Load data")
    SSIMs = pickle.load(open("./processed_data/SSIMs.pkl", "rb"))
    vae_ssim_test_data = pickle.load(open("./processed_data/vae_ssim_test_data.pkl", "rb"))
    ces_with_highest_ssim = pickle.load(open("./processed_data/ces_with_highest_ssim.pkl", "rb"))
    counter_examples_times = pickle.load(open("./processed_data/counter_examples_times.pkl", "rb"))

    if not os.path.exists('./images'):
        os.makedirs('./images')

    plot_img_quality_full(SSIMs, vae_ssim_test_data, './images/ce_ssim_graph_quartiles.png')
    plot_counter_examples_images_rotated(ces_with_highest_ssim['with_decoder'], './images/rotated_counter_examples_with_decoder_SSIM_description.png')
    plot_counter_examples_images_rotated(ces_with_highest_ssim['without_decoder'], './images/rotated_counter_examples_without_decoder_SSIM_description.png')
    plot_counter_examples_images(ces_with_highest_ssim['with_decoder'], './images/counter_examples_with_decoder_SSIM_description.png')
    plot_counter_examples_images(ces_with_highest_ssim['without_decoder'], './images/counter_examples_without_decoder_SSIM_description.png')
    plot_ce_times_full(counter_examples_times, './images/times_to_find_ce_all_logScale.png')


if __name__ == '__main__':
    main()