import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import torch
import pickle
import time
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import warnings; warnings.simplefilter('ignore')
import sys
sys.path.append('./common')
import ssim
sys.path.append('./saved_models')
from train_models import Vae1, Decoder1, Encoder1, Vae2, Decoder2, Encoder2, Vae4, Decoder4, Encoder4, Network
from train_vae_mrs import VaeMrs, EncoderMrs, DecoderMrs


def load_models():
    vaes = {}
    decoders = {}
    for latent_space in [1,2,4,8,16,32]:
        vaes[latent_space] = dict()
        decoders[latent_space] = dict()
        for number_layer in [1,2,4]:
            vaes[latent_space][number_layer] = dict()
            decoders[latent_space][number_layer] = dict()
            for number_neuron in [16,32,64,128,256]:
                model_path = './saved_models/latent_space'+str(latent_space)+'/number_layer'+str(number_layer)+'/vae/vae'+str(number_neuron)

                if number_layer == 1:
                    model = Vae1(latent_space, number_neuron)
                elif number_layer == 2:
                    model = Vae2(latent_space, number_neuron)
                elif number_layer == 4:
                    model = Vae4(latent_space, number_neuron)
                
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

                vaes[latent_space][number_layer][number_neuron] = model
                decoders[latent_space][number_layer][number_neuron] = model.decoder
    return vaes, decoders


def decode_ce(ce, variational_decoder):
    decoded_img = variational_decoder(torch.tensor(ce))
    decoded_img = decoded_img.detach().numpy()
    return decoded_img


def scrap_counter_examples_multidim(decoders):
    counter_examples = {}
    for latent_space in [1,2,4,8,16,32]:
        counter_examples[latent_space] = dict()
        for number_layer in [1,2,4]:
            counter_examples[latent_space][number_layer] = dict()
            for number_neuron in [16,32,64,128,256]:
                counter_examples[latent_space][number_layer][number_neuron] = dict()
                counter_examples[latent_space][number_layer][number_neuron]['PGD'] = list()
                for run in range(5):
                    for prop in range(1,3):
                        ce_path = './output/multidim_study/latent_space'+str(latent_space)+'/number_layer'+str(number_layer)+'/number_neuron'+str(number_neuron)+                            '/PGD/run'+str(run)+'/property'+str(prop)+'/counter_examples/'
                        for ce_name in sorted(os.listdir(ce_path)):
                            ce = np.load(ce_path + ce_name)
                            counter_examples[latent_space][number_layer][number_neuron]['PGD'].append(
                                decode_ce(ce, decoders[latent_space][number_layer][number_neuron]))

    return counter_examples


def mse_1000(vae, images):
    all_mse_list = list()
    for i in range(1000):
        decoded_images,_,_ = vae(images.view(images.shape[0],784))
        mse_list = ((images.view(images.shape[0],784) - decoded_images)**2).detach().numpy().mean(axis=1)
        all_mse_list.append(mse_list)
    all_mse_list = np.array(all_mse_list)
    return all_mse_list.mean(axis=0), all_mse_list.mean()


def scrap_counter_examples_times_multidim():
    counter_examples_times = dict()
    types = ['with_decoder', 'without_decoder']

    sbatch_outputs = ["test1.err","test1.output","test2.err","test2.output"]

    for latent_space in [1,2,4,8,16,32]:
        counter_examples_times[latent_space] = dict()
        for number_layer in [1,2,4]:
            counter_examples_times[latent_space][number_layer] = dict()
            for number_neuron in [16,32,64,128,256]:
                counter_examples_times[latent_space][number_layer][number_neuron] = list()
                for run in range(5):
                    for prop in range(1,3):
                        summary_path = './output/multidim_study/latent_space'+str(latent_space)+'/number_layer'+str(number_layer)+'/number_neuron'+                            str(number_neuron)+'/PGD/run'+str(run)+'/property'+str(prop)+'/result_summary.md'
                        
                        summary = open(summary_path, 'r')
                        aux_status = ''

                        for l in summary:
                            if '##Status:' in l:
                                aux_status = l.split(' ')[-1].split('\n')[0]
                            if '###Time:' in l:
                                if aux_status == 'sat':
                                    counter_examples_times[latent_space][number_layer][number_neuron].append(float(l.split(' ')[-1]))
                                aux_status = ''
                        summary.close()
    
    return counter_examples_times


def scrap_counter_examples_ls(decoder):
    counter_examples = dict()
    for std in [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]:   # output
        counter_examples[std] = list()
        std_folder = "std"+str(std)
        for run_folder in sorted(os.listdir('./output/ls_study/'+std_folder)):   # std
            for property_folder in sorted(os.listdir('./output/ls_study/'+std_folder+'/'+run_folder)):   # property
                ce_path = './output/ls_study/'+std_folder+'/'+run_folder+'/'+property_folder+'/counter_examples/'
                for ce_name in sorted(os.listdir(ce_path)): # counter_examples
                    ce = np.load(ce_path + ce_name)
                    counter_examples[std].append(decode_ce(ce,decoder))
    return counter_examples


def scrap_times_ls():
    ce_times = dict()
    for std in [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]:    #output
        times = list()
        std_folder = "std"+str(std)
        for run_folder in sorted(os.listdir('./output/ls_study/'+std_folder)):   # std
            for property_folder in sorted(os.listdir('./output/ls_study/'+std_folder+'/'+run_folder)):   # property
                r_path = './output/ls_study/'+std_folder+'/'+run_folder+'/'+property_folder+'/result_summary.md'
                summary = open(r_path, 'r')

                aux_status = ''

                for l in summary:
                    if '##Status:' in l:
                        aux_status = l.split(' ')[-1].split('\n')[0]
                    if '###Time:' in l:
                        if aux_status == 'sat':
                            times.append(float(l.split(' ')[-1]))
                        aux_status = ''
                summary.close()
                
        ce_times[std] = np.array(times)
    return ce_times


def main():

    # # Test data
    # test_data = datasets.FashionMNIST(
    #     root='.data/FashionMNIST',
    #     train=False,
    #     download=True,
    #     transform=transforms.Compose([transforms.ToTensor()])
    # )

    # # Test data loader
    # test_loader = DataLoader(
    #     test_data,
    #     batch_size=10000,
    #     shuffle=False
    # )

    # dataiter = iter(test_loader)
    # test, test_labels = dataiter.next()


    ### Load models
    print("Load models")
    vaes, decoders = load_models();

    network = Network()
    network.load_state_dict(torch.load('./saved_models/network', map_location=torch.device('cpu')));

    print("Load VAE MRS")
    vae_mrs = VaeMrs(100)
    vae_mrs.load_state_dict(torch.load('./saved_models/vae_mrs', map_location=torch.device('cpu')));


    ###----------     MULTIDIM STUDY    ----------###
    print("\nMULTIDIM STUDY\n")
    if not os.path.exists('./processed_data'):
        os.makedirs('./processed_data')


    ### Load counter-examples
    print("Load counter-examples")
    counter_examples = scrap_counter_examples_multidim(decoders)

    f = open("./processed_data/multidim_counter_examples.pkl","wb")
    pickle.dump(counter_examples,f)
    f.close()


    ### Calculate MSE of counter-examples
    print("\nCalculate MSE of counter-examples")
    MSEs = dict()
    for latent_space in [1,2,4,8,16,32]:
        print("Latent space: "+str(latent_space))
        MSEs[latent_space] = dict()
        MSEs[latent_space]['PGD'] = list()
        for number_layer in [1,2,4]:
            for number_neuron in [16,32,64,128,256]:
                print("# Layer: "+str(number_layer)+' - # Neurons: '+str(number_neuron), end =" ")
                start_t = time.time()

                mse_aux, _ = mse_1000(vae_mrs, torch.tensor(counter_examples[latent_space][number_layer][number_neuron]['PGD']))
                MSEs[latent_space]['PGD'].append(mse_aux)
                
                end_t = time.time()
                duration = end_t - start_t
                print("Duration: "+str(duration))

    f = open("./processed_data/multidim_MSEs_1000_final.pkl","wb")
    pickle.dump(MSEs,f)
    f.close()


    # ### Calculate models MSE on test data
    # print("\nCalculate models MSE on test data")
    # models_mse = list()
    # for latent_space in [1,2,4,8,16,32]:
    #     print("Latent space: "+str(latent_space))
    #     sub_models_mse = list()
    #     for number_layer in [1,2,4]:
    #         for number_neuron in [16,32,64,128,256]:
    #             print("# Layer: "+str(number_layer)+' - # Neurons: '+str(number_neuron), end =" ")
    #             start_t = time.time()

    #             aux_mse, aux_mse_mean = mse_1000(vaes[latent_space][number_layer][number_neuron], test.reshape(10000,784))
    #             sub_models_mse.append(aux_mse)

    #             end_t = time.time()
    #             duration = end_t - start_t
    #             print("Duration: "+str(duration))
    #     models_mse.append(sub_models_mse)

    # f = open("./processed_data/multidim_models_mse_final.pkl","wb")
    # pickle.dump(models_mse,f)
    # f.close()


    ### Calculate SSIM of counter-examples
    print("\nCalculate SSIM of counter-examples")
    SSIMs = dict()
    for latent_space in [1,2,4,8,16,32]:
        print("Latent space: "+str(latent_space))
        SSIMs[latent_space] = dict()
        SSIMs[latent_space]['PGD'] = list()
        for number_layer in [1,2,4]:
            for number_neuron in [16,32,64,128,256]:
                print("# Layer: "+str(number_layer)+' - # Neurons: '+str(number_neuron), end =" ")
                start_t = time.time()

                mse_aux, _ = ssim.ssim_100(vae_mrs, torch.tensor(counter_examples[latent_space][number_layer][number_neuron]['PGD']))
                SSIMs[latent_space]['PGD'].append(mse_aux.mean(axis=0))
                
                end_t = time.time()
                duration = end_t - start_t
                print("Duration: "+str(duration))

    f = open("./processed_data/multidim_SSIMs_100_final.pkl","wb")
    pickle.dump(SSIMs,f)
    f.close()


    # ### Calculate models SSIM on test data
    # print("\nCalculate models SSIM on test data")
    # models_ssim = list()
    # for latent_space in [1,2,4,8,16,32]:
    #     print("Latent space: "+str(latent_space))
    #     sub_models_ssim = list()
    #     for number_layer in [1,2,4]:
    #         for number_neuron in [16,32,64,128,256]:
    #             print("# Layer: "+str(number_layer)+' - # Neurons: '+str(number_neuron), end =" ")
    #             start_t = time.time()

    #             aux_ssim, aux_ssim_mean = ssim.ssim_100(vaes[latent_space][number_layer][number_neuron], test.reshape(10000,784))
    #             sub_models_ssim.append(aux_ssim.mean(axis=0))

    #             end_t = time.time()
    #             duration = end_t - start_t
    #             print("Duration: "+str(duration))
    #     models_ssim.append(sub_models_ssim)

    # f = open("./processed_data/multidim_models_ssim_final.pkl","wb")
    # pickle.dump(models_ssim,f)
    # f.close()


    ### Load counter-examples times
    print("\nLoad counter-examples times")
    counter_examples_times_pgd = scrap_counter_examples_times_multidim()

    f = open("./processed_data/multidim_ce_times_pgd.pkl","wb")
    pickle.dump(counter_examples_times_pgd,f)
    f.close()


    ###----------     LS STUDY    ----------###
    print("\nLS STUDY\n")
    

    ### Load counter-examples
    print("Load counter-examples")
    counter_examples = scrap_counter_examples_ls(decoders[8][2][256])

    f = open("./processed_data/ls_counter_examples.pkl","wb")
    pickle.dump(counter_examples,f)
    f.close()


    ### Calculate MSE of counter-examples
    print("\nCalculate MSE of counter-examples")
    MSEs = dict()
    for std in [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]:
        print("STD "+str(std), end =" ")

        start_t = time.time()

        mse_aux, _ = mse_1000(vae_mrs, torch.tensor(counter_examples[std]))
        MSEs[std] = mse_aux
        
        end_t = time.time()
        duration = end_t - start_t
        print("Duration: "+str(duration))

    f = open("./processed_data/ls_MSEs_1000.pkl","wb")
    pickle.dump(MSEs,f)
    f.close()


    ### Calculate SSIM of counter-examples
    print("\nCalculate SSIM of counter-examples")
    SSIMs = dict()
    for std in [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]:
        print("STD "+str(std), end =" ")

        start_t = time.time()

        mse_aux, _ = ssim.ssim_100(vae_mrs, torch.tensor(counter_examples[std]))
        SSIMs[std] = mse_aux.mean(axis=0)
        
        end_t = time.time()
        duration = end_t - start_t
        print("Duration: "+str(duration))

    f = open("./others/processed_data/ls_SSIMs_100.pkl","wb")
    pickle.dump(SSIMs,f)
    f.close()


    ### Load times
    print("\nLoad times")
    ce_times = scrap_times_ls()

    f = open("./processed_data/ls_ce_times.pkl","wb")
    pickle.dump(ce_times,f)
    f.close()


if __name__ == '__main__':
    main()