import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import warnings; warnings.simplefilter('ignore')
import sys
sys.path.append('./saved_models')
from train_vae import Vae, Decoder, Encoder, Network
from train_vae_mrs import VaeMrs, EncoderMrs, DecoderMrs
sys.path.append('./common')
import ssim


#### Common functions

def mse_1000(vae, images):
    all_mse_list = list()
    for i in range(1000):
        decoded_images,_,_ = vae(images.view(images.shape[0],784))
        mse_list = ((images.view(images.shape[0],784) - decoded_images)**2).detach().numpy().mean(axis=1)
        all_mse_list.append(mse_list)
    all_mse_list = np.array(all_mse_list)
    return all_mse_list.mean(axis=0), all_mse_list.mean()


def decode_ce(ce, variational_decoder):
    decoded_img = variational_decoder(torch.tensor(ce))
    decoded_img = decoded_img.detach().numpy()
    return decoded_img


def scrap_counter_examples(output_file, decoder):

    counter_examples_with_decoder = []
    counter_examples_without_decoder = []

    if os.path.exists(output_file):
        for i, run in enumerate(sorted(os.listdir(output_file))):
            f_path = output_file+'/'+run
            if os.path.exists(f_path):
                for j, property_file in enumerate(sorted(os.listdir(f_path))):
                    f_path = output_file + '/'+property_file
                    if os.path.exists(f_path):
                        for l, r_type in enumerate(sorted(os.listdir(f_path))):
                            f_path = output_file +'/'+r_type+'/counter_examples/'
                            if os.path.exists(f_path):
                                for k, ce_name in enumerate(sorted(os.listdir(f_path))):
                                    ce = np.load(f_path + ce_name)

                                    if r_type == 'results':
                                        counter_examples_with_decoder.append(decode_ce(ce, decoder))
                                    if r_type == 'resultsDNN':
                                        counter_examples_without_decoder.append(ce)
    
    return counter_examples_with_decoder, counter_examples_without_decoder


def scrap_counter_examples_times():

    counter_examples_times = dict()
    types = ['with_decoder', 'without_decoder']

    for tool in ['Falsifiers', 'Verifiers']:
        counter_examples_times[tool] = dict()

        f_path = './output/'+tool
        if os.path.exists(f_path):
            for tool_name in os.listdir(f_path):
                counter_examples_times[tool][tool_name] = dict()
                counter_examples_times[tool][tool_name][types[0]] = list()
                counter_examples_times[tool][tool_name][types[1]] = list()

                f_path = f_path + '/' + tool_name
                if os.path.exists(f_path):
                    for r, run in enumerate(sorted(os.listdir(f_path))):
                        f_path = f_path + '/' + run
                        if os.path.exists(f_path):
                            for pf, property_file in enumerate(sorted(os.listdir(f_path))):
                                f_path = f_path + '/' + property_file
                                if os.path.exists(f_path):
                                    for rt, r_type in enumerate(sorted(os.listdir(f_path))):
                                        f_path = f_path + '/'+r_type+'/result_summary.md'
                                        if os.path.exists(f_path):

                                            summary = open(f_path, 'r')
                                            aux_status = ''

                                            for l in summary:
                                                if '##Status:' in l:
                                                    aux_status = l.split(' ')[-1].split('\n')[0]
                                                if '###Time:' in l:
                                                    if aux_status == 'sat':
                                                        counter_examples_times[tool][tool_name][types[rt]].append(float(l.split(' ')[-1]))
                                                    aux_status = ''
                                            summary.close()
    
    return counter_examples_times


def scrap_counter_examples_status(output_file, prop=2):
    ### runs / models / properties
    property_status = np.array([[["timeout"]*5] * 2] * 10, dtype = 'object')

    if os.path.exists(output_file):
        for i, run in enumerate(sorted(os.listdir(output_file))):
            f_path = output_file+'/'+run+'/property'+str(prop)
            if os.path.exists(f_path):
                for j, r_type in enumerate(sorted(os.listdir(f_path))):
                    f_path = f_path + '/'+r_type+'/result_summary.md'
                    if os.path.exists(f_path):
                        summary = open(f_path, 'r')
                        for l in summary:
                                if "#Property" in l:
                                    aux = int(l.split(' ')[-1])
                                if '##Status:' in l:
                                    if 'NeurifyError' in l:
                                        property_status[aux][j][i] = 'error'
                                    elif 'NnenumTranslatorError' in l:
                                        property_status[aux][j][i] = 'error'
                                    elif 'NnenumError' in l:
                                        property_status[aux][j][i] = 'error'
                                    else:
                                        property_status[aux][j][i] = l.split(' ')[-1].split('\n')[0]
    return property_status


def scrap_specific_counter_examples(output_file, property_type, ce_number, decoder):

    counter_examples_with_decoder = []
    counter_examples_without_decoder = []

    if os.path.exists(output_file):
        for i, run in enumerate(sorted(os.listdir(output_file))):
            f_path = output_file+'/'+run+'/property'+str(property_type)
            if os.path.exists(f_path):
                for l, r_type in enumerate(sorted(os.listdir(f_path))):
                    f_path = f_path + '/'+r_type+'/counter_examples/ce_property' + str(ce_number) + '.npy'
                    if os.path.exists(f_path):
                        ce = np.load(f_path)

                        if r_type == 'results':
                            counter_examples_with_decoder.append(decode_ce(ce, decoder))
                        if r_type == 'resultsDNN':
                            counter_examples_without_decoder.append(ce)
    
    return counter_examples_with_decoder, counter_examples_without_decoder


def main():
    ### Load test data

    # Test data
    test_data = datasets.FashionMNIST(
        root='.data/FashionMNIST',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    # Test data loader
    test_loader = DataLoader(
        test_data,
        batch_size=10000,
        shuffle=False
    )

    dataiter = iter(test_loader)
    test, test_labels = dataiter.next()


    ### Load models
    print("Load models")
    vae = Vae(2)
    vae.load_state_dict(torch.load('./saved_models/vae', map_location=torch.device('cpu')))
    decoder = vae.decoder

    ### Load vae_mrs
    print("Load vae_mrs")
    vae_mrs = VaeMrs(100)
    vae_mrs.load_state_dict(torch.load('./saved_models/vae_mrs', map_location=torch.device('cpu')))


    ### Load all counter-examples
    print("\nLoad all counter-examples")

    counter_examples = dict()

    for t, tool in enumerate(['BIM','DeepFool','FGSM','PGD','Neurify','Nnenum','Verinet']):
        counter_examples[tool] = dict()
        if t < 4:
            tool_type = "Falsifiers"
        else:
            tool_type = "Verifiers"

        f_path = "./output/"+tool_type+"/"+tool## INSIDE scrap_counter_examples
        if os.path.exists(f_path):
            counter_examples[tool]['with_decoder'], counter_examples[tool]['without_decoder'] = scrap_counter_examples(output_file=f_path, decoder=decoder)
        else:
            counter_examples[tool]['with_decoder'], counter_examples[tool]['without_decoder'] = [], []

    if not os.path.exists('./processed_data'):
        os.makedirs('./processed_data')
    f = open("./processed_data/counter_examples.pkl","wb")
    pickle.dump(counter_examples,f)
    f.close()


    ### Calculate MSE of counter-examples
    print("\nCalculate MSE of counter-examples")

    MSEs = dict()
    for t, tool in enumerate(['BIM','DeepFool','FGSM','PGD','Neurify','Nnenum','Verinet']):
        
        print(tool)

        MSEs[tool] = dict()
        MSEs[tool]['with_decoder'], _ = mse_1000(vae_mrs, torch.tensor(counter_examples[tool]['with_decoder']))
        MSEs[tool]['without_decoder'], _ = mse_1000(vae_mrs, torch.tensor(counter_examples[tool]['without_decoder']))

    if not os.path.exists('./processed_data'):
        os.makedirs('./processed_data')
    f = open("./processed_data/MSEs.pkl","wb")
    pickle.dump(MSEs,f)
    f.close()


    ### Calculate VAE MSE on test set
    print("\nCalculate VAE MSE on test set")

    vae_mse_test_data, _ = mse_1000(vae, test.reshape(10000,784))

    if not os.path.exists('./processed_data'):
        os.makedirs('./processed_data')
    f = open("./processed_data/vae_mse_test_data.pkl","wb")
    pickle.dump(vae_mse_test_data,f)
    f.close()


    ### Calculate SSIM of counter-examples
    print("\nCalculate SSIM of counter-examples")

    SSIMs = dict()
    for t, tool in enumerate(['BIM','DeepFool','FGSM','PGD','Neurify','Nnenum','Verinet']):
        
        print(tool)

        SSIMs[tool] = dict()
        SSIMs[tool]['with_decoder'] = ssim.ssim_100(vae_mrs, torch.tensor(counter_examples[tool]['with_decoder']))[0].mean(axis=0)
        SSIMs[tool]['without_decoder'] = ssim.ssim_100(vae_mrs, torch.tensor(counter_examples[tool]['without_decoder']))[0].mean(axis=0)

    if not os.path.exists('./processed_data'):
        os.makedirs('./processed_data')
    f = open("./processed_data/SSIMs.pkl","wb")
    pickle.dump(SSIMs,f)
    f.close()


    ### Calculate VAE SSIM on test set
    print("\nCalculate VAE SSIM on test set")

    vae_ssim_test_data, _ = ssim.ssim_100(vae, test.reshape(10000,784))

    if not os.path.exists('./processed_data'):
        os.makedirs('./processed_data')
    f = open("./processed_data/vae_ssim_test_data.pkl","wb")
    pickle.dump(vae_ssim_test_data,f)
    f.close()


    ### Load counter-examples times
    print("\nLoad counter-examples times")

    counter_examples_times = scrap_counter_examples_times()

    if not os.path.exists('./processed_data'):
        os.makedirs('./processed_data')
    f = open("./processed_data/counter_examples_times.pkl","wb")
    pickle.dump(counter_examples_times,f)
    f.close()


    # Load properties status
    print("\nLoad properties status")

    tools = {
        'Falsifiers': ['PGD', 'BIM', 'FGSM', 'DeepFool'],
        'Verifiers': ['Neurify', 'Nnenum', 'Verinet']
    }
    properties_status = {
        1: dict(),
        2: dict()
    }
    for prop_type in range(1,3): 
        for tool_type in ['Falsifiers', 'Verifiers']:
            for tool in tools[tool_type]:
                properties_status[prop_type][tool] = scrap_counter_examples_status('./output/'+tool_type+'/'+tool, prop=prop_type)

    if not os.path.exists('./processed_data'):
        os.makedirs('./processed_data')
    f = open("./processed_data/properties_status.pkl","wb")
    pickle.dump(properties_status,f)
    f.close()


    # Get the counter-examaples with lowest MSE for each property
    print("\nGet the counter-examaples with lowest MSE for each property")

    ces_with_lowest_mse = {
        'with_decoder': [],
        'without_decoder': [],
    }

    for prop_type in range(1,3):
        for prop in range(10):
            row_with_decoder = list()
            row_without_decoder = list()
            for tool in ['Falsifiers/DeepFool','Falsifiers/BIM','Falsifiers/FGSM','Falsifiers/PGD','Verifiers/Neurify','Verifiers/Nnenum','Verifiers/Verinet']:
                aux_with_decoder, aux_without_decoder = scrap_specific_counter_examples(output_file="./output/"+tool, property_type=prop_type, ce_number=prop, decoder=decoder)
                
                # With decoder
                if len(aux_with_decoder) != 0:
                    aux_index,_ = mse_1000(vae_mrs, torch.tensor(aux_with_decoder))
                    row_with_decoder.append(aux_with_decoder[aux_index.argmin()])
                else:
                    row_with_decoder.append(np.zeros([28,28]))

                # Without decoder
                if len(aux_without_decoder) != 0:
                    aux_index,_ = mse_1000(vae_mrs, torch.tensor(aux_without_decoder))
                    row_without_decoder.append(aux_without_decoder[aux_index.argmin()])
                else:
                    row_without_decoder.append(np.zeros([28,28]))

            ces_with_lowest_mse['with_decoder'].append(row_with_decoder)
            ces_with_lowest_mse['without_decoder'].append(row_without_decoder)

    if not os.path.exists('./processed_data'):
        os.makedirs('./processed_data')
    f = open("./processed_data/ces_with_lowest_mse.pkl","wb")
    pickle.dump(ces_with_lowest_mse,f)
    f.close()


    # Get the counter-examaples with highest SSIM for each property
    print("\nGet the counter-examaples with highest SSIM for each property")

    ces_with_highest_ssim = {
        'with_decoder': [],
        'without_decoder': [],
    }

    for prop_type in range(1,3):
        for prop in range(10):
            row_with_decoder = list()
            row_without_decoder = list()
            for tool in ['Falsifiers/DeepFool','Falsifiers/BIM','Falsifiers/FGSM','Falsifiers/PGD','Verifiers/Neurify','Verifiers/Nnenum','Verifiers/Verinet']:
                aux_with_decoder, aux_without_decoder = scrap_specific_counter_examples(output_file="./output/"+tool, property_type=prop_type, ce_number=prop, decoder=decoder)
                
                # With decoder
                if len(aux_with_decoder) != 0:
                    aux_index = ssim.ssim_100(vae_mrs, torch.tensor(aux_with_decoder))[0].mean(axis=0)
                    row_with_decoder.append(aux_with_decoder[aux_index.argmax()])
                else:
                    row_with_decoder.append(np.zeros([28,28]))

                # Without decoder
                if len(aux_without_decoder) != 0:
                    aux_index = ssim.ssim_100(vae_mrs, torch.tensor(aux_without_decoder))[0].mean(axis=0)
                    row_without_decoder.append(aux_without_decoder[aux_index.argmax()])
                else:
                    row_without_decoder.append(np.zeros([28,28]))

            ces_with_highest_ssim['with_decoder'].append(row_with_decoder)
            ces_with_highest_ssim['without_decoder'].append(row_without_decoder)

    if not os.path.exists('./processed_data'):
        os.makedirs('./processed_data')
    f = open("./processed_data/ces_with_highest_ssim.pkl","wb")
    pickle.dump(ces_with_highest_ssim,f)
    f.close()


if __name__ == "__main__":
    main()