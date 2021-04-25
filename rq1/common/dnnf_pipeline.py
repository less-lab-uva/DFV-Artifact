# DNNF pipeline for torch models 

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append('../')
from train_vae import Decoder, Network
import subprocess
import os
from pathlib import Path

class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Execute DNNV using the command line
def execute_dnnv_properties(dnn_onnx_path, backend_name, properties_path, results_path):
    if not os.path.exists(results_path + '/counter_examples'):
        os.makedirs(results_path + '/counter_examples')
    if not os.path.exists(results_path + '/dnnf_output'):
        os.makedirs(results_path + '/dnnf_output')
    for i, filename in enumerate(sorted(os.listdir(properties_path))):
        property_name = filename.split('.')[0]
        print("VERIFYING PROPERTY: " + str(filename))
        with open(results_path + '/dnnf_output/results_' + property_name + '.txt', 'w') as f:
            if backend_name == "cleverhans.ProjectedGradientDescent":
                subprocess.run(
                    ['timeout', "--signal=SIGINT", '3600',
                    'python', '-m', 'dnnf',
                    properties_path + '/' + filename,
                    '--network', 'N', dnn_onnx_path, 
                    '--backend' + backend_name,
                    '--save-violation', results_path + '/counter_examples/ce_' + property_name],
                stdout=f, stderr=f, text=True)
            elif  backend_name == "cleverhans.FastGradientSignMethod":
                subprocess.run(
                    ['timeout', "--signal=SIGINT", '3600',
                    'python', '-m', 'dnnf',
                    properties_path + '/' + filename,
                    '--network', 'N', dnn_onnx_path, 
                    '--backend' + backend_name, '--n_start', '1',
                    '--save-violation', results_path + '/counter_examples/ce_' + property_name],
                stdout=f, stderr=f, text=True)
            elif  backend_name == "cleverhans.DeepFool":
                subprocess.run(
                    ['timeout', "--signal=SIGINT", '3600',
                    'python', '-m', 'dnnf',
                    properties_path + '/' + filename,
                    '--network', 'N', dnn_onnx_path, 
                    '--backend' + backend_name,'--set', 'cleverhans.DeepFool', 'nb_candidate', '2' , '--n_start', '1',
                    '--save-violation', results_path + '/counter_examples/ce_' + property_name],
                stdout=f, stderr=f, text=True)
            elif  backend_name == "cleverhans.BasicIterativeMethod":
                subprocess.run(
                    ['timeout', "--signal=SIGINT", '3600',
                    'python', '-m', 'dnnf',
                    properties_path + '/' + filename,
                    '--network', 'N', dnn_onnx_path, 
                    '--backend' + backend_name, '--n_start', '1',
                    '--save-violation', results_path + '/counter_examples/ce_' + property_name],
                stdout=f, stderr=f, text=True)


# Generate JSON with important data from Result
def get_data_from_result_file(file_path):
    result_file = open(file_path, "r")

    result = {}
    result['status'] = None
    result['time'] = None

    for l in result_file:
        if("result: " in l):
            result['status'] = l.split('result: ')[1].split('\n')[0]
        
        if("time: " in l):
            result['time'] = l.split("time: ")[1].split('\n')[0]
            
    result_file.close()
    return result


def create_digit_image(imgNumber, results_path, decoder):
    if os.path.exists(results_path + '/counter_examples/ce_property' + str(imgNumber) + '.npy'):
        ce = np.load(results_path + '/counter_examples/ce_property' + str(imgNumber) + '.npy')
        is_encoded = not (len(ce[0]) == 784)
        if is_encoded:
            image = decoder(torch.tensor(ce))
            image = image.detach().numpy()
        else:
            image = ce[0]
        plt.imshow(image.reshape(28,28))
        plt.savefig(results_path + "/images/image"+str(imgNumber)+".png")
        return image
    else:
        return None


def generate_results(dnn, decoder, results_path):
    if not os.path.exists(results_path + '/images'):
        os.makedirs(results_path + '/images')

    result_summary = open(results_path + "/result_summary.md", "w")
    for i, filename in enumerate(sorted(os.listdir(results_path + '/dnnf_output'))):

        result_summary.write("#Property " + str(i) + '\n')
        result_data = get_data_from_result_file(str(results_path+"/dnnf_output/"+filename))

        if(result_data['status'] != None):
            result_summary.write("##Status: " + result_data['status'] + '\n')
            result_summary.write("###Time: "+result_data['time']+'\n')

        decoded_digit = create_digit_image(i, results_path, decoder)

        if type(decoded_digit) == np.ndarray:
            result_summary.write("![](./images/image"+str(i)+".png)"+'\n')
            
            result_summary.write("###DNN Prediction: "+'\n')
            dnn_prediction = dnn(torch.tensor(decoded_digit.reshape(1,784))).detach().numpy()
            selected_class = np.argmax(dnn_prediction)
            for j, pred in enumerate(dnn_prediction[0]):
                if j == selected_class:
                    result_summary.write("<mark>" + str(class_names[j]) + " :" + str(pred)+ "</mark>" +'\n')
                else:
                    result_summary.write(str(class_names[j]) + " :" + str(pred)+'\n')

    result_summary.close()


def main():
    if len(sys.argv) < 7:
        print("Need: Dnn path, Decoder path, model.onnx path, properties path, results path and verifier name")
        return

    dnn_path = sys.argv[1]
    decoder_path = sys.argv[2]
    dnn_onnx_path = sys.argv[3]
    properties_path = sys.argv[4]
    results_path = sys.argv[5]
    backend_name = sys.argv[6]

    dnn = torch.load(dnn_path, map_location=torch.device('cpu'))
    decoder = torch.load(decoder_path, map_location=torch.device('cpu'))
    
    execute_dnnv_properties(dnn_onnx_path, backend_name, properties_path, results_path)
    generate_results(dnn, decoder, results_path)


if __name__ == '__main__':
    main()