#!/usr/bin/bash

for latent_space in '1' '2' '4' '8' '16' '32'; do
    mkdir -p ./output/multidim_study/latent_space$latent_space
    for number_layer in '1' '2' '4'; do
        mkdir -p ./output/multidim_study/latent_space$latent_space/number_layer$number_layer
        for number_neuron in '16' '32' '64' '128' '256'; do
            mkdir -p ./output/multidim_study/latent_space$latent_space/number_layer$number_layer/number_neuron$number_neuron
            mkdir -p ./output/multidim_study/latent_space$latent_space/number_layer$number_layer/number_neuron$number_neuron/PGD
            for ((run = 0 ; run < 5 ; run++)); do
                mkdir -p ./output/multidim_study/latent_space$latent_space/number_layer$number_layer/number_neuron$number_neuron/PGD/run$run
                for ((property = 1 ; property < 3 ; property++)); do
                    mkdir -p ./output/multidim_study/latent_space$latent_space/number_layer$number_layer/number_neuron$number_neuron/PGD/run$run/property$property
                    python ./common/dnnf_pipeline_multidim_study.py \
                        ./saved_models/network \
                        ./saved_models/latent_space$latent_space/number_layer$number_layer/vae/vae$number_neuron \
                        $latent_space \
                        $number_layer \
                        $number_neuron \
                        ./saved_models/latent_space$latent_space/number_layer$number_layer/onnx/modifiedDnn$number_neuron.onnx \
                        ./properties/multidim_study/final_properties$property \
                        ./output/multidim_study/latent_space$latent_space/number_layer$number_layer/number_neuron$number_neuron/PGD/run$run/property$property \
                        "cleverhans.ProjectedGradientDescent"
                done
            done
        done
    done
done