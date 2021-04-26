#!/usr/bin/bash

for std in 0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.25 2.5 2.75 3 3.25 3.5 3.75 4; do
    mkdir -p ./output/ls_study/std$std
    for ((run = 0 ; run < 5 ; run++)); do
        mkdir -p ./output/ls_study/std$std/run$run
        for ((property = 1 ; property < 3 ; property++)); do
            mkdir -p ./output/ls_study/std$std/run$run/property$property
            python ./common/dnnf_pipeline_ls_study.py \
                ./saved_models/network \
                ./saved_models/latent_space8/number_layer2/vae/vae256 \
                ./saved_models/latent_space8/number_layer2/onnx/modifiedDnn256.onnx \
                ./properties/ls_study/final_properties$property \
                ./output/ls_study/std$std/run$run/property$property \
                "cleverhans.ProjectedGradientDescent" \
                $std
        done
    done
done