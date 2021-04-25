#!/usr/bin/bash

tool="${1:-all}"

mkdir -p output

if [[ "$tool" == "all" || "$tool" == "neurify" ]]; then
    for ((run = 0 ; run < 5 ; run++)); do
        mkdir -p ./output/Verifiers/Neurify/run$run
        for ((property = 1 ; property < 3 ; property++)); do
            mkdir -p ./output/Verifiers/Neurify/run$run/property$property
            for ((model_type = 0 ; model_type < 2 ; model_type++)); do
                if [[ $model_type == 0 ]]; then
                    python ./common/dnnv_pipeline.py \
                        ./saved_models/network \
                        ./saved_models/vae \
                        ./saved_models/onnx/network.onnx \
                        ./properties/final_properties_dnn$property \
                        ./output/Verifiers/Neurify/run$run/property$property/resultsDNN \
                        "neurify"
                else
                    python ./common/dnnv_pipeline.py \
                        ./saved_models/network \
                        ./saved_models/vae \
                        ./saved_models/onnx/modifiedDnn.onnx \
                        ./properties/final_properties$property \
                        ./output/Verifiers/Neurify/run$run/property$property/results \
                        "neurify"
                fi
            done        
        done
    done
fi

if [[ "$tool" == "all" || "$tool" == "nnenum" ]]; then
    for ((run = 0 ; run < 5 ; run++)); do
        mkdir -p ./output/Verifiers/run$run
        for ((property = 1 ; property < 3 ; property++)); do
            mkdir -p ./output/Verifiers/Nnenum/run$run/property$property
            for ((model_type = 0 ; model_type < 2 ; model_type++)); do
                if [[ $model_type == 0 ]]; then
                    python ./common/dnnv_pipeline.py \
                        ./saved_models/network \
                        ./saved_models/vae \
                        ./saved_models/onnx/network.onnx \
                        ./properties/final_properties_dnn$property \
                        ./output/Verifiers/Nnenum/run$run/property$property/resultsDNN \
                        "nnenum"
                else
                    python ./common/dnnv_pipeline.py \
                        ./saved_models/network \
                        ./saved_models/vae \
                        ./saved_models/onnx/modifiedDnn.onnx \
                        ./properties/final_properties$property \
                        ./output/Verifiers/Nnenum/run$run/property$property/results \
                        "nnenum"
                fi
            done
        done
    done
fi

if [[ "$tool" == "all" || "$tool" == "verinet" ]]; then
    for ((run = 0 ; run < 5 ; run++)); do
        mkdir -p ./output/Verifiers/Verinet/run$run
        for ((property = 1 ; property < 3 ; property++)); do
            mkdir -p ./output/Verifiers/Verinet/run$run/property$property
            for ((model_type = 0 ; model_type < 2 ; model_type++)); do
                if [[ $model_type == 0 ]]; then
                    python ./common/dnnv_pipeline.py \
                        ./saved_models/network \
                        ./saved_models/vae \
                        ./saved_models/onnx/network.onnx \
                        ./properties/final_properties_dnn$property \
                        ./output/Verifiers/Verinet/run$run/property$property/resultsDNN \
                        "verinet"
                else
                    python ./common/dnnv_pipeline.py \
                        ./saved_models/network \
                        ./saved_models/vae \
                        ./saved_models/onnx/modifiedDnn.onnx \
                        ./properties/final_properties$property \
                        ./output/Verifiers/Verinet/run$run/property$property/results \
                        "verinet"
                fi
            done            
        done
    done
fi