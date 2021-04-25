#!/usr/bin/bash

tool="${1:-all}"

mkdir -p output

if [[ "$tool" == "all" || "$tool" == "deepfool" ]]; then
    for ((run = 0 ; run < 5 ; run++)); do
        mkdir -p ./output/Falsifiers/DeepFool/run$run
        for ((property = 1 ; property < 3 ; property++)); do
            mkdir -p ./output/Falsifiers/DeepFool/run$run/property$property
            for ((model_type = 1 ; model_type < 2 ; model_type++)); do
                if [[ $model_type == 0 ]]; then
                    python ./common/dnnf_pipeline.py \
                        ./saved_models/network \
                        ./saved_models/decoder \
                        ./saved_models/onnx/network.onnx \
                        ./properties/final_properties_dnn$property \
                        ./output/Falsifiers/DeepFool/run$run/property$property/resultsDNN \
                        "cleverhans.DeepFool"
                else
                    python ./common/dnnf_pipeline.py \
                        ./saved_models/network \
                        ./saved_models/decoder \
                        ./saved_models/onnx/modifiedDnn.onnx \
                        ./properties/final_properties$property \
                        ./output/Falsifiers/DeepFool/run$run/property$property/results \
                        "cleverhans.DeepFool"
                fi
            done
        done
    done
fi

if [[ "$tool" == "all" || "$tool" == "bim" ]]; then
    for ((run = 0 ; run < 5 ; run++)); do
        mkdir -p ./output/Falsifiers/BIM/run$run
        for ((property = 1 ; property < 3 ; property++)); do
            mkdir -p ./output/Falsifiers/BIM/run$run/property$property
            for ((model_type = 0 ; model_type < 2 ; model_type++)); do
                if [[ $model_type == 0 ]]; then
                    python ./common/dnnf_pipeline.py \
                        ./saved_models/network \
                        ./saved_models/decoder \
                        ./saved_models/onnx/network.onnx \
                        ./properties/final_properties_dnn$property \
                        ./output/Falsifiers/BIM/run$run/property$property/resultsDNN \
                        "cleverhans.BasicIterativeMethod"
                else
                    python ./common/dnnf_pipeline.py \
                        ./saved_models/network \
                        ./saved_models/decoder \
                        ./saved_models/onnx/modifiedDnn.onnx \
                        ./properties/final_properties$property \
                        ./output/Falsifiers/BIM/run$run/property$property/results \
                        "cleverhans.BasicIterativeMethod"
                fi
            done        
        done
    done
fi

if [[ "$tool" == "all" || "$tool" == "fgsm" ]]; then
    for ((run = 0 ; run < 5 ; run++)); do
        mkdir -p ./output/Falsifiers/FGSM/run$run
        for ((property = 1 ; property < 3 ; property++)); do
            mkdir -p ./output/Falsifiers/FGSM/run$run/property$property
            for ((model_type = 1 ; model_type < 2 ; model_type++)); do
                if [[ $model_type == 0 ]]; then
                    python ./common/dnnf_pipeline.py \
                        ./saved_models/network \
                        ./saved_models/decoder \
                        ./saved_models/onnx/network.onnx \
                        ./properties/final_properties_dnn$property \
                        ./output/Falsifiers/FGSM/run$run/property$property/resultsDNN \
                        "cleverhans.FastGradientSignMethod"
                else
                    python ./common/dnnf_pipeline.py \
                        ./saved_models/network \
                        ./saved_models/decoder \
                        ./saved_models/onnx/modifiedDnn.onnx \
                        ./properties/final_properties$property \
                        ./output/Falsifiers/FGSM/run$run/property$property/results \
                        "cleverhans.FastGradientSignMethod"
                fi
            done        
        done
    done
fi

if [[ "$tool" == "all" || "$tool" == "pgd" ]]; then
    for ((run = 0 ; run < 5 ; run++)); do
        mkdir -p ./output/Falsifiers/PGD/run$run
        for ((property = 1 ; property < 3 ; property++)); do
            mkdir -p ./output/Falsifiers/PGD/run$run/property$property
            for ((model_type = 1 ; model_type < 2 ; model_type++)); do
                if [[ $model_type == 0 ]]; then
                    python ./common/dnnf_pipeline.py \
                        ./saved_models/network \
                        ./saved_models/decoder \
                        ./saved_models/onnx/network.onnx \
                        ./properties/final_properties_dnn$property \
                        ./output/Falsifiers/PGD/run$run/property$property/resultsDNN \
                        "cleverhans.ProjectedGradientDescent"
                else
                    python ./common/dnnf_pipeline.py \
                        ./saved_models/network \
                        ./saved_models/decoder \
                        ./saved_models/onnx/modifiedDnn.onnx \
                        ./properties/final_properties$property \
                        ./output/Falsifiers/PGD/run$run/property$property/results \
                        "cleverhans.ProjectedGradientDescent"
                fi
            done            
        done
    done
fi
