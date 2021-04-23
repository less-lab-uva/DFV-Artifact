#!/usr/bin/bash

export TF_CPP_MIN_LOG_LEVEL=3

model="${1:-all}"

cexdir="${2:-cex}"
timeout=${3:-3600}

mkdir -p $cexdir/npy
mkdir -p $cexdir/png

echo "Falsifying model $model with a timeout of $timeout seconds."
echo "Counter examples will be saved to $cexdir."

for ((j = 0; j < 5; j++)); do
    for ((i = 0; i < 10; i++)); do
        if [[ "$model" == "all" || "$model" == "dnn" ]]; then
            echo "DNN PGD $i seed=$j"
            python resmonitor.py -T $timeout dnnf benchmark/properties/dronet_property_$i.py --network N benchmark/onnx/dronet.onnx --backend cleverhans.ProjectedGradientDescent --save-violation $cexdir/npy/dnn_pgd_${i}_${j}.npy -p 2 --seed=$j
        fi

        if [[ "$model" == "all" || "$model" == "vae" ]]; then
            echo "VAE+DNN PGD $i seed=$j"
            python resmonitor.py -T $timeout dnnf benchmark/vae_properties/dronet_property_$i.py --network VAE dronet_fc_vae_decoder.onnx --network DNN benchmark/onnx/dronet.onnx --backend cleverhans.ProjectedGradientDescent --save-violation $cexdir/npy/vae+dnn_pgd_${i}_${j}.npy -p 2 --seed=$j
        fi

        if [[ "$model" == "all" || "$model" == "gan" ]]; then
            echo "GAN+DNN PGD $i seed=$j"
            python resmonitor.py -T $timeout dnnf benchmark/vae_properties/dronet_property_$i.py --network VAE dronet_dcgan_generator.onnx --network DNN benchmark/onnx/dronet.onnx --backend cleverhans.ProjectedGradientDescent --save-violation $cexdir/npy/gan+dnn_pgd_${i}_${j}.npy -p 2 --seed=$j
        fi
    done
done
