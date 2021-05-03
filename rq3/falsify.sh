#!/usr/bin/bash

export TF_CPP_MIN_LOG_LEVEL=3

model="${1:-all}"

cexdir="${2:-cex}"
timeout=${3:-3600}

logdir="${4:-logs/falsification_logs}"

mkdir -p $cexdir/npy
mkdir -p $cexdir/png
mkdir -p $logdir

echo "Falsifying model $model with a timeout of $timeout seconds."
echo "Counter examples will be saved to $cexdir."

if [[ "$model" == "all" || "$model" == "dnn" ]]; then
    rm $logdir/log_falsify_dnn.out 
    rm $logdir/log_falsify_dnn.err
fi
if [[ "$model" == "all" || "$model" == "vae" ]]; then
    rm $logdir/log_falsify_vae.out 
    rm $logdir/log_falsify_vae.err
fi
if [[ "$model" == "all" || "$model" == "gan" ]]; then
    rm $logdir/log_falsify_gan.out 
    rm $logdir/log_falsify_gan.err
fi

for ((j = 0; j < 5; j++)); do
    for ((i = 0; i < 10; i++)); do
        if [[ "$model" == "all" || "$model" == "dnn" ]]; then
            echo "DNN PGD $i seed=$j" >>$logdir/log_falsify_dnn.out 2>>$logdir/log_falsify_dnn.err
            python resmonitor.py -T $timeout dnnf benchmark/properties/dronet_property_$i.py --network N benchmark/onnx/dronet.onnx --backend cleverhans.ProjectedGradientDescent --save-violation $cexdir/npy/dnn_pgd_${i}_${j}.npy -p 2 --seed=$j >>$logdir/log_falsify_dnn.out 2>>$logdir/log_falsify_dnn.err
        fi

        if [[ "$model" == "all" || "$model" == "vae" ]]; then
            echo "VAE+DNN PGD $i seed=$j" >>$logdir/log_falsify_vae.out 2>>$logdir/log_falsify_vae.err
            python resmonitor.py -T $timeout dnnf benchmark/vae_properties/dronet_property_$i.py --network VAE models/dronet_fc_vae_decoder.onnx --network DNN benchmark/onnx/dronet.onnx --backend cleverhans.ProjectedGradientDescent --save-violation $cexdir/npy/vae+dnn_pgd_${i}_${j}.npy -p 2 --seed=$j >>$logdir/log_falsify_vae.out 2>>$logdir/log_falsify_vae.err
        fi

        if [[ "$model" == "all" || "$model" == "gan" ]]; then
            echo "GAN+DNN PGD $i seed=$j" >>$logdir/log_falsify_gan.out 2>>$logdir/log_falsify_gan.err
            python resmonitor.py -T $timeout dnnf benchmark/vae_properties/dronet_property_$i.py --network VAE models/dronet_dcgan_generator.onnx --network DNN benchmark/onnx/dronet.onnx --backend cleverhans.ProjectedGradientDescent --save-violation $cexdir/npy/gan+dnn_pgd_${i}_${j}.npy -p 2 --seed=$j >>$logdir/log_falsify_gan.out 2>>$logdir/log_falsify_gan.err
        fi
    done
done
