git clone https://github.com/dlshriver/DNNV.git
cd DNNV/
git checkout 3f41534b46704746c3c9c02ca701b058822da056
./manage.sh init
. .env.d/openenv.sh
./manage.sh install neurify nnenum verinet
pip install --upgrade pip setuptools "dnnf==0.0.4"
pip install "cleverhans==3.0.1"
pip install tqdm
pip install googledrivedownloader
pip install pandas