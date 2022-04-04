echo 'Create a virtual environment'
virtualenv .env -p python3

echo 'Activate the virtual environment'
source .env/bin/activate
# echo 'Update the virtual environment'
pip install -U pip setuptools wheel

echo 'Installing env Shepherd..'
pip install -r environments/shepherd/requirements.txt

echo 'Installing env PRIMAL..'
pip install -r environments/primal/requirements.txt
cd environments/primal/od_mstar3
python3 setup.py build_ext --inplace
cd ../../..
cd environments/primal/astarlib3
python3 setup.py build_ext --inplace
cd ../../..

echo "Installing other environments' dependencies.."
pip install -r environments/requirements.txt

echo 'Installing XARL..'
pip install -e ./package # cmake is needed
