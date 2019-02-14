module load Anaconda3/5.0.1
module unload gcc
module load cuda/9.0

echo "Setting up conda virtual env..."
conda create -y -n QOC_qiskit
source activate QOC_qiskit
python -V
conda install -y tensorflow
yes | pip install qutip
yes | pip install ipython
yes | pip install jupyter
yes | pip install matplotlib
yes | pip install pyscf

echo "Installing Qiskit packages..."
yes | pip install qiskit-ibmq-provider
cd /qiskit-terra/
yes | pip install -r requirements.txt
yes | pip install -r requirements-dev.txt
yes | pip install -e .
cd ../qiskit-aqua/
yes | pip install -r requirements.txt
yes | pip install -r requirements-dev.txt
yes | pip install -e .
cd ../qiskit-chemistry/
yes |pip install -r requirements.txt
yes |pip install -r requirements-dev.txt
yes |pip install -e .

echo "Installing quantum optimal control..."
cd ../PY3_quantum-optimal-control/
pip install --user -e .
echo "QOC_Qiskit installation completed."
cd ..

