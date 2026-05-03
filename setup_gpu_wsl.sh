#!/bin/bash
# Configura el entorno GPU para practica-3-IA-ll en WSL2 (Ubuntu)
# Ejecutar desde WSL2: bash /mnt/c/Users/sluis/Desktop/practica-3-IA-ll/setup_gpu_wsl.sh

set -e

PROJ_DIR="/mnt/c/Users/sluis/Desktop/practica-3-IA-ll"
VENV_DIR="$HOME/.venvs/practica3_ia2_gpu"

echo "=== Verificando NVIDIA en WSL2 ==="
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi no encontrado. Asegurate de tener WSL2 con GPU."
    echo "Verifica que el driver de NVIDIA en Windows sea >= 525.x"
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo ""
echo "=== Actualizando paquetes del sistema ==="
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip python3-venv python3-dev

echo ""
echo "=== Creando entorno virtual en $VENV_DIR ==="
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q

echo ""
echo "=== Instalando TensorFlow con soporte CUDA (puede tardar varios minutos) ==="
pip install "tensorflow[and-cuda]" -q

echo ""
echo "=== Instalando paquetes de NLP y ciencia de datos ==="
pip install numpy pandas matplotlib seaborn scikit-learn -q

echo ""
echo "=== Instalando Jupyter ==="
pip install jupyter ipykernel jupyterlab -q

echo ""
echo "=== Registrando kernel en Jupyter ==="
python -m ipykernel install --user \
    --name practica3_ia2_gpu \
    --display-name "Practica 3 IA-II (GPU - WSL2)"

echo ""
echo "=== Verificando instalacion ==="
python -c "
import tensorflow as tf
import numpy as np
gpus = tf.config.list_physical_devices('GPU')
print(f'TensorFlow: {tf.__version__}')
print(f'NumPy:      {np.__version__}')
print(f'GPUs:       {gpus}')
if gpus:
    print('✓ GPU detectada correctamente!')
else:
    print('✗ GPU no detectada. Comprueba el driver de Windows (>= 525.x).')
"

echo ""
echo "=== Setup completo ==="
echo ""
echo "Para lanzar Jupyter desde WSL2:"
echo "  source ~/.venvs/practica3_ia2_gpu/bin/activate"
echo "  jupyter lab --no-browser --ip=0.0.0.0 --port=8888 --notebook-dir='$PROJ_DIR'"
echo ""
echo "Luego abre en Windows: http://localhost:8888"
echo "(copia el token que aparece en la terminal WSL2)"
