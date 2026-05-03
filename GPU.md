  El entorno está listo. Aquí el resumen completo:

  Lo que está hecho                                                                                                     
  El venv .venv ya está creado en la carpeta del proyecto con todos los paquetes:                                         - TensorFlow 2.21.0, NumPy 2.4.4, scikit-learn 1.8.0, pandas, matplotlib, seaborn
  - Jupyter + JupyterLab + kernel registrado como "Practica 3 IA-II (TF GPU)"

  Para usar el notebook ahora mismo (CPU):
  - En VS Code: abre PW3_E1_Group_2.ipynb y selecciona el kernel Practica 3 IA-II (TF GPU)
  - O desde terminal: .venv\Scripts\jupyter lab

  ---
  Para activar la GPU (RTX 2060)

  TensorFlow ≥ 2.11 no soporta GPU en Windows nativo — necesitas ejecutar en WSL2. Tienes que hacer esto una sola vez:

  Paso 1 — Instalar Ubuntu en WSL2

  Abre PowerShell como Administrador y ejecuta:
  wsl --install -d Ubuntu-22.04
  Te pedirá un nombre de usuario y contraseña de Linux (puedes poner lo que quieras). No requiere reinicio ya que WSL2
  ya está habilitado.

  Paso 2 — Ejecutar el script de configuración

  Una vez que Ubuntu esté abierta, ejecuta:
  bash /mnt/c/Users/sluis/Desktop/practica-3-IA-ll/setup_gpu_wsl.sh
  Esto instala automáticamente TensorFlow con CUDA, todos los paquetes y registra el kernel de Jupyter.

  Paso 3 — Lanzar Jupyter desde WSL2

  source ~/.venvs/practica3_ia2_gpu/bin/activate
  jupyter lab --no-browser --ip=0.0.0.0 --port=8888 --notebook-dir='/mnt/c/Users/sluis/Desktop/practica-3-IA-ll'
  Abre http://localhost:8888 en tu navegador Windows y copia el token que aparece en la terminal.