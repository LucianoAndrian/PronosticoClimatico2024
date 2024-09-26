# Crear un Environment de Conda en Linux

1. Es necesario tener Anaconda o Miniconda ya instalado (Si es el caso, pasar al paso 2).
Para instalar Miniconda por terminal:  
`mkdir -p ~/miniconda3`  
`wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh`  
`bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3`  
`rm -rf ~/miniconda3/miniconda.sh`

Inicia conda para bash y zsh:  
`~/miniconda3/bin/conda init bash`  
`~/miniconda3/bin/conda init zsh`

2. Crear el environment de conda.  
Por terminal, ejecutar:  
`conda env create -f ./EnvSet/PronoClim.yml`

3. Activar el environment.  
Una vez creado, para activar el environment:  
`conda activate PronoClim`
