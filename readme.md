# GENAI MBA - USP

Repositório com dados e códigos do TCC do curso "Data Science & Analytics"


### 📁 Estrutura do Repositório

O projeto está organizado da seguinte forma:

- `env.template`: Arquivo que armeza a chave da OpenAI. Para usar, renomear o arquivo para .env.

#### 📂 Notebooks/
Contém os notebooks utilizados nas diferentes etapas do projeto, desde a coleta e preparação dos dados até a análise de resultados.

- `00.EA_data__Sampling.ipynb`: Responsável pela amostragem e pré-processamento inicial dos dados.
- `01.EA_prompts_OpenAi.ipynb`: Explora a criação e execução de prompts com modelos da OpenAI.
- `02.Processing_llm_openai.ipynb`: Processa as respostas geradas pelos LLMs.
- `03.BERT_Sentiment_Analyzer__generated.ipynb`: Implementa e testa a análise de sentimentos utilizando o modelo BERT.
- `04.EA_results.ipynb`: Consolida e visualiza os resultados obtidos ao longo do projeto.

---

#### 📂 src/
Diretório com o código-fonte reutilizável e organizado em módulos.

 
##### 📂 utils/
Contém funções auxiliares utilizadas pelos notebooks e scripts.

- `llm.py`: Implementa funções relacionadas ao uso de modelos de linguagem, como chamadas à API da OpenAI.




## OpenAI

Obtenha a chave de acesso na plataforma da OpenAI [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

Para usar os modelos da OpenAI usando a API é necessário definir a variável de ambiente OPENAI_API_KEY com o valor da chave.


## BERT

### Install dependecies

Execute o comando
```
poetry install
```

### Execução com GPU

Para executar treinamento ou uso dos modelos de BERT com GPU. Siga os seguintes passos

Passo 1: Verifique se você tem CUDA instalado

- Abra o prompt de comando e digite ```nvidia-smi```.
- Se CUDA estiver instalado, você verá informações sobre sua GPU, incluindo a versão do driver e CUDA.
- Se você não vir essas informações, você precisará instalar o CUDA Toolkit no site da NVIDIA. Certifique-se de escolher a versão do CUDA suportada pelo PyTorch que você está usando, ou pela sua placa. 

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 561.03                 Driver Version: 561.03         CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce MX450         WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   54C    P0             N/A / ERR!  |       0MiB /   2048MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

Passo 2: Instale o PyTorch com suporte a CUDA

- Abra o prompt de comando como administrador.
- Execute o seguinte comando, mas antes, certifique-se de selecionar no site do pytorch as opções corretas para seu sistema, especialmente a versão do CUDA. Neste caso 12.6

    ```
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    ```


