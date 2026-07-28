<h1 align="center"> Agentic-R: Learning to Retrieve for Agentic Search</a></h1>

<div align="center">
<a href="https://arxiv.org/pdf/2601.11888" target="_blank"><img src=https://img.shields.io/badge/Paper-arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://huggingface.co/papers/2601.11888" target="_blank"><img src=https://img.shields.io/badge/Paper-Hugging%20Face-yellow?logo=huggingface></a>
<a href="https://huggingface.co/collections/liuwenhan/agentic-r" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Models-27b3b4.svg></a>
<a href="https://modelscope.cn/collections/lwhlwh/Agentic-R" target="_blank"><img src=https://custom-icon-badges.demolab.com/badge/ModelScope%20Models-624aff?style=flat&logo=modelscope&logoColor=white></a>
<a href="https://opensource.org/licenses/MIT"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green.svg"></a>
<a href="https://www.python.org/downloads/release/python-3100/"><img alt="Static Badge" src="https://img.shields.io/badge/Python-3.10+-blue.svg"></a>
</div>
<h5 align="center"> Si te gusta nuestro proyecto, por favor danos una estrella ⭐ en GitHub.</h5>

## 📣 Últimas Noticias
- **[6 Abr, 2026]**: 🔔 ¡Nuestro artículo ha sido aceptado en **ACL 2026 (Findings)**!
- **[26 Ene, 2026]**: 🚀 Hemos lanzado nuestro **[🤗search agent](https://huggingface.co/liuwenhan/triviaqa_hotpotqa_train-search-r1-ppo-qwen2.5-7b-em-iter1)** entrenado y nuestro **[🤗wikipedia corpus](https://huggingface.co/datasets/liuwenhan/retrieval_corpus)**.
- **[15 Ene, 2026]**: 🚀 Hemos lanzado nuestro código completo y nuestro modelo recuperador **[🤗Agentic-R_e5](https://huggingface.co/liuwenhan/Agentic-R_e5)**.

## 1. Introducción a Agentic-R

### 💡 1.1 Descripción General

**Agentic-R** es un recuperador denso diseñado específicamente para la búsqueda agentica. Para entrenarlo, primero diseñamos un enfoque novedoso para medir la utilidad de los pasajes en la búsqueda agentica y luego proponemos un enfoque de optimización iterativa Agente-Recuperador.

<p align="center">
<img width="80%" alt="image" src="https://8421bcd.oss-cn-beijing.aliyuncs.com/img/image-20260113201106723.png" />
</p>

### 📊 1.2 Rendimiento General

<p align="center">
<img width="80%" alt="image" src="https://8421bcd.oss-cn-beijing.aliyuncs.com/img/image-20260113201234638.png" />
</p>

## ⚡ 2. Inicio Rápido para probar Agentic-R

### **📘** 2.1 Entorno y Preparación

##### Entorno

En este paso, describiremos los paquetes necesarios para realizar inferencias con Agentic-R. Recomendamos encarecidamente el uso de un entorno de conda separado.

```bash
# ---------------------------------- crear entorno ----------------------------------
conda create -n agentic-r python=3.10 -y
source ~/.bashrc
conda activate agentic-r
# ---------------------------------- instalar paquetes ----------------------------------
cd FlashRAG
pip install -e .
pip install vllm==0.10.1
pip install sentence-transformers
pip install pyserini
pip install GPUtil
pip install nvitop
pip install termcolor
pip install numpy==1.26
pip install deepspeed==0.18.0
pip install qwen_omni_utils
pip install modelscope
pip install faiss_gpu==1.7.3
pip install transformers==4.57.1
```

##### Preparación

**a.** Después de instalar los paquetes necesarios, recuerda **actualizar** ``WORKSPACE_DIR`` y ``PROJECT_DIR`` (ambos deben ser rutas absolutas) en ``config.py``. Estos dos parámetros se utilizarán tanto en nuestros códigos de inferencia como en los de entrenamiento. Aquí tienes una estructura de directorios recomendada:

```bash
{WORKSPACE_DIR}
├── trained_models
│   ├── Agentic-R_e5
│   └── triviaqa_hotpotqa_train-search-r1-ppo-qwen2.5-7b-em-iter1
│
├── data
│   └── FlashRAG_Dataset
│       ├── nq
│       ├── hotpotqa
│       ├── retrieval_corpus
│       └── ...
│
└── {PROJECT_DIR}  (es decir, Agentic-R)
    ├── FlashRAG
    ├── Search-R1
    ├── tevatron
    └── config.py
```

**b**. Descarga los conjuntos de datos para pruebas (como nq, hotpotqa, ...) desde [FlashRAG_Dataset](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/main) y colócalos en el directorio `{WORKSPACE_DIR}/data/FlashRAG_Dataset/`. Descarga nuestro [search agent](https://huggingface.co/liuwenhan/triviaqa_hotpotqa_train-search-r1-ppo-qwen2.5-7b-em-iter1) entrenado y colócalo en el directorio `{WORKSPACE_DIR}/trained_models/`. Descarga el [retrieval corpus](https://huggingface.co/datasets/liuwenhan/retrieval_corpus) y colócalo en el directorio `{WORKSPACE_DIR}/data/FlashRAG_Dataset/`.

**c**. Descarga [Agentic-R](https://huggingface.co/liuwenhan/Agentic-R_e5) y colócalo en el directorio `{WORKSPACE_DIR}/trained_models/` y construye el índice de wikipedia basándote en el siguiente código:

```shell
conda activate agentic-r
model_name=Agentic-R_e5
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m flashrag.retriever.index_builder \
    --retrieval_method ${model_name} \
    --model_path {WORKSPACE_DIR}/trained_models/${model_name} \
    --corpus_path {WORKSPACE_DIR}/data/FlashRAG_Dataset/retrieval_corpus/wiki18_100w.jsonl \
    --save_dir {WORKSPACE_DIR}/data/FlashRAG_Dataset/retrieval_corpus/ \
    --use_fp16 \
    --max_length 256 \
    --batch_size 128 \
    --faiss_type Flat \
    --sentence_transformer \
    --instruction "passage: "
```

#### 3.1.2 Probar Agentic-R basándose en nuestro Agente entrenado

```shell
conda activate agentic-r
cd FlashRAG/examples/methods
bash run_exp.sh
```

*Nota: Para nuestro Agentic-R, el parámetro `agentic_retriever_input` se establece como True, lo que utiliza 'Question [SEP] query' para la recuperación.*

## 🔥 3. ¿Cómo entrenar Agentic-R?

En nuestro trabajo, diseñamos un marco de optimización iterativa Agente-Recuperador que optimiza iterativamente el agente de búsqueda y nuestro Agentic-R. A continuación, utilizaremos la ***primera iteración*** como ejemplo para presentar los códigos de entrenamiento de nuestro agente de búsqueda y Agentic-R.

### 📘 3.1 Entorno y Preparación

**a.** Instalación del entorno

**Recomendamos encarecidamente utilizar un entorno de conda separado para el entrenamiento del agente (siguiendo Search-R1).**

```bash
# ---------------------------------- crear entorno ----------------------------------
conda create -n searchr1 python=3.10 -y
source ~/.bashrc
conda activate searchr1

# ---------------------------------- instalar paquetes ----------------------------------
pip install torch==2.4.0+cu118
pip3 install vllm==0.6.3
cd Search-R1
pip install -e .
pip install wandb
pip install flash_attn==2.7.3
pip install triton==3.0.0
pip install xformers==0.0.27.post2+cu118
```

**Recomendamos utilizar otro entorno de conda separado para el entrenamiento del recuperador.**

```bash
# ---------------------------------- crear entorno ----------------------------------
conda create -n tevatron python=3.10 -y
conda activate tevatron
cd tevatron
pip install -e .

# ---------------------------------- instalar paquetes ----------------------------------
pip install deepspeed==0.18.0
pip install accelerate
pip install transformers==4.57.1
pip install qwen_omni_utils
pip install peft
pip install torch==2.7.0
pip install faiss_gpu==1.7.3
pip install numpy==1.26.0
pip install uvicorn fastapi
```

**b.** Descarga el corpus de wiki `wiki18_100w.jsonl` desde [retrieval_corpus](https://modelscope.cn/datasets/lwhlwh/retrieval_corpus/files) y coloca estos archivos en `{WORKSPACE_DIR}/data/FlashRAG_Dataset/retrieval_corpus/`. 

**c.** Utiliza e5-base-v2 para construir el índice de wikipedia basándote en el siguiente script:

```shell
conda activate tevatron
model_name=e5-base-v2
model_path={WORKSPACE_DIR}/llm/$model_name
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m flashrag.retriever.index_builder \
    --retrieval_method ${model_name} \
    --model_path $model_path \
    --corpus_path {WORKSPACE_DIR}/data/FlashRAG_Dataset/retrieval_corpus/wiki18_100w.jsonl \
    --save_dir {WORKSPACE_DIR}/data/FlashRAG_Dataset/retrieval_corpus/ \
    --use_fp16 \
    --max_length 256 \
    --batch_size 256 \
    --faiss_type Flat \
    --sentence_transformer \
```

### 🔥 3.2 Entrenamiento del Agente de Búsqueda

#### 3.2.1 Iniciar el recuperador

Antes de entrenar el agente de búsqueda, ejecuta el siguiente código para iniciar un recuperador (usamos E5 en la primera iteración):

```shell
conda activate tevatron
cd Search-R1
bash retrieval_launch.sh
```

#### 3.2.2 Entrenar el Agente

Por defecto, utilizamos hotpotqa y triviaqa como nuestros datos de entrenamiento. Los conjuntos de datos de entrenamiento y prueba son generados por los scripts `qa_search_train_merge.py` y `qa_search_test_merge.py` bajo el directorio `Search-R1/scripts/data_process`. También puedes descargar nuestros datos preprocesados (training.parquet y test.parquet) desde [aquí](https://huggingface.co/datasets/liuwenhan/Agent_training_data/tree/main) y colocarlos bajo el directorio `Search-R1/scripts/data_process/data/`.

```shell
conda activate searchr1
bash train_ppo.sh
```

*Nota: Este script también incluye códigos para entrenar el agente basándose en nuestro Agentic-R, el cual envía 'Question [SEP] query' al recuperador como consulta. En este código, el parámetro `retriever.agentic_retriever_input` se establece como `true`.*

### 🔥 3.3 Entrenamiento de Agentic-R

#### 3.3.1 Generar Datos de Entrenamiento

**a. Generar trayectoria del agente de búsqueda**

```shell
conda activate agentic-R
cd FlashRAG/examples/methods
bash step1_generate_trajectory.sh
```

**b. Generar pasajes candidatos**

En esta parte, para cada consulta generada por el agente de búsqueda, utilizamos un recuperador denso para recuperar pasajes de entrenamiento (para la primera iteración, el recuperador es E5; para la segunda iteración, el recuperador es el Agentic-R entrenado tras la primera iteración).

```shell
conda activate agentic-R
bash step2_generate_passage_candidates.sh
```

**c-1. generar relevancia local (utilidad del pasaje 1)**

```shell
conda activate agentic-R
# primero genera la sub-respuesta usando Qwen-72B-Instruct
bash step3-0_generate_subanswer.sh
# luego puntúa los pasajes candidatos
bash step3-1_generate_local_utility.sh
```

**c-2. generar corrección de la respuesta final (utilidad del pasaje 2)**

```shell
conda activate agentic-R
bash step3-2_generate_global_utility.sh
```

**d. construir datos de entrenamiento del recuperador**

```python
python step4_construct_retriever_data.py
```

También proporcionamos los datos de entrenamiento finales para el entrenamiento de la primera iteración; puedes descargarlos desde [aquí](https://huggingface.co/datasets/liuwenhan/retriever_training_data/tree/main) y colocarlos bajo el directorio `FlashRAG/examples/methods/training_data/` .

#### 3.3.2 Entrenamiento del Recuperador

```shell
cd tevatron/scripts/
bash train_agentic-R.sh
# el parámetro agentic_retriever_input se establece como True, lo que controla la entrada de la consulta del recuperador.
```

Después del entrenamiento, utiliza el siguiente código para construir el índice:

```shell
cd FlashRAG/scripts/
bash build_index_after_train.sh
```

## 📄 Citación

Si encuentras este trabajo útil, por favor cita nuestros artículos:
```bibtex
@article{liu2026agentic,
  title={Agentic-R: Learning to Retrieve for Agentic Search},
  author={Liu, Wenhan and Ma, Xinyu and Zhu, Yutao and Li, Yuchen and Shi, Daiting and Yin, Dawei and Dou, Zhicheng},
  journal={arXiv preprint arXiv:2601.11888},
  year={2026}
}
```
## 🤝 Agradecimientos

Nuestros códigos están construidos sobre [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG), [Search-R1](https://github.com/PeterGriffinJin/Search-R1) y [tevatron](https://github.com/texttron/tevatron). Nuestro trabajo se basa en la serie de modelos [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), y agradecemos sinceramente al equipo de Qwen por sus destacadas contribuciones a la comunidad de código abierto.


## 📄 Licencia

Este proyecto se publica bajo la [Licencia MIT](LICENSE).

## 📞 Contacto

Para cualquier pregunta o comentario, por favor contáctanos en [lwh@ruc.edu.cn](lwh@ruc.edu.cn).

## Historial de Estrellas

[![Star History Chart](https://api.star-history.com/svg?repos=8421BCD/Agentic-R&type=timeline&legend=top-left)](https://www.star-history.com/#8421BCD/Agentic-R&type=timeline&legend=top-left)
