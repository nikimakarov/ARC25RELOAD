# ARC25RELOAD

Проект для решения задач из [ARC (Abstraction and Reasoning Challenge)](https://github.com/fchollet/ARC-AGI) с использованием fine-tuning языковых моделей.

## 📋 Описание

Этот репозиторий содержит код и notebooks для:
1. **Создания датасетов** для обучения LLM на задачах ARC
2. **Fine-tuning моделей** Qwen2.5 на двух типах задач
3. **Test-time fine-tuning** для улучшения точности на конкретных задачах

## 🗂 Структура проекта

```
ARC_RELOAD/
├── arc24-source-code/          # Исходный код утилит и обучения
│   ├── arc24/                  # Модули для работы с данными
│   │   ├── data.py            # Загрузка данных ARC
│   │   ├── prompting.py       # Генерация промптов
│   │   ├── encoders.py        # Кодирование сеток
│   │   └── data_augmentation.py
│   ├── fine-tuning.py         # Основной скрипт fine-tuning
│   └── inference.py           # Inference скрипт
├── kaggle/input/              # Входные данные
│   └── arc-prize-2025/        # Датасет ARC 2025
├── 1_make_datasets.ipynb      # Создание датасетов
├── 2_training_and_validation.ipynb  # Обучение модели
└── 3_test_time_fine_tuning.ipynb    # Test-time fine-tuning
```

## 🚀 Быстрый старт

### 1. Создание датасетов

Notebook `1_make_datasets.ipynb` создает 6 датасетов для двух задач:

**Task 1: examples + input → output** (основная задача предсказания)
- `task1_train.parquet`
- `task1_validation.parquet`
- `task1_test.parquet`

**Task 2: inputs → input** (генерация новых сеток)
- `task2_train.parquet`
- `task2_validation.parquet`
- `task2_test.parquet`

Каждый датасет содержит:
- `text` - сгенерированный промпт
- `task_id` - идентификатор задачи
- `test_sample_index` - индекс тестового примера
- `prompt_length` - длина промпта в токенах
- `num_train_samples` - количество примеров
- `input_height/width` - размеры входной сетки
- `output_height/width` - размеры выходной сетки
- `ground_truth_output` - ground truth для оценки (JSON)

### 2. Конфигурация (CFG)

Использует класс `CFG` из `fine-tuning.py`:

```python
cfg = CFG(
    model_path='Qwen/Qwen2.5-0.5B-Instruct',
    grid_encoder='GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))',
    max_seq_len=4096,
    learning_rate=1e-4,
    lora_r=32,
    # ... другие параметры
)
```

### 3. Обучение модели

```bash
# Используя notebook
jupyter notebook 2_training_and_validation.ipynb

# Или через командную строку
python arc24-source-code/fine-tuning.py \
    --model_path Qwen/Qwen2.5-0.5B-Instruct \
    --train_datasets path/to/task1_train.parquet output-from-examples-v1 \
    --val_dataset path/to/task1_validation.parquet output-from-examples-v1
```

## 📊 Типы задач

### Task 1: Output from Examples (основная)
Модель учится понимать паттерн трансформации из примеров и применять его к новым входам.

**Prompt template:** `output-from-examples-v1`

### Task 2: Input Generation
Модель учится генерировать новые сетки, следуя распределению входных данных.

**Prompt template:** `input-from-inputs-v0`

## 🔧 Технологии

- **Model:** Qwen2.5-0.5B-Instruct
- **Training:** LoRA + SFT (Supervised Fine-Tuning)
- **Grid Encoding:** GridShapeEncoder + RowNumberEncoder + MinimalGridEncoder
- **Framework:** Transformers, PEFT, TRL

## 📦 Зависимости

```bash
pip install transformers datasets pandas torch peft trl
```

Или используйте:
```bash
bash install_libraries_local.sh
```

## 🎯 Метаданные и Ground Truth

Все датасеты содержат колонку `ground_truth_output` с правильными ответами для:
- **Training:** Обучение и верификация
- **Validation:** Оценка качества модели во время обучения
- **Test:** Финальная оценка (если доступны решения)

Для извлечения ground truth:
```python
import pandas as pd
import json

df = pd.read_parquet('datasets_qwen25/task1_validation.parquet')

# Извлечь ground truth для задачи
for _, row in df.iterrows():
    gt_grid = json.loads(row['ground_truth_output'])
    # Сравнить с предсказанием модели
```

## 📝 .gitignore

Репозиторий настроен игнорировать:
- Большие файлы моделей (`*.safetensors`, `*.bin`)
- Директории с моделями (`qwen2.5/`, `loras/`)
- Временные файлы (`tmp/`, `output/`)
- Conda окружения (`.conda/`)
- Предсобранные wheels (`making-wheels-of-necessary-packages-for-vllm/`)

## 🔗 Ссылки

- [ARC Challenge](https://github.com/fchollet/ARC-AGI)
- [ARC Prize 2025](https://arcprize.org/)
- [Qwen2.5](https://huggingface.co/Qwen)

## 📄 Лицензия

MIT

## 👤 Автор

Nikita Makarov (@nikimakarov)

---

⭐ Если проект полезен, поставьте звезду на GitHub! 