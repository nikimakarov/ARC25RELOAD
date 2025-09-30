#!/usr/bin/env python3
"""
Создание небольшого датасета для тестирования Test-time Fine-tuning (TTFT)
для соревнования ARC25

Этот модуль предоставляет функционал для:
1. Создания небольшого тестового датасета из оригинального ARC датасета
2. Генерации N-1 датасета для TTFT обучения
3. Валидации структуры данных

Автор: nikimakarov
Дата: 2024
"""

import json
import random
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class TTFTDatasetCreator:
    """
    Класс для создания тестовых датасетов для Test-time Fine-tuning
    """
    
    def __init__(self, random_seed: Optional[int] = 42):
        """
        Инициализация создателя датасета
        
        Args:
            random_seed: Seed для воспроизводимости результатов
        """
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
    
    def load_dataset(self, input_path: str) -> Dict:
        """
        Загружает датасет из JSON файла
        
        Args:
            input_path: путь к JSON файлу с датасетом
            
        Returns:
            Словарь с задачами ARC
        """
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
            print(f"✅ Загружен датасет из {input_path}: {len(data)} задач")
            return data
        except FileNotFoundError:
            print(f"❌ Файл не найден: {input_path}")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ Ошибка парсинга JSON: {e}")
            raise
    
    def validate_task(self, task_id: str, task: Dict) -> bool:
        """
        Валидирует структуру задачи ARC
        
        Args:
            task_id: идентификатор задачи
            task: данные задачи
            
        Returns:
            True если задача валидна, False иначе
        """
        required_keys = ['train', 'test']
        
        if not all(key in task for key in required_keys):
            print(f"⚠️  Задача {task_id} не содержит обязательные ключи: {required_keys}")
            return False
        
        if not isinstance(task['train'], list) or len(task['train']) == 0:
            print(f"⚠️  Задача {task_id}: train должен быть непустым списком")
            return False
        
        if not isinstance(task['test'], list) or len(task['test']) == 0:
            print(f"⚠️  Задача {task_id}: test должен быть непустым списком")
            return False
        
        # Проверяем структуру train примеров
        for i, example in enumerate(task['train']):
            if not isinstance(example, dict) or 'input' not in example or 'output' not in example:
                print(f"⚠️  Задача {task_id}: train[{i}] должен содержать 'input' и 'output'")
                return False
        
        # Проверяем структуру test примеров
        for i, example in enumerate(task['test']):
            if not isinstance(example, dict) or 'input' not in example:
                print(f"⚠️  Задача {task_id}: test[{i}] должен содержать 'input'")
                return False
        
        return True
    
    def create_small_dataset(self, 
                           input_path: str, 
                           output_path: str, 
                           num_tasks: int = 5,
                           min_train_examples: int = 2) -> Dict:
        """
        Создает небольшой датасет из случайных задач
        
        Args:
            input_path: путь к оригинальному датасету
            output_path: путь для сохранения маленького датасета
            num_tasks: количество задач для тестирования
            min_train_examples: минимальное количество train примеров
            
        Returns:
            Словарь с выбранными задачами
        """
        # Загружаем оригинальный датасет
        full_data = self.load_dataset(input_path)
        
        # Фильтруем задачи по критериям
        valid_tasks = {}
        for task_id, task in full_data.items():
            if (self.validate_task(task_id, task) and 
                len(task['train']) >= min_train_examples):
                valid_tasks[task_id] = task
        
        print(f"📊 Найдено {len(valid_tasks)} валидных задач из {len(full_data)}")
        
        if len(valid_tasks) < num_tasks:
            print(f"⚠️  Недостаточно валидных задач. Запрошено: {num_tasks}, доступно: {len(valid_tasks)}")
            num_tasks = len(valid_tasks)
        
        # Выбираем случайные задачи
        task_ids = list(valid_tasks.keys())
        selected_tasks = random.sample(task_ids, num_tasks)
        
        # Создаем маленький датасет
        small_dataset = {task_id: valid_tasks[task_id] for task_id in selected_tasks}
        
        # Сохраняем
        self.save_dataset(small_dataset, output_path)
        
        print(f"✅ Создан датасет с {len(small_dataset)} задачами:")
        for task_id in selected_tasks:
            task = small_dataset[task_id]
            print(f"  📋 {task_id}: {len(task['train'])} train, {len(task['test'])} test")
        
        return small_dataset
    
    def create_n_minus_1_dataset(self, data: Dict) -> Dict:
        """
        Создает N-1 датасет для TTFT (первый train пример становится test)
        
        Args:
            data: исходный датасет
            
        Returns:
            N-1 датасет для TTFT
        """
        new_data = {}
        skipped_tasks = []
        
        for task_id, task in data.items():
            if len(task['train']) < 2:
                skipped_tasks.append(task_id)
                continue
            
            new_data[task_id] = {
                'train': task['train'][1:],  # Берем все train примеры кроме первого
                'test': task['train'][:1],   # Первый train пример становится "test"
            }
        
        if skipped_tasks:
            print(f"⚠️  Пропущено {len(skipped_tasks)} задач (недостаточно train примеров): {skipped_tasks}")
        
        print(f"✅ Создан N-1 датасет с {len(new_data)} задачами")
        return new_data
    
    def save_dataset(self, data: Dict, output_path: str) -> None:
        """
        Сохраняет датасет в JSON файл
        
        Args:
            data: данные для сохранения
            output_path: путь для сохранения
        """
        # Создаем директорию если не существует
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Датасет сохранен: {output_path}")
    
    def create_ttft_test_datasets(self, 
                                input_path: str,
                                output_dir: str = ".",
                                num_tasks: int = 5,
                                min_train_examples: int = 2) -> Tuple[Dict, Dict]:
        """
        Создает полный набор датасетов для тестирования TTFT
        
        Args:
            input_path: путь к оригинальному датасету
            output_dir: директория для сохранения файлов
            num_tasks: количество задач для тестирования
            min_train_examples: минимальное количество train примеров
            
        Returns:
            Кортеж (original_dataset, n_minus_1_dataset)
        """
        print("�� Создание тестовых датасетов для TTFT...")
        
        # Пути к файлам
        small_output = os.path.join(output_dir, 'small_test_dataset.json')
        n_minus_1_output = os.path.join(output_dir, 'small_test_dataset_n-1.json')
        
        # Создаем маленький тестовый датасет
        small_dataset = self.create_small_dataset(
            input_path, small_output, num_tasks, min_train_examples
        )
        
        # Создаем N-1 версию для TTFT
        n_minus_1_dataset = self.create_n_minus_1_dataset(small_dataset)
        
        # Сохраняем N-1 датасет
        self.save_dataset(n_minus_1_dataset, n_minus_1_output)
        
        print(f"\n📁 Файлы сохранены в {output_dir}:")
        print(f"  - {os.path.basename(small_output)} (оригинальный датасет)")
        print(f"  - {os.path.basename(n_minus_1_output)} (N-1 датасет для TTFT)")
        
        return small_dataset, n_minus_1_dataset


def main():
    """
    Основная функция для создания тестовых датасетов
    """
    # Конфигурация
    input_path = '/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json'
    output_dir = '.'
    num_tasks = 5
    random_seed = 42
    
    # Создаем экземпляр создателя датасета
    creator = TTFTDatasetCreator(random_seed=random_seed)
    
    try:
        # Создаем тестовые датасеты
        original_dataset, n_minus_1_dataset = creator.create_ttft_test_datasets(
            input_path=input_path,
            output_dir=output_dir,
            num_tasks=num_tasks,
            min_train_examples=2
        )
        
        print("\n�� Готово! Теперь можно использовать:")
        print("1. small_test_dataset.json - для обычного тестирования")
        print("2. small_test_dataset_n-1.json - для TTFT обучения")
        
        return original_dataset, n_minus_1_dataset
        
    except Exception as e:
        print(f"❌ Ошибка при создании датасетов: {e}")
        raise


if __name__ == "__main__":
    main()