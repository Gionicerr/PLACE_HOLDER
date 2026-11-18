# B42 — Student на ответах Teacher B

Участник № 3 обучил `Qwen/Qwen2.5-0.5B-Instruct` с `seed=42` на 1 000 ответах Teacher B (`Qwen2.5-3B-Instruct`, GGUF Q4_K_M). Для обучения использовался общий [train.py](../../train.py), для входных данных и запуска — [пакет B42 из ветки участника № 2](https://github.com/Gionicerr/PLACE_HOLDER/tree/bae8237b5782fda37563dccb86254858a620e1a1/B42).

| Показатель | Результат |
| --- | ---: |
| Эпохи и шаги оптимизатора | 2 эпохи, 126 шагов |
| Train loss | 0,2305 |
| Validation loss на 150 задачах | 2,8204 |
| Время обучения на Tesla T4 | 538,6 с |
| Пиковая видеопамять, выделенная PyTorch | 5,31 ГиБ |
| Test accuracy | Ожидает оценки |

Точные значения сохранены в [run_results.json](run_results.json). [run_config.json](run_config.json) фиксирует ревизию модели, параметры и версии библиотек. [trainer_state.json](trainer_state.json) содержит историю шагов и loss.

## Передача участнику № 4

Используй 200 задач из `test.jsonl` командного архива `project_data.zip`. SHA-256 файла внутри архива — `d615995981e42a14e132cba14974637668c6344a356fee953584b658225fe138`. Запусти baseline и B42 одним скриптом на тех же задачах с одинаковыми промптами, настройками генерации и парсером ответов.

Для сопоставимости с A123 после условия задачи добавляй два перевода строки и общую инструкцию: `Solve the math problem. Show a short calculation. Finish with a separate line containing #### followed by the final number, with no units or additional text.` Используй шаблон диалога из токенизатора, жадное декодирование, лимит 192 новых токенов и числовую проверку финальной строки `#### число`. Правило разбора есть в [protocol.py участника № 2](https://github.com/Gionicerr/PLACE_HOLDER/blob/bae8237b5782fda37563dccb86254858a620e1a1/protocol.py).

Установи зависимости из [requirements.txt](../../requirements.txt) в отдельное окружение Python. Этот пример загружает адаптер из корня репозитория:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter = "results/B42/adapter"
tokenizer = AutoTokenizer.from_pretrained(adapter, padding_side="left")
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
    use_safetensors=True,
)
model = PeftModel.from_pretrained(base, adapter).eval()
```

Для baseline выполни генерацию в контексте `model.disable_adapter()`, для B42 — с активным адаптером. В итоговую таблицу внеси число верных ответов, test accuracy обеих моделей и разницу в процентных пунктах.
