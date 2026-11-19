# B123 · Student на ответах Teacher B

`Qwen/Qwen2.5-0.5B-Instruct` прошла две эпохи LoRA SFT на 1 000 ответах Teacher B с seed 123 и 126 шагами оптимизатора. Обучение заняло 1 357,3 секунды на NVIDIA GeForce RTX 5060. На 150 задачах validation получен loss 2,9583. На общем тесте B123 решила 68 из 200 задач (34,0%). [Сравнение с baseline и B42](../member-04/notes.md)

- [Файлы LoRA-адаптера](adapter) содержат веса и конфигурацию для точной ревизии базовой модели. SHA-256 весов `a232400833c95577c6aaba379859bf08e7aaa117cd5f55d77e1f4c0caff8c409`.
- [Конфигурация запуска](run_config.json) и [хеши входов](inputs.json) фиксируют данные, seed, параметры, версии и устройство.
- [Журнал шагов](trainer_state.json) и [итоги обучения](run_results.json) содержат значения loss, время и пиковую видеопамять.

Для оценки из корня репозитория используйте командный архив и путь к опубликованному адаптеру.

```sh
python evaluate.py --data project_data.zip --output runs/eval-B123 --adapter B123=results/B123/adapter
```

Токенизатор загружается из исходной ревизии Qwen. [Карточка базовой модели](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775) · [Лицензия модели](LICENSE-base-model.txt)
