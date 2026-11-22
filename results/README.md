# Результаты обучения Student

Для сравнения трёх Teacher при `seed=42` используй адаптеры из таблицы. Каждый запуск начался с одной ревизии `Qwen/Qwen2.5-0.5B-Instruct`, прошёл 2 эпохи и 126 шагов на 1 000 общих задачах. Validation loss рассчитан на 150 задачах. [Датасеты Teacher](../data/teachers/README.md)

| Запуск | Адаптер для оценки | Данные запуска | Train loss | Validation loss |
| --- | --- | --- | ---: | ---: |
| A42 · Teacher A | [`A42/adapter`](A42/adapter/) | [Конфиг](A42/run_config.json) · [метрики](A42/run_results.json) · [логи](A42/trainer_state.json) | 0,2313 | 2,2396 |
| B42 · Teacher B, повтор от 22 ноября | [`B42/rerun/adapter`](B42/rerun/adapter/) | [Конфиг](B42/rerun/run_config.json) · [метрики](B42/rerun/run_results.json) · [логи](B42/rerun/trainer_state.json) | 0,2305 | 2,8204 |
| C42 · Teacher C | [`C42/adapter`](C42/adapter/) | [Конфиг](C42/run_config.json) · [метрики](C42/run_results.json) · [логи](C42/trainer_state.json) | 0,2459 | 2,2763 |

## Оценка

Для этой тройки предстоит посчитать test accuracy на 200 задачах из [командного архива](../data/project_data.zip). Для запуска оценки передай участнику № 4 пути к адаптерам из таблицы. [Протокол оценки](B42/README.md)

Первый B42 от 18 ноября хранится в [`B42/adapter`](B42/adapter/) для воспроизведения уже проведённой оценки. Результат 74/200 из PR № 2 относится к первому адаптеру по SHA-256. B123 и результаты его оценки также доступны в этом PR. [PR № 2](https://github.com/Gionicerr/PLACE_HOLDER/pull/2)
