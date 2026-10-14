# Воспроизведение

## Установка

Рекомендуемая версия — Python 3.13. CI проверяет Python 3.11 и 3.13. GPU для проекта не требуется; основные вычисления выполняет CPU CatBoost.

```bash
git clone https://github.com/gimacorp/recsys-auto-features.git
cd recsys-auto-features
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

`pyproject.toml` фиксирует прямые зависимости. `requirements.lock.txt` сохраняет версии всех пакетов фактического окружения Python 3.13. Для воспроизведения этого окружения:

```bash
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps -e .
```

## Проверки и короткий запуск

```bash
ruff check src tests
ruff format --check src tests
pytest -q
recsys download
recsys run --config configs/smoke.json --output artifacts/smoke-local
```

Короткий запуск проверяет все основные этапы на полном датасете с сокращённым бюджетом моделей и поиска. Его метрики имеют диагностический статус.

## Полное исследование

Опубликованные артефакты находятся в `artifacts/full`. Новый запуск сохраняйте в отдельный каталог, чтобы сравнить его с исходными результатами:

```bash
recsys run --config configs/full.json --output artifacts/reproduced
python -m recsys_auto.audit --experiment artifacts/reproduced
python -m recsys_auto.from_static --experiment artifacts/reproduced
python -m recsys_auto.catalogue --experiment artifacts/reproduced
python -m recsys_auto.diagnostics --experiment artifacts/reproduced
python -m recsys_auto.report --experiment artifacts/reproduced --output docs/reproduced
```

Основной запуск выполняет шесть настроек ручного бустинга на каждом fold, 180 проверок дополнительных пакетов и финальное обучение пяти моделей для каждого fold/seed. Дополнительный поиск с исходными статическими признаками выполняет ещё 288 проверок пакетов. Фактическое время и среда сохранены в результатах; порядок времени на CPU — минуты, продолжительность зависит от машины и нагрузки.

Незавершённый основной запуск продолжается командой:

```bash
recsys run --config configs/full.json --output artifacts/reproduced --resume
```

Продолжение использует checkpoint после завершённого fold/seed и проверяет совпадение конфигурации, исходного кода и архива. Прерванная пара fold/seed выполняется заново. Дополнительные эксперименты запускаются после завершения основного; для них повторная команда выполняет весь соответствующий эксперимент.

## Артефакты

- `results.json`, `fold*-seed*.json`: метрики, параметры, столбцы, все попытки основного поиска и интервалы.
- `tuning-fold*.json`: полный журнал настройки базовой модели.
- `catalogue.json`: ранжирование доступного каталога на фиксированный момент.
- `diagnostics.json`: абляции и замороженная история.
- `from_static.json`: поиск с исходными статическими признаками.
- `provenance.json`: SHA-256 данных, конфигурации и модулей; Git revision, Python и версии библиотек.
- `audit.json`: результат пересчёта основных метрик из сохранённых предсказаний.
- `feature_stats.json`: время расчёта каждого пакета и размер матрицы.
- `artifacts/models/<имя-запуска>` и `artifacts/predictions/<имя-запуска>`: локальные модели и предсказания, исключённые из Git.
- `docs/generated_results.md` и `docs/figures`: таблицы и графики, сгенерированные из JSON.

Публикуются агрегированные результаты. Исходные данные скачиваются с GroupLens; модели и индивидуальные предсказания создаются локально при воспроизведении. Самостоятельная генерация отчёта использует сохранённые JSON, а аудит и дополнительные эксперименты требуют локального выполнения основной команды обучения.

## Детерминизм

Фиксированы seed, порядок событий, конфигурации и количество потоков. Одинаковые timestamp сортируются стабильным порядком user/item, но исключаются из истории друг друга. Межплатформенные различия CatBoost и BLAS могут дать небольшие численные изменения. Bootstrap повторяется с фиксированным seed 2026; его интервалы оценивают вариативность пользователей конкретного тестового периода.

В provenance основного запуска зафиксированы модули, существовавшие в момент его старта. Дополнительные модули анализа и отчёта добавлялись отдельными коммитами и имеют собственную provenance там, где они выполняют эксперименты. История Git связывает сохранённые результаты с этапами разработки.
