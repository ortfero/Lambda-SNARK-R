# Lean 4 Formal Verification

Эта директория содержит Lean 4-проект, подтверждающий корректность
вероятностного форкинг-инфраструктуры ΛSNARK-R. Все основные леммы
получены конструктивно; Lean-кernel не использует сторонних аксиом.

## Структура

```
formal/
├── lakefile.lean          # Конфигурация Lake
├── Main.lean              # Точка входа (компилирует весь проект)
├── LambdaSNARK.lean       # Корневой модуль
└── LambdaSNARK/
    ├── ForkingInfrastructure.lean  # heavy-row, fork-success, extraction
    ├── Core.lean                    # R1CS, свидетели, модель противника
    ├── Polynomial.lean              # Инструменты для полиномов & Z_H
    ├── Soundness.lean               # Сборка теоремы знания
    └── tests/                       # Примеры сертификатов
```

## Сборка и проверки

```bash
# Сборка библиотеки и регрессионных лемм
lake build LambdaSNARK

# Изучить конкретный модуль в интерактивном режиме (пример)
lake env lean LambdaSNARK/ForkingInfrastructure.lean

# Прогнать Lean-тесты/примеры
lake env lean tests/ForkingCertificateExample.lean
```

## Текущее состояние (ноябрь 2025)

- ✅ Лемма о вероятности успеха: `successProbability_eq_successfulRandomness_card_div`
- ✅ Heavy-row лемма: конструктивная реконструкция «тяжёлых» коммитментов
- ✅ Форкинг с нижней границей: `fork_success_bound` (ε²/2 − 1/|F|)
- ✅ Экстракция свидетеля: `tests/ForkingCertificateExample.lean`
- ✅ Healthcare-quotient сертификат: `tests/HealthcareQuotient.lean`
- ✅ Полиномиальный инструментарий для деления на Z_H и связок с PMF
- ⏳ Интеграция в `Soundness.lean` (полная теорема знания)
- ⏳ Completeness & zero-knowledge (следующие этапы)

## Дорожная карта

- Связать доказанные вероятностные леммы с `Soundness.lean`
- Добавить новые тестовые схемы (PLAQUETTE, LWE, дополнительные R1CS)
- Формализовать completeness и zero-knowledge внутри Lean 4
- Поддерживать CI с `lake build LambdaSNARK` на каждый коммит

## Ключевые результаты

### Heavy-row & Fork Success

- `successProbability_eq_successfulRandomness_card_div`
- `heavy_row_lemma`
- `fork_success_bound`

Эти утверждения образуют сердцевину форкинг-аргумента и уже формально доказаны.

### Экстракция свидетеля

```lean
theorem extraction_soundness … :=
  …
```

Полная связка с реальными примерами доступна в `tests/ForkingCertificateExample.lean`
и `tests/HealthcareQuotient.lean`.

## Источники

- Lean 4 manual: https://leanprover.github.io/lean4/doc/
- Mathlib4: https://github.com/leanprover-community/mathlib4 (модули `PMF`, `ENNReal`, `Polynomial`)
- Обзор проекта ΛSNARK-R: ../README.md
