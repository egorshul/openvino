# Подготовка моделей для OpenVINO 2026.1 — статический режим / NPU (FP16)

Тулкит готовит IR трёх моделей так, чтобы они запускались на устройствах
**без рантайм-динамизма** (целевое устройство — Intel **NPU**), причём модели
**принимают входы beam search** (`beam_idx`):

| Модель | HF id | Что получаем |
|---|---|---|
| Stable Diffusion XL | `stabilityai/stable-diffusion-xl-base-1.0` | Полностью **статический** FP16 IR (фикс. batch/разрешение). Beam search неприменим (диффузия). |
| Whisper large v3 | `openai/whisper-large-v3` | FP16 IR, encoder со статичным mel-входом, stateful decoder с `beam_idx`. |
| Llama 3.1 8B Instruct | `meta-llama/Llama-3.1-8B-Instruct` | **Stateful** FP16 IR с входом `beam_idx`; статическая компиляция — пайплайном GenAI NPU. |

> **Готовите MLPerf Inference?** Используйте флаг `--mlperf` — см. раздел
> [«MLPerf Inference»](#mlperf-inference-closed-datacenter) ниже. Важно: MLPerf
> требует **greedy-декодирования, а не beam search**, и точных коммитов чекпойнтов.

## Версии (зафиксированы и совместимы)

```
openvino            == 2026.1.0
openvino-tokenizers == 2026.1.0.0
openvino-genai      == 2026.1.0.0
optimum-intel       == 1.27.0      # требует openvino >= 2025.4.0  ⇒  2026.1 OK
nncf                >= 2.19.0
transformers         >= 4.45, < 4.58
```

## Установка

```bash
cd tools/npu_static_model_prep
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

> CPU-сборка PyTorch указана в `requirements.txt` — для экспорта GPU не нужен.

## Требования к ресурсам

* **Диск:** ~60 GB свободно (исходники с HF + IR + архив).
* **RAM:** **≥ 32 GB** для экспорта Llama 3.1 8B в FP16 (≈16 GB только весов;
  на 15 GB будет OOM). SDXL и Whisper укладываются в 16 GB.
* **HF-токен:** Llama 3.1 — **gated**. Примите лицензию Meta на странице модели
  и задайте токен:
  ```bash
  export HF_TOKEN=hf_xxx
  ```

## Запуск

Всё сразу + упаковка архива:

```bash
./run_all.sh
```

Или по отдельности:

```bash
# SDXL — полностью статический FP16
python export_sdxl.py    --batch-size 1 --height 1024 --width 1024

# Whisper large v3 — FP16; --num-beams фиксирует batch декодера статически
python export_whisper.py --num-beams 5

# Llama 3.1 8B — stateful FP16 (с beam_idx). Нужен HF_TOKEN.
python export_llm.py     --weight-format fp16
```

Проверка форм и наличия `beam_idx`:

```bash
python verify_static.py models/*-ov-*
```

Упаковка в архив с манифестом и sha256:

```bash
./make_archive.sh models openvino-2026.1-npu-models-fp16.tar.gz
```

## Что именно означает «нет рантайм-динамизма» и «beam search входы»

### Llama 3.1 8B (и в целом causal LLM)
* Экспорт идёт **stateful** (по умолчанию). У stateful causal-LM есть вход
  **`beam_idx`** — это и есть «вход beam search»: на каждом шаге рантайм передаёт
  `beam_idx`, чтобы переставить внутренний KV-кэш под выбранные лучи, **без
  перекомпиляции графа**.
* optimum-intel **сознательно не запекает** полностью статические формы в IR
  causal-LM (сообщение «Static shapes are not supported for causal language
  model»), потому что prefill (seq=N) и decode (seq=1) требуют разных форм.
* Поэтому отсутствие рантайм-динамизма обеспечивает **пайплайн OpenVINO GenAI
  для NPU**, который из этого же IR статически компилирует два подграфа
  (prefill / decode):

  ```python
  import openvino_genai as ov_genai
  pipe = ov_genai.LLMPipeline(
      "models/llama-3.1-8b-instruct-ov-fp16", "NPU",
      MAX_PROMPT_LEN=1024,      # фиксированная длина промпта (статика)
      MIN_RESPONSE_LEN=256,     # фиксированная длина ответа (статика)
  )
  cfg = pipe.get_generation_config()
  cfg.num_beams = 5             # beam search через присутствующий beam_idx
  cfg.num_return_sequences = 1
  print(pipe.generate("Hello", cfg))
  ```

  > Для NPU обычно компрессуют веса (`--weight-format int4`/`int8`) — это меньше
  > и быстрее. Здесь по вашему требованию используется FP16; при необходимости
  > просто поменяйте флаг в `export_llm.py`.

### Whisper large v3
* Encoder получает фиксированный лог-мел вход `[batch, 128, 3000]` — статичен.
* Stateful decoder имеет `beam_idx`; `--num-beams N` дополнительно фиксирует
  размер по лучам. Рантайм:

  ```python
  import openvino_genai as ov_genai
  pipe = ov_genai.WhisperPipeline("models/whisper-large-v3-ov-fp16", "NPU")
  cfg = pipe.get_generation_config(); cfg.num_beams = 5
  print(pipe.generate(raw_audio_16k, cfg))
  ```

### Stable Diffusion XL
* Beam search неприменим. «Статика» = фиксированные batch, разрешение и число
  изображений на промпт. `pipe.reshape(...)` корректно учитывает удвоение batch
  UNet при classifier-free guidance. Все суб-модели становятся полностью
  статическими (проверяется `verify_static.py`).

## Точные команды экспорта (под капотом)

```bash
# Llama 3.1 8B — stateful FP16 (beam_idx сохраняется)
optimum-cli export openvino -m meta-llama/Llama-3.1-8B-Instruct \
    --task text-generation-with-past --weight-format fp16 \
    models/llama-3.1-8b-instruct-ov-fp16

# Whisper large v3 — FP16
optimum-cli export openvino -m openai/whisper-large-v3 \
    --task automatic-speech-recognition --weight-format fp16 \
    models/whisper-large-v3-ov-fp16

# SDXL — экспорт + статический reshape + FP16 (через optimum API, см. export_sdxl.py)
```

## MLPerf Inference (Closed, Datacenter)

Эти три модели — бенчмарки MLPerf Inference. Тулкит умеет выравнивать аргументы
под reference-реализации MLCommons. Включается флагом `--mlperf` (или `MLPERF=1
./run_all.sh`); пресеты в `mlperf_presets.py`.

```bash
export HF_TOKEN=hf_xxx
MLPERF=1 ./run_all.sh
# или поштучно:
python export_llm.py     --mlperf
python export_whisper.py --mlperf
python export_sdxl.py    --mlperf --revision <commit_вашего_раунда>
```

Что делает `--mlperf`:

* **Пинит точный коммит чекпойнта** (snapshot-download → экспорт из локального пути,
  т.к. у `optimum-cli` нет `--revision`):
  * Llama 3.1 8B — `be673f326cab4cd22ccfef76109faf68e41aa5f1`
  * Whisper v3 — `06f233fe06e710322aca913c1bc4249a0d71fce1`
  * SDXL — reference снапшотит HF-пайплайн; зафиксируйте коммит своего раунда через
    `--revision` (в README MLCommons фиксированного хеша нет).
* **Нацеливает на GREEDY** (это критично для Closed): `num_beams=1`,
  `do_sample=False`. У Whisper `--num-beams` принудительно игнорируется. Вход
  `beam_idx` остаётся в IR, но **не используется** — модель валидна.

Reference-параметры MLPerf (из исходников MLCommons):

| Модель | Декодирование | Длины / рантайм | Точность (Datacenter) |
|---|---|---|---|
| Llama 3.1 8B | greedy, bf16 reference | вход паддинг до 1024 ток. | ROUGE1≥38.78, R2≥15.91, RL≥24.50, RLsum≥35.79, gen_len 90% (99% порог) |
| Whisper v3 | greedy, `temperature=0`, `max_new_tokens=200` | 30 c аудио, mel 128×3000, `max_model_len=448` | WER ≤ 2.0671% / 99% от reference |
| SDXL | EulerDiscreteScheduler, 20 шагов, guidance=8, 1024×1024, заданный negative prompt, latents с внешним сидом | fp32 reference (fp16/bf16 ок) | FID∈[23.011, 23.950], CLIP∈[31.686, 31.813] |

### Что НЕ закрывает тулкит (нужно для валидного Closed-сабмишна)
* **LoadGen-харнес** и сценарии (Offline/Server), `user.conf`, equal-issue.
* **Официальные датасеты** (CNN/DailyMail, LibriSpeech, COCO-2014) и препроцессинг.
* **Accuracy-скрипты** MLPerf (ROUGE / WER / FID+CLIP) и калибровочный датасет (если
  квантуете — для Datacenter quantization разрешён с калибровкой по правилам).
* **Проверка точности FP16**: reference у Llama — bf16; FP16-веса допустимы как
  оптимизация, но обязаны держать 99% порог. Если не проходит — берите bf16/int8
  с калибровкой.

> Источники: MLCommons inference (`language/llama3.1-8b`, `speech2text`,
> `text_to_image`, `backend_pytorch.py`) и `inference_policies/inference_rules.adoc`.
> Пороги/коммиты меняются по раундам — сверяйтесь с правилами своего раунда.

## Замечания
* `--stateless` в `export_llm.py` добавляет `--disable-stateful` (явные
  past/present KV вместо stateful). Использовать только если ваш рантайм не умеет
  stateful-модели — **этот режим убирает `beam_idx`**.
* Сам многогигабайтный архив в репозиторий не коммитится; выгружайте его в своё
  хранилище (HF Hub / облако).
