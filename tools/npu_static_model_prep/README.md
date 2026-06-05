# Подготовка моделей для OpenVINO 2026.1 — статический режим / NPU (FP16)

Тулкит готовит IR трёх моделей так, чтобы они запускались на устройствах
**без рантайм-динамизма** (целевое устройство — Intel **NPU**), причём модели
**принимают входы beam search** (`beam_idx`):

| Модель | HF id | Что получаем |
|---|---|---|
| Stable Diffusion XL | `stabilityai/stable-diffusion-xl-base-1.0` | Полностью **статический** FP16 IR (фикс. batch/разрешение). Beam search неприменим (диффузия). |
| Whisper large v3 | `openai/whisper-large-v3` | FP16 IR, encoder со статичным mel-входом, stateful decoder с `beam_idx`. |
| Llama 3.1 8B Instruct | `meta-llama/Llama-3.1-8B-Instruct` | **Stateful** FP16 IR с входом `beam_idx`; статическая компиляция — пайплайном GenAI NPU. |

> **Готовите MLPerf Inference?** Идите по разделу **«Пошагово вручную»** ниже —
> он уже выровнен под MLPerf (Closed/Datacenter). Важно: MLPerf требует
> **greedy-декодирования, а не beam search**, и точных коммитов чекпойнтов.

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

## Пошагово вручную (без общего скрипта)

Ниже — полный путь «руками», без `run_all.sh`, выровненный под **MLPerf Inference
(Closed / Datacenter)**: пин коммитов чекпойнтов, FP16, **greedy** (без beam search).
Все команды — из каталога `tools/npu_static_model_prep` при активированном venv.

### Шаг 1. Окружение

```bash
cd tools/npu_static_model_prep
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
mkdir -p models
```

### Шаг 2. HF-токен (нужен для gated Llama 3.1)

Примите лицензию Meta на странице `meta-llama/Llama-3.1-8B-Instruct`, затем:

```bash
export HF_TOKEN=hf_xxx        # или: huggingface-cli login
```

### Шаг 3. Запинить точные коммиты чекпойнтов

У `optimum-cli` нет `--revision`, поэтому сначала скачиваем нужный коммит, а потом
экспортируем из локального пути. `huggingface-cli download` печатает путь снапшота:

```bash
LLAMA_DIR=$(huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --revision be673f326cab4cd22ccfef76109faf68e41aa5f1)

WHISPER_DIR=$(huggingface-cli download openai/whisper-large-v3 \
    --revision 06f233fe06e710322aca913c1bc4249a0d71fce1)

echo "$LLAMA_DIR"; echo "$WHISPER_DIR"
```

### Шаг 4. Экспорт Llama 3.1 8B → stateful FP16 IR

```bash
optimum-cli export openvino -m "$LLAMA_DIR" \
    --task text-generation-with-past --weight-format fp16 \
    models/llama-3.1-8b-instruct-ov-fp16
```

> Stateful-экспорт сохраняет вход `beam_idx`, но для MLPerf он **не используется**
> (greedy). Полностью статические формы в IR causal-LM не запекаются — статику даёт
> пайплайн GenAI NPU (Шаг 8).

### Шаг 5. Экспорт Whisper large v3 → FP16 IR (greedy)

```bash
optimum-cli export openvino -m "$WHISPER_DIR" \
    --task automatic-speech-recognition --weight-format fp16 \
    models/whisper-large-v3-ov-fp16
```

> Для MLPerf **не** добавляйте reshape под лучи: декодирование greedy, аудио
> паддится до 30 с (mel `[1,128,3000]`), `max_model_len=448`.

### Шаг 6. Экспорт SDXL → полностью статический FP16 IR

У SDXL reshape+half делаются через Python (CLI это не покрывает). Впишите коммит
своего раунда в `REV` (в README MLCommons фиксированного хеша нет):

```bash
python - <<'PY'
from optimum.intel import OVStableDiffusionXLPipeline
MID = "stabilityai/stable-diffusion-xl-base-1.0"
REV = None  # <- укажите коммит вашего MLPerf-раунда, напр. "462165..."
kw = {"revision": REV} if REV else {}
pipe = OVStableDiffusionXLPipeline.from_pretrained(MID, export=True, **kw)
pipe.reshape(batch_size=1, height=1024, width=1024, num_images_per_prompt=1)  # MLPerf: 1024x1024
pipe.half()                                                                   # FP16
pipe.save_pretrained("models/sdxl-base-1.0-ov-fp16-static")
print("saved")
PY
```

### Шаг 7. Проверка форм (static / beam_idx)

```bash
python verify_static.py models/llama-3.1-8b-instruct-ov-fp16 \
                        models/whisper-large-v3-ov-fp16 \
                        models/sdxl-base-1.0-ov-fp16-static
```

Ожидаемо: SDXL — все суб-модели `static`; Llama/Whisper — основной граф остаётся
`DYNAMIC` c `[has beam_idx]` (это нормально, статику обеспечит NPU-пайплайн);
`*tokenizer*.xml` — `dynamic by design`.

### Шаг 8. Упаковка архива

```bash
tar -czf openvino-2026.1-mlperf-models-fp16.tar.gz -C models .
sha256sum openvino-2026.1-mlperf-models-fp16.tar.gz | tee openvino-2026.1-mlperf-models-fp16.tar.gz.sha256
```

### Шаг 9. Рантайм на NPU (greedy, MLPerf)

```python
import openvino_genai as ov_genai

# Llama: статическая компиляция prefill/decode, greedy
llm = ov_genai.LLMPipeline("models/llama-3.1-8b-instruct-ov-fp16", "NPU",
                           MAX_PROMPT_LEN=1024, MIN_RESPONSE_LEN=256)
cfg = llm.get_generation_config(); cfg.num_beams = 1; cfg.do_sample = False
print(llm.generate("Summarize: ...", cfg))

# Whisper: greedy, аудио 16 кГц, паддинг до 30 с
asr = ov_genai.WhisperPipeline("models/whisper-large-v3-ov-fp16", "NPU")
acfg = asr.get_generation_config(); acfg.num_beams = 1
print(asr.generate(raw_audio_16k, acfg))
```

SDXL в рантайме (см. также вывод `export_sdxl.py --mlperf`): `EulerDiscreteScheduler`,
`num_inference_steps=20`, `guidance_scale=8.0`, `1024x1024`, заданный negative prompt,
latents с внешним сидом.

---

### Быстрый аналог через скрипты тулкита (необязательно)

Те же шаги одной командой на модель (флаг `--mlperf` сам пинит коммиты и ставит
greedy):

```bash
export HF_TOKEN=hf_xxx
python export_llm.py     --mlperf
python export_whisper.py --mlperf
python export_sdxl.py    --mlperf --revision <commit_вашего_раунда>
python verify_static.py  models/*-ov-*
```

> Без MLPerf (например, если хотите именно beam search на NPU): уберите `--mlperf`
> и используйте `python export_whisper.py --num-beams 5`, а в рантайме — `num_beams>1`.

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

  > Пример выше показывает beam search (`num_beams=5`) как возможность модели.
  > **Для MLPerf ставьте `num_beams=1` (greedy)** — см. пошаговую инструкцию.
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

## MLPerf Inference (Closed, Datacenter) — справка

Пошаговая инструкция выше уже выровнена под MLPerf. Ниже — что именно делает
выравнивание и reference-параметры; пресеты — в `mlperf_presets.py`.

Что делает выравнивание (флаг `--mlperf` в скриптах или ручные Шаги 3–6):

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
