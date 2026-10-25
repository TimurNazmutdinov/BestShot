"""Настройки этапа 2 (analyze.py). Меняйте значения здесь — код трогать не нужно.

Технические пороги и вес оценок можно менять без повторного анализа:
тяжёлые результаты (CLIP, лица, резкость) лежат в БД, а итоговые баллы,
группы и выбор пересчитываются заново при каждом запуске.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
THUMBS_DIR = ROOT / "thumbs"
MODELS_DIR = ROOT / "models"

THUMB_SIZE = 512          # длинная сторона превью, px
FACE_WORK_SIZE = 2000     # длинная сторона изображения для поиска лиц (на групповых кадрах лица мелкие)
FACE_SCORE_MIN = 0.6      # порог уверенности детектора YuNet
FACE_MIN_PX = 32          # лица уже этого (px на FACE_WORK_SIZE) считаем, но улыбку/моргание не оцениваем
FACE_CROP_MARGIN = 0.4    # запас вокруг лица при вырезании (доля размера лица с каждой стороны)
FACE_CROP_SIZE = 384      # вырезка масштабируется до этой длинной стороны перед MediaPipe
FACE_METHOD = "yunet+mediapipe-v1"  # при смене метода лица пересчитываются без повторного анализа CLIP

# --- брак (определяется по превью 512px) ---
BLUR_MAX = 60.0           # дисперсия лапласиана в центре кадра ниже этого — смаз
DARK_LUMA_MAX = 25.0      # средняя яркость (0..255) ниже — слишком тёмный кадр
DARK_FRAC_MAX = 0.70      # доля тёмных пикселей (<16) выше — тёмный кадр
BRIGHT_FRAC_MAX = 0.60    # доля пересвета (>240) выше — пересвеченный кадр
CLIP_LOW, CLIP_HIGH = 16, 240

# --- итоговый балл: базовая оценка = LAION + TOPIQ-IAA + TOPIQ-NR + эмоции ---
W_LAION = 0.25            # эстетика LAION Aesthetic v2
W_TOPIQ_IAA = 0.25        # эстетика pyiqa topiq_iaa
W_TOPIQ_NR = 0.15         # техническое качество pyiqa topiq_nr
W_EMOTION = 0.35          # вес эмоций для кадров из групп с лицами
W_EMOTION_NO_FACES = 0.15  # вес эмоций без лиц; остальные веса пропорционально масштабируются до суммы 1
W_SHARPNESS = 0.0         # доп. вес резкости (0 — не используется)

# эмоции: основа — CLIP zero-shot; улыбка — только бонус (спокойные портреты не штрафуются)
SMILE_BONUS = 0.25
BLINK_PENALTY = 0.5       # штраф к эмоциям на долю лиц с закрытыми глазами
BLINK_THRESHOLD = 0.5     # blendshape eyeBlink выше — глаз закрыт

# технически ненормальные кадры (TOPIQ-NR ниже порога) не рекомендуются, как брак
TECH_NR_MIN = 0.40

# --- группы похожих ---
SIM_THRESHOLD = 0.92      # косинусная близость эмбеддингов CLIP
GROUP_MAX_GAP_S = 180     # максимум 3 минуты между кадрами группы

# --- отбор ---
TARGET_SELECT_FRACTION = 0.38   # цель: доля кадров события в рекомендации
# разнообразие: в группе похожих рекомендуем до N кадров, если они заметно отличаются
DIVERSITY_MAX_PER_GROUP = 3
DIVERSITY_SIM_MAX = 0.95        # косинусная близость CLIP между рекомендованными кадрами группы ниже этого
DIVERSITY_MIN_SCORE_RATIO = 0.5   # и балл не ниже этой доли от лучшего в группе

# --- личная модель (train.py) ---
PERSONAL_MODEL = MODELS_DIR / "personal.pkl"
PERSONAL_PREV = MODELS_DIR / "personal_prev.pkl"
PERSONAL_PCA = 64         # размерность PCA для эмбеддинга CLIP
PERSONAL_C = 0.05         # обратная сила регуляризации логистической регрессии (меньше = сильнее)
PERSONAL_K = 300          # вес личной модели в итоговом балле = n / (n + K), n — число меток
PERSONAL_MAX_FOLDS = 10     # проверка по событиям: если событий больше, они делятся на столько блоков (иначе — по одному)
PERSONAL_MIN_LABELS = 30  # меньше меток или меньше 3 событий — не обучаем

# --- CLIP ---
CLIP_MODEL = "ViT-L-14-quickgelu"   # веса OpenAI обучены с QuickGELU
CLIP_PRETRAINED = "openai"   # именно с этими весами обучена голова LAION Aesthetic v2
AESTHETIC_HEAD = MODELS_DIR / "sac_logos_ava1_l14_linearMSE.pth"
FACE_MODEL = MODELS_DIR / "face_landmarker.task"
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
BATCH = 8

# файлы, которые скачиваются один раз при первом запуске analyze.py (CLIP ~1,7 ГБ скачивает open_clip в models/clip)
MODEL_URLS = {
    AESTHETIC_HEAD: "https://raw.githubusercontent.com/christophschuhmann/improved-aesthetic-predictor/main/"
                    "sac%2Blogos%2Bava1-l14-linearMSE.pth",
    FACE_MODEL: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
                "face_landmarker.task",
    YUNET_MODEL: "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/"
                 "face_detection_yunet_2023mar.onnx",
}

POSITIVE_PROMPTS = ["a joyful candid moment", "people laughing", "a hug", "a tender family moment"]
NEGATIVE_PROMPTS = ["a boring snapshot", "people with closed eyes",
                    "a screenshot", "a photo of a document", "a receipt", "a blurry photo"]

# --- кадры без людей: помечаются как брак с причиной «без людей» (не рекомендуются, но отобрать можно) ---
EXCLUDE_NO_PEOPLE = True       # False — не помечать
PEOPLE_CLIP_MIN = 0.35         # «люди есть», если в кадре нашлись лица или CLIP даёт долю людей не ниже этой
PEOPLE_POSITIVE_PROMPTS = ["a photo of people", "a photo of a person", "a photo of a child", "a family photo",
                           "a group of people", "a photo of a baby", "a portrait of a person", "a person's face",
                           "people posing for a photo"]
PEOPLE_NEGATIVE_PROMPTS = ["a landscape without people", "an empty room", "a photo of an object", "a photo of nature",
                           "a photo of a building", "a photo of food", "an animal", "a photo of toys",
                           "a photo of clothes", "a photo of a vehicle or a road", "a photo of a document or paper",
                           "a photo of furniture", "a photo of flowers", "a photo of a pet", "a close-up of a thing",
                           "an interior of a room"]

# --- большие архивы: разбивка по частям ---
REVIEW_PART_SIZE = 150                 # кадров на странице («части») события во вкладке «Отбор» по умолчанию
REVIEW_PART_CHOICES = [50, 100, 150, 300, 500, 0]   # варианты размера части; 0 — всё событие сразу (тяжело для больших)
ANALYZE_PART_CHOICES = [0, 500, 1000, 2000, 5000]   # порция анализа за один запуск; 0 — все сразу

# --- папки (дерево слева в review.py) ---
FOLDER_LIST_TIMEOUT_S = 6      # таймаут чтения списка подпапок (медленные/сетевые диски)
FOLDER_WALK_TIMEOUT_S = 900    # общий лимит фонового подсчёта фото на диске; частичный результат сохраняется

# --- дубликаты (вкладка «Дубли», dupes_scan.py) ---
DUP_PH_MAX = 6            # макс. расстояние Хэмминга pHash (из 64 бит) между «той же фотографией»
DUP_DH_MAX = 10           # макс. расстояние Хэмминга dHash
DUP_CLIP_MIN = 0.90       # мин. косинусная близость CLIP (для повёрнутых копий считается по повёрнутому кадру)
DUP_PIXEL_CORR_MIN = 0.998  # мин. корреляция уменьшенных 16×16 копий: пересохранённые ≥ 0.9993, серийные кадры ≤ 0.988
DUP_PIXEL_MAD_MAX = 0.05  # макс. средняя абс. разность нормированных 16×16 копий (копии ≤ 0.025, серии ≥ 0.09)
DUP_DATE_TOLERANCE_S = 2  # допуск по дате съёмки, если дата есть у обоих файлов
DUP_UNCERTAIN_MARGIN = 1  # «автоматика не уверена»: до порога хэшей осталось не больше этого числа бит / CLIP ближе чем на 0.01

# --- живое обновление «Отбора» во время анализа ---
LIVE_SCORE_EVERY_S = 300       # раз в столько секунд баллы и события пересчитываются по уже обработанным кадрам
LIVE_SCORE_MIN_NEW = 40        # ...если с прошлого пересчёта обработано не меньше стольких новых фото

# --- о программе ---
APP_VERSION = "1.0"
