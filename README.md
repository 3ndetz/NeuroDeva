[<img src="https://img.shields.io/badge/Habr-%D0%A7%D0%B8%D1%82%D0%B0%D1%82%D1%8C-%23000000?style=for-the-badge&link=https://habr.com/ru/articles/812387&logo=habr&logoColor=%23FFFFFF&labelColor=%2365A3BE"/>](https://habr.com/ru/articles/812387/)

# NeuroDeva

Streams block game. Plays block game. Talking with people.

![Глаза2.gif](.github/Глаза2.gif)

[<img src="https://img.shields.io/youtube/channel/views/UCy6HXAVZo3X9W3q9SrCPInQ?style=flat&label=youtube-views&link=https%3A%2F%2Fwww.youtube.com%2F%40NetTyan"/>](https://www.youtube.com/@NetTyan)
[<img src="https://img.shields.io/github/stars/3ndetz/AutoClef?style=flat&label=game-bot-repo&link=https%3A%2F%2Fgithub.com%2F3ndetz%2FAutoClef"/>](https://github.com/3ndetz/autoclef)
[<img src="https://img.shields.io/twitch/status/neurodeva?style=flat&link=https%3A%2F%2Fwww.twitch.tv%2Fneurodeva"/>](https://www.twitch.tv/neurodeva)

Automatic virtual streaming system. Completely autonomus.

Name aliases: NeuroDeva, NetTyan

## Кратко о главном

Полностью автоматическая виртуальная стримерша. Играет в Minecraft в мини-режим SkyWars. Общается с игроками и зрителями в реальном времени.

Подробнее с проектом можно ознакомиться на [хабре](https://habr.com/ru/articles/812387/), здесь только общее описание технической части.

[<img src=".github/portfolio-details-2.jpg" height="250"/>](https://habr.com/ru/articles/812387/)


## ФИЧИ

```
Это что-то вроде списка технологий, фич и содержания одновременно =)
```

<details><summary>О пометках</summary>

Формат: Название - ([ссылки](#ФИЧИ)) - _связанные файлы в репозитории, путь_

Ссылки:
- ([хабр](https://habr.com/ru/articles/812387/)): соответствующий раздел статьи на Хабре для подробностей по разработке (для удобства в ней также есть своя [навигация](https://habr.com/ru/articles/812387/#Portal)). 
- (репо): ссылка на другой репозиторий в GitHub
</details>

---

- Бот для автоматической игры - ([хабр](https://habr.com/ru/articles/812387/#CodeDisclaimer)) - _([репо](https://github.com/3ndetz/autoclef))_ + _HyperAI_BRIDGE.py_
   - Java-часть: ([репо](https://github.com/3ndetz/autoclef)) ([хабр](https://habr.com/ru/articles/812387/#CodeDisclaimer))
   - HyperAI_Bridge: сетевой мост между главным Python-скриптом и Java-ботом
   - Собственный ИИ для распознавания MC-капчи ([репо](https://github.com/3ndetz/MapResolverMC)) ([хабр](https://habr.com/ru/articles/812387/#MCMapCaptchaSolver)) ([потыкать на HF space](https://huggingface.co/spaces/3ndetz/mc_map_resolver))
   - <details><summary>Гифка: сбор ресурсов с побеждённого игрока</summary>
         <img src="https://habrastorage.org/getpro/habr/upload_files/f83/1ff/61c/f831ff61cf2cdd8d5b68b10e8dd9a8a5.gif" height="250"/>
      </details>
- Виртуальный аватар: VTube Studio - _HyperAI_VTube_
   - Live 2D: LiveroiD (автор разрешает использование для проведения трансляций)
   - Связь аватара с игрой и речью: эмоции, взгляд - ([хабр](https://habr.com/ru/articles/812387/#MineEyeBridge)) - _HyperAI.py + HyperAI_BRIDGE.py_
   - <details><summary>Гифка: аватар поворачивает взгляд в сторону цели игрового бота</summary>
         <img src=".github/Глаза2.gif" height="250"/>
      </details>
- Синтез речи - ([хабр](https://habr.com/ru/articles/812387/#AnotherCode)) - _HyperAI.py/TTS_PROCESS_
   - Silero TTS ([silero-models](https://github.com/snakers4/silero-models))
   - Realtime субтитры в OBS через веб-приложение на Flask ([хабр](https://habr.com/ru/articles/812387/#AnotherCode))
   - "Проброс" звука в OBS через virtual audio cable ([хабр](https://habr.com/ru/articles/812387/#AnotherCode))
- Наложение динамических элементов в OBS 
   - Элементы передаются и обновляются в OBS через веб-приложение на Flask ([хабр](https://habr.com/ru/articles/812387/#AnotherCode))
   - Элементы: индикатор настроения и псевдосинхронизированные субтитры ([хабр](https://habr.com/ru/articles/812387/#AnotherCode))
- Диалоговая система - ([хабр](https://habr.com/ru/articles/812387/#ChatSystem)) - _HyperAI_Models/LLM_
   - RAG-like подход ([хабр со всеми подробными схемами](#github-pages-installation))
   - <details><summary>Пикча схемы сборщика промпта (абстрактно)</summary>
         <img src="https://habrastorage.org/getpro/habr/upload_files/18b/a93/94f/18ba9394f9b6cfc7b67c9bd74f44ec93.jpg" height="500"/>
      </details>
   - <details><summary>Хранение постоянных данных с помощью sqlite</summary>
         <img src="https://habrastorage.org/getpro/habr/upload_files/304/fe2/401/304fe240195c033080477044fbe1d310.png" height="400"/>
      </details>  
     
     _HyperAI.py + HyperAI_Database.py_
   - LLM - FredT5 (на Docker) ([Почему он? Ответ на хабр!](#github-pages-installation))
   - Фильтры ИИ: токсичность, запретные темы ([хабр](#local-installation)) - _HyperAI_Models/Filters_
   - Обычный list-like фильтр на самые "опасные" слова
- Распознавание речи - ([хабр](https://habr.com/ru/articles/812387/#AnotherCode)) - _HyperAI_Models/STT/docker_to_send_
   - Docker-based - ([хабр](#configuration))
   - Модель: Nvidia fastconformer hybrid large ([HF](https://huggingface.co/nvidia/stt_ru_fastconformer_hybrid_large_pc))
- Подключение к социальным сетям - чтение чата и публикация ответов ([хабр](https://habr.com/ru/articles/812387/#AnotherCode))
   - youtube-data-api read & write
   - twitch-api read & write
   - trovo-api read & write
   - discord-bot-api (pycord) read & write



