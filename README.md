# NeuroDeva - TTS Branch

### This branch focuses on refactoring and improving the Text-to-Speech (TTS) component of the NeuroDeva virtual streamer system.

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="##updates">Updates</a>
    </li>
     <li>
      <a href="#quick-start">Quick Start</a>
    </li>
  </ol>
</details>


## TTS System Overview
The TTS system is responsible for converting generated text responses from LLM into natural-sounding speech output. It integrates with:

* Silero TTS Model Integration
* Real-time lip sync with Live2D model
* Audio processing and routing
* WebSocket-based VTube Studio integration

<!-- UPDATES -->
## Updates 

### 23.10.2024:
So far only separation of TTS module from the main branch, changed the structure & implementation in some parts. Improved project structure, better separation of concerns, proper async/await implementation, enhanced error handling.

## System Requirements
| Component  | Requirement |
| ------------- | ------------- |
| Python  | 3.9.7  |
| RAM  | 6GB+  |
| Audio  | Virtual Cable support  |
| OS  | Windows 10/11  |
| Memory  | 1 GB |

<!-- Quick Start -->
## Quick Start
1. Clone the repo
   ```
   git clone -b tts_tests https://github.com/3ndetz/NeuroDeva.git
   ```
2. Install the required packages:
   ```
   cd ...\NeuroDeva
   pip install -r requirements.txt
   ```
3. VTube Studio Setup:
* Install Vtube studio app through Steam
* Enable Plugin API (port 8001)
* Configure lip sync parameters

5. VTube Studio Setup:
```
python main.py
```

# Original Project
For full project context, see main branch and Habr article.
