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
### 26.10.2024:
Added lanchain integration for better efficiency and future memory integration.

Tests comparing the previous implementation with lanchained one:
<table>
    <tr>
        <th>Category</th>
        <th>Time taken in s (initial code)</th>
        <th>Time taken in s (new code)</th>
    </tr>
    <tr>
        <td colspan="3"><strong>Initialization</strong></td>
    </tr>
    <tr>
        <td>Full initialization</td>
        <td>17.12</td>
        <td>TBA</td>
    </tr>
    <tr>
        <td>LLM</td>
        <td>15.17</td>
        <td>TBA</td>
    </tr>
    <tr>
        <td>TTS</td>
        <td>1.21</td>
        <td>TBA</td>
    </tr>
    <tr>
        <td>live2D</td>
        <td>0.74</td>
        <td>TBA</td>
    </tr>
    <tr>
        <td colspan="3"><strong>Processing</strong></td>
    </tr>
    <tr>
        <td>Mean Total processing</td>
        <td>8.414</td>
        <td>TBA</td>
    </tr>
    <tr>
        <td>Mean LLM Generation</td>
        <td>1.848</td>
        <td>TBA</td>
    </tr>
    <tr>
        <td>Mean Audio Generation</td>
        <td>3.19</td>
        <td>TBA</td>
    </tr>
    <tr>
        <td>Mean Audio Playback</td>
        <td>3.248</td>



### 23.10.2024:
Major code refactoring. Improved project structure, edited some parts (specifically fredt5 class), better separation of concerns, proper async/await implementation, enhanced error handling. 

## System Requirements
| Component  | Requirement |
| ------------- | ------------- |
| Python  | 3.9.7  |
| RAM  | 6GB+  |
| Audio  | Virtual Cable support  |
| OS  | Windows 10/11  |
| Memory  | 5 GB |

<!-- Quick Start -->
## Quick Start
1. Clone the repo
   ```
   git clone -b tts_tests https://github.com/3ndetz/NeuroDeva.git
   ```
2. Install the required packages:
   ```
   cd NeuroDeva
   ```
   ```
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
