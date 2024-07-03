### ORIGINAL REPOSITORY: [https://github.com/imartemy1524/vk_captcha](https://github.com/imartemy1524/vk_captcha) 
<h1 align="center">
<a href="https://github.com/imartemy1524/vk_captcha">vk_captcha</a> 
- AI <a href="https://vk.com/dev">VK</a> captcha solver for <b>93.2%</b> accuracy
</h1>

## Requirements
> Python3.3 - python3.10

~~Python 3.10 is not supported yet because 
[onnxruntime](https://pypi.org/project/onnxruntime/) 
is not supporting **python3.10**~~

#### UPDATE: Python3.10 is supported

## Installation


```
pip install vk_captcha
or
pip install https://github.com/imartemy1524/vk_captcha/raw/main/dist/vk_captcha-2.0.tar.gz
or
pip install git+https://github.com/imartemy1524/vk_captcha
```


### Fast examples:
Look into [VkHacker](VkHacker) for examples of accounts bruteforce

#### using [vk_api](https://github.com/python273/vk_api) library:

```python
from vk_captcha import vk_api_handler
vk = vk_api_handler.VkApiCaptcha("88005553535", "efwoewkofokw")  # this login will create captcha
vk_api_handler.Solver.logging = True  # enable logging
vk.auth() # get captcha error and automatically solve it
```

#### another way with [vk_api](https://github.com/python273/vk_api):

```python
from vk_captcha import VkCaptchaSolver
from vk_api import VkApi
solver = VkCaptchaSolver(logging=True)  # use logging=False on deploy
vk = VkApi(login='', password='', captcha_handler=solver.vk_api_captcha_handler)
vk.method("any.method.with.captcha.will.be.handled")
```

#### using [vkbottle](https://github.com/vkbottle/vkbottle):

```python
from vkbottle.bot import Bot # Or "from vkbottle.user import User"
from vk_captcha import VkCaptchaSolver

bot = Bot(token=...) # Or "bot = User(token=...)"
solver = VkCaptchaSolver()

bot.api.add_captcha_handler(solver.vkbottle_captcha_handler)
```

#### just solve captcha from *url* / *bytes*


```python
from vk_captcha import VkCaptchaSolver
import random, requests

session = requests.Session()  
session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0'

solver = VkCaptchaSolver(logging=True)  # use logging=False on deploy
sid = random.randint(122112, 10102012012012)
easy_captcha = False
url = f"https://api.vk.com/captcha.php?sid={sid}&s={int(easy_captcha)}"

answer, accuracy = solver.solve(
    url=url,
    minimum_accuracy=0.33,  # keep solving captcha while accuracy < 0.33
    repeat_count=14,  # if we solved captcha with less than minimum_accuracy, then retry repeat_count times
    session=session  # optional parameter. Useful if we want to use proxy or specific headers
)
# or
#answer, accuracy = solver.solve(bytes_data=session.get(url))
print(f"I solved captcha = {answer} with accuracy {accuracy:.4}")
```

#### async way:

```python
from vk_captcha import VkCaptchaSolver
import random, asyncio
solver = VkCaptchaSolver(logging=False)  # use logging=False on deploy
async def captcha_solver():
    sid = random.randint(122112, 10102012012012)
    easy_captcha = False
    url = f"https://api.vk.com/captcha.php?sid={sid}&s={int(easy_captcha)}"
    answer, accuracy = await solver.solve_async(url=url, minimum_accuracy=0.4, repeat_count=10)
    print(f"Solved captcha = {answer} with accuracy {accuracy:.4}")
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy()) # Only in windows
asyncio.run(captcha_solver())
```
Also, you can get some statistics of solving captcha:
```python
from vk_captcha import VkCaptchaSolver
solver = VkCaptchaSolver()
...
# solve some captchas
...
time_for1captcha = solver.argv_solve_time
total_solved = solver.TOTAL_COUNT
fail_count = solver.FAIL_COUNT  # you need directly increase it after getting second captcha error
```

In theory, for other languages you can use command line solver ( **NOT RECOMMENDED**, it will always load model again):

```
python -m vk_captcha -url "https://api.vk.com/captcha.php?sid=2323832899382092" -minimum-accuracy 0.33 -repeat-count 13
```