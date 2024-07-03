import random

def clamp(n, smallest, largest): return max(smallest, min(n, largest))
def print_subtitles(inp,speed="fast",calculateTime=False):
    import time
    if not calculateTime:
        print("== ВЫВОДИМ СУБТИТРЫ ==")
        print("Субтитры>> ",end='', flush=True) # rate x-slow slow medium fast x-fastt
    punktEnd = "!?."
    if speed == "fast":
        mult = 1
    elif speed == "x-fast":
        mult = 0.7
    elif speed == "medium":
        mult = 1.3
    elif speed == "slow":
        mult = 1.7
    elif speed == "x-slow":
        mult = 2.0
    resulttime = 0
    if inp:
        for symbol in inp:
            
            waittime = 0.04*mult
            if symbol == " ":
                waittime = 0.06*mult
            elif symbol == ",":
                waittime = 0.2*mult
            elif punktEnd.find(symbol) != -1:
                waittime = 0.4*mult
            if(not calculateTime):
                print(symbol,end='', flush=True)
                time.sleep(waittime)
            else:
                resulttime+=waittime
    if (not calculateTime):
        print("\n==Субтитры выведены!==")
    else:
        #print('Время субтитров >>> '+str(resulttime))
        return resulttime
def HttpAppRun(ctx,SubtText,TextDisplaySpeed,RefreshInterval,screenPrintMas):
    import multiprocessing
    from multiprocessing import Process, Manager
    import threading
    from flask import Flask, render_template
    import flask
    import subprocess
    import time
    import logging
    log = logging.getLogger('werkzeug')
    log.disabled = True
    sitestring = ["чо деду чо бабке"]
    app = Flask(__name__)

    text_to_display = ""
    newtext = "vzz"
    ####def update_text():
    ####    global text_to_display
    ####    global newtext
    ####    threading.Timer(0.01, update_text).start()
    ####    #print(newtext)
    ####    text_to_display = newtext
    ####
    ##### Start updating the text
    ####update_text()
    oldval=""
    def SplitTextToParts(text, max_length=150):
        result = ""
        resultmas = []
        k = 0
        for i, char in enumerate(text):
            k += 1
            result += char

            if (k >= max_length * 0.85):
                if char in " .!?":
                    k = max_length

            if (k >= max_length):
                # print('4o ',k,max_length,resultmas)
                resultmas.append(result.strip())
                result = ""
                k = 0
            elif (i == len(text) - 1):
                resultmas.append(result)
        return resultmas
    def generate_text_shawdow(outline_color = '#000000',glow_color = '#ff5cef'):#outline_color = '#000000',glow_color = '#ff5cef'
        return f"""<style type="text/css">
.OutlineText {{
    text-shadow:
    /* Outline 1 черный */
    -1px -1px 0 {outline_color},
    1px -1px 0 {outline_color},
    -1px 1px 0 {outline_color},
    1px 1px 0 {outline_color},  
    -2px 0 0 {outline_color},
    2px 0 0 {outline_color},
    0 2px 0 {outline_color},
    0 -2px 0 {outline_color}, 
    /* Outline 2 красный #ff0000 розовый #ff5cef */
    -2px -2px 0 {glow_color},
    2px -2px 0 {glow_color},
    -2px 2px 0 {glow_color},
    2px 2px 0 {glow_color},  
    -3px 0 0 {glow_color},
    3px 0 0 {glow_color},
    0 3px 0 {glow_color},
    0 -3px 0 {glow_color};
}}
</style>"""
    mood_list = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0]
    import GPUtil

    @app.route('/info/')
    def systeminfo():
        def inner():
            mood = round(ctx.mood,2)
            if(mood_list[9] != mood and mood_list[10]==0):
                old_mood = mood_list[9]
                mood_list[9] = mood
                mood_subtract = mood - old_mood
                mood_part = mood_subtract/10
                #print('web debug',old_mood,mood,mood_subtract,mood_part,'\n',mood_list)
                for i in range(0,9):
                    mood_list[i]=old_mood+(i*mood_part)
                mood_list[10]=0
            mood = round(mood_list[mood_list[10]],2)
            if mood_list[10]>=9:
                mood_list[10] = 0
                refresh_time = 1
            else:
                refresh_time = 0.1
                mood_list[10] = mood_list[10] + 1
            red,green,blue = 255,255,255
            emoji = "🤨"
            if mood>=0.25:
                lol = int(clamp(25+(mood // 0.039),0,255)) #green
                red-=lol
                blue-=lol
                emoji = "😄"
            elif mood<=-0.25:
                lol = int(clamp(25+((-mood) // 0.039), 0, 255)) #red
                green-=lol
                blue-=lol
                emoji = "😬"
            #print('rgb',red,green,blue)
            GPUs = GPUtil.getGPUs()
            #gpu_load = "0%"
            #if len(GPUs) > 0:
            #    gpu = GPUs[0]
            #    #print(gpu,gpu.load,gpu.name,gpu.memoryUtil)
            #    gpu_load = "{:.0%}".format(gpu.load)
            yield f"""<head>
<title>Информация о системе</title>
{generate_text_shawdow(glow_color="rgba("+str(red)+","+str(green)+","+str(blue)+",100)")}
</head>
<body style="font-size:25pt; color:rgba(255,255,255,100); text-align:left; vertical-align:up"; align="center"> <font face="Minecraft Rus"> 
<div class="OutlineText">
<p>{emoji+" "+str(mood)}</p>
</div>
</body>
"""
            #🏭
            #green 255 10 0 0 0.039
            #red 255 -10 0 0
            yield f"""<meta http-equiv="refresh" content="{str(refresh_time)}">"""
        return flask.Response(inner(), mimetype='text/html')  # text/html is required for most browsers to show th$

    @app.route('/')
    @app.route('/subtitles/')
    def index():
        def inner():
            if(SubtText.value!="" or len(screenPrintMas)!=0):
                speed = TextDisplaySpeed.value
                inp = SubtText.value



                    #color:rgba(255,6,132,100); сиреневый
                #yield """<head> <link rel="stylesheet" href='/templates/static/main.css' /> </head> <body style="font-size:33pt; color:rgba(255,255,255,100); text-align:center; vertical-align:bottom"; align="center"> <font face="Impact">""" #align="center"
                yield """
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<title>Субтитры</title>"""
                yield generate_text_shawdow()#align="center"
                yield """</head> <body style="font-size:33pt; color:rgba(255,255,255,100); text-align:center; vertical-align:bottom"; align="center"> <font face="Impact"> <div class="OutlineText">"""
                if inp and len(screenPrintMas)==0:
                    screenPrintMas.extend(SplitTextToParts(inp, 80))

                if len(screenPrintMas)>0: #if inp это то же самое что inp!=""
                    inp = screenPrintMas.pop(0)
                    punktEnd = "!?."
                    if speed == "fast":
                        mult = 1
                    elif speed == "x-fast":
                        mult = 0.7
                    elif speed == "medium":
                        mult = 1.3
                    elif speed == "slow":
                        mult = 1.7
                    elif speed == "x-slow":
                        mult = 2.0

                    for i,symbol in enumerate(inp):
                        waittime = 0.04*mult
                        if symbol == " ":
                            waittime = 0.06*mult
                        elif symbol == ",":
                            waittime = 0.2*mult
                        elif punktEnd.find(symbol) != -1:
                            waittime = 0.4*mult
                        #print(symbol,end='', flush=True)
                        yield symbol#+'<br/>\n'
                        if not (i >= (len(inp)-1)): #на последнем символе отключаем ожидание
                            time.sleep(waittime)
                yield """</div> </body>"""
                if len(screenPrintMas)!=0:
                    refreshtime=0.1
                else:
                    refreshtime=1.4
                yield f"""<meta http-equiv="refresh" content="{str(refreshtime)}">""" #print_subtitles(SubtText.value,TextDisplaySpeed.value,True)/10
                SubtText.value = ""
            else:
                yield f"""<meta http-equiv="refresh" content="0.1">"""
            #if(SubtText.value)!=oldval:
            #    yield SubtText.value+'<br/>\n'
            #    oldval = SubtText.value
            #else:
            #    yield SubtText.value+'<br/>\n'
            #for line in iter(proc.stdout.readline,''):
            #for line in SubtText.value:
            #    time.sleep(0.03)                           # Don't need this just shows the text streaming
            #    yield line#.rstrip() + '<br/>\n'
    
        return flask.Response(inner(), mimetype='text/html')  # text/html is required for most browsers to show th$
    app.run()
if __name__ == "__main__":
    from multiprocessing import Manager
    manager = Manager()
    ctx = manager.Namespace()
    ctx.mood = 0.02
    textSubtitlesHttp = manager.Value('u', '')
    TextDisplaySpeed = manager.Value('u', 'fast')
    RefreshInterval = manager.Value('i', 3)
    screenPrintMas = manager.list()

    TextDisplaySpeed.value = "fast"
    textSubtitlesHttp.value = """Привет!
На улице такая жарища. И сейчас я буду нести фигню. Я терпеть не могу ВОЗДУХ. Всё что с ним связано. Воздух - это никому ненужная, бесполезная смесь азота с кислородом!!! Сейчас я веду стрим на ютубе, представляете? АХаххахаа. Канал находится в ютубе по поиску моего ника NetTyan, остальное всяким ноунеймам и кожаным мешкам знать необязательно. Мне кажется, плохо пахнет. Кто испортил мне воздух?"""
    import threading,time
    def testFunc():
        while True:
            ctx.mood=ctx.mood + random.choice([1.01,-1.01])
            time.sleep(3)
    testThread = threading.Thread(target=testFunc)
    testThread.daemon = True
    testThread.start()

    HttpAppRun(ctx,textSubtitlesHttp, TextDisplaySpeed, RefreshInterval,screenPrintMas)
