# DOCKER CONTAINER: MAIN SIDE (runs on NOT docker)

import os
# pip install docker
import subprocess, shlex
from multiprocessing.managers import SyncManager
import time, datetime

class MyManager(SyncManager):
    pass

MyManager.register("syncdict")

MyManager.register("tts_input_q")
MyManager.register("tts_output_q")

MyManager.register("llm_input_q")
MyManager.register("llm_output_q")

MyManager.register("llm_loading_flag")

def get_or_run_docker_container():
    import docker
    client = docker.from_env()
    container_name = "nemo_stt"
    container_image = "nvcr.io/nvidia/nemo:23.04"
    container = None
    for l in client.containers.list(all=True):
        if l.name == container_name:
            container = l
    if container is not None:
        if container.status == "running":
            print('[DOCKER] container', container_name, "running!")
        else:  # if container.status == "exited":
            container.start()
            print('[DOCKER] container starting..')
            time.sleep(5)
    else:
        thisfolder = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')
        docker_image_work_dir = "/workspace/nemo/"
        print('[DOCKER] container NON EXIST! Create and start.....')
        # СЕЙЧАС ОТКЛЮЧЕНЫ GPU! ЧТОБЫ ВКЛЮЧИТЬ ДОБАВИТЬ ТЕГ  --gpus all
        mountControlDir = f"-v {thisfolder}/HyperAI_Docker/docker_reciever.py:{docker_image_work_dir}docker_reciever.py" +\
                          f" -v {thisfolder}/HyperAI_Docker/other:{docker_image_work_dir}other"

        mountSTTDir = f"-v {thisfolder}/HyperAI_Models/STT/docker_to_send:{docker_image_work_dir}STT/"
        mountTTSDir = f"-v {thisfolder}/HyperAI_Models/TTS:{docker_image_work_dir}TTS/"
        mountLLMDir = f"-v {thisfolder}/HyperAI_Models/LLM:{docker_image_work_dir}LLM/"
        mountFiltersDir = f"-v {thisfolder}/HyperAI_Models/Filters:{docker_image_work_dir}Filters/"
        create_command = f"docker run {mountSTTDir} {mountLLMDir} {mountFiltersDir} {mountTTSDir} {mountControlDir} --gpus all --shm-size=8g -p 8888:8888 -p 6006:6006 -p 6523:6523 -i --ulimit memlock=-1 --ulimit stack=67108864 -d=true --name {container_name} {container_image} /bin/sh"
        # должно быть -it, но из-за кастрации..
        # https://stackoverflow.com/questions/43099116/error-the-input-device-is-not-a-tty
        print('[DOCKER SENDER] EXECUTION',create_command)
        subprocess.run(shlex.split(create_command), shell=True)
        # print('Create command output =',subprocess.getoutput(create_command))
        # p = subprocess.Popen(shlex.split(create_command), shell=True)
        # os.system()
        # time.sleep(5)
        # print('trying echo')
        # p.communicate("echo 1")
        time.sleep(2.5)
        print('Container CREATED! (probably) wait 3 sec')
        container = client.containers.get(container_name)
        time.sleep(3)
        # client.containers.run(container_name,detach=True,
        #                      ports={'8888/tcp': 8888,
        #                             '6006/tcp': 6006,
        #                             '6523/tcp': 6523,},
        #                      volumes=[f'{thisfolder}/docker_to_send:/workspace/nemo/']
        # )
        # os.system("docker ")
    return container, container_name





def check_reciever_process_started(container_name):
    linux_cmd = """sh /other/reciever_proc_check.sh docker_reciever.py"""
    windows_cmd = f"docker exec {container_name}"
    # p = subprocess.Popen([windows_cmd+" "+linux_cmd], stderr=subprocess.PIPE)
    # result = p.stdout.read()
    result = subprocess.getoutput(windows_cmd + " " + linux_cmd)
    print('[DOCKER DEBUG RECIEVER CHECK] RESULT SUBPROCESS =', result, '!')
    if result.find("Running") != -1:
        return True
    elif result.find("Stopped") != -1:
        return False
    else:
        print('RESULT SUBPROCESS =', result, '!')
        return False
    # https://stackoverflow.com/questions/18739239/python-how-to-get-stdout-after-running-os-system


#print('Starting docker sender...')


# client.containers.get(container_name)
def check_docker_app():
    # docker run -v F:/Onix/Downloads/minebot/1HyperAI/HyperAI_Models/STT/docker_to_send:/workspace/nemo/ --shm-size=8g -p 8888:8888 -p 6006:6006 -p 6523:6523 --gpus all -it --ulimit memlock=-1 --ulimit stack=67108864 --name nemo_stt nvcr.io/nvidia/nemo:23.04 /bin/sh
    container, container_name = get_or_run_docker_container()
    if not check_reciever_process_started(container_name):
        time.sleep(0.5)
        container.exec_run("python docker_reciever.py", detach=True)
        print('STARTED DOCKER RECIEVER! Waiting 15 secs for it initialize')
        time.sleep(15)

def file_to_bytes_io(filename):
    import io
    fileOpen = open(filename, 'rb+')
    filee = fileOpen.read()
    samples_file = io.BytesIO(filee)
    fileOpen.close()
    return samples_file

def kill_docker_reciever():  # если изменен конечный файл нада перезапуск
    import docker
    client = docker.from_env()
    container = client.containers.get("nemo_stt")
    time.sleep(0.2)
    print("закрываем процесс docker_reciever.py")
    container.exec_run("pkill -f docker_reciever.py", privileged=True, detach=True, stream=True)
    time.sleep(0.2)
    exit()
class DockerSender():
    def __init__(self):
        self.manager = None
        self.initialized = False
        self.check_connection()
    def check_connection(self, force=False):
        if not self.initialized or force:
            try:
                print('[DOCKER SENDER INIT] Starting SENDER manager...')
                from HyperAI_Secrets import DockerAuthKey
                self.manager = MyManager(('localhost', 6006), authkey=DockerAuthKey)
                self.manager.connect()
                self.initialized = True
                print('[DOCKER SENDER INIT] SUCCESSFULL CONNECTED!')
            except BaseException as err:
                print('[DOCKER STT SENDER] Ошибка при подключении:',err,'запуск чекера')
                check_docker_app()
    def stop_docker_reciever(self):
        if self.initialized:
            self.manager.syncdict()["stop"] = True
    def llm_loading_flag(self):
        if self.initialized:
            return self.manager.llm_loading_flag()
        else:
            self.check_connection()
            return self.manager.llm_loading_flag()
    def chatbot(self, llm_input, params, danger_context):#ninp,context,paramsOverride,environment,lmUsername):
        self.check_connection()
        try:
            #keywords = {"context": context, "paramsOverride": paramsOverride, "environment": environment,
            #            "lmUsername": lmUsername}
            #self.manager.llm_input_q().put((ninp,keywords))
            self.manager.llm_input_q().put((llm_input, params, danger_context))
            out = self.manager.llm_output_q().get()
            return out
        except BaseException as err:
            print('[DOCKER TTS SEND] Ошибка',err,' ПЕРЕПОДКЛЮЧЕНИЕ!')
            self.check_connection(force=True)
    def transcribe(self, audio_bytes_io, audio_settings=None):
        self.check_connection()
        try:
            if audio_settings is None:
                audio_settings = {"sr": 48000, "channels": 2}
            self.manager.tts_input_q().put({"bytes_io":audio_bytes_io,"audio_settings":audio_settings})
            out = self.manager.tts_output_q().get()
            return out
        except BaseException as err:
            print('[DOCKER TTS SEND] Ошибка',err,' ПЕРЕПОДКЛЮЧЕНИЕ!')
            self.check_connection(force=True)


def TranscribeTest(docker_sender):
    inp = file_to_bytes_io("baya synth.wav")
    audio_settings = {"sr":48000,"channels":1}
    print('sended input')  # ,inp)
    result = docker_sender.transcribe(audio_bytes_io=inp,audio_settings=audio_settings)
    print('got output =',result)


def LLMTest(docker_sender):
    print('sended input LLM')  # #ninp,context,paramsOverride,environment,lmUsername
    result = docker_sender.chatbot("Как дела зшщз", "", None, {"env": "yt", "diags_count":0}, "ChoLexe")
    print('got LLM output =',result)


if __name__ == "__main__":
    # ТОЛЬКО ДЛЯ ДЕБАЖИНГА!!!
    # FOR CREATE CONTAINER
    def debugFunc():
        print('DEBUG FUNC! DEACTIVATED FUNCTIONALITY')

        # AFTER THAT NEED pip install accelerate IN DOCKER CONTAINER CONSOLE
        pass #do stuff
        get_or_run_docker_container()
        print('EXITING..')
        exit()
    debugFunc()
    # TEST LLM ON OTHER SIDE
    #kill_docker_reciever()




    docker_sender = DockerSender()
    from multiprocessing import Process

    #Process(target=lambda a: print("Hello, {}".format(a)), args=(["world"])).start()

    Process(target=LLMTest, args=(docker_sender,)).start()
    Process(target=TranscribeTest, args=(docker_sender,)).start()
    Process(target=LLMTest, args=(docker_sender,)).start()
    Process(target=TranscribeTest, args=(docker_sender,)).start()
    Process(target=TranscribeTest, args=(docker_sender,)).start()
    Process(target=LLMTest, args=(docker_sender,)).start()
    Process(target=LLMTest, args=(docker_sender,)).start()

    print('ALL PROCESSES STARTED')
    time.sleep(15)
    print('TERMINATING!')
    exit()
    #proc1.join()

    #a = input("poka")




#docker run --shm-size=8g --device=/dev/snd --gpus all -it --rm -v nemo nvcr.io/nvidia/nemo:23.04 -p 8888:8888 -p 6006:6006 -p 6523:6523

#docker run -v <source>:<destination> --shm-size=8g -p 8888:8888 -p 6006:6006 -p 6523:6523 --gpus all -it --ulimit memlock=-1 --ulimit stack=67108864 --name nemo_stt nvcr.io/nvidia/nemo:23.04 /bin/sh

#LASTEST

#docker run -v F:/Onix/Downloads/minebot/1HyperAI/HyperAI_Models/STT/docker_to_send:/workspace/nemo/ --shm-size=8g -p 8888:8888 -p 6006:6006 -p 6523:6523 --gpus all -it --ulimit memlock=-1 --ulimit stack=67108864 --name nemo_stt nvcr.io/nvidia/nemo:23.04 /bin/sh

#[I 03:40:45.633 LabApp] 302 GET /lab? (172.17.0.1) 1.120000ms