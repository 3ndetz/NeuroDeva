from obswebsocket import obsws, events, requests  # noqa: E402


class OBS_Websocket():
    host = "localhost"
    port = 4455
    password = "h4sXG6mppje0wll4"
    connected = False
    ws = None

    def check_connection(self):
        if not self.connected:
            try:
                self.ws = obsws(self.host, self.port, self.password)
                self.ws.connect()
                self.connected = True
                return True
            except BaseException as err:
                print('[OBS WS CONNECT ERR]', err)
                self.connected = False
                return False
        else:
            return True

    def call_get(self, request):
        try:
            result = self.ws.call(request)
            return dict(result.datain)
        except BaseException as err:
            print('[OBS WS REQ GET ERR]', err)
            self.connected = False
            return None

    def call(self, request):
        try:
            self.ws.call(request)
            return True
        except BaseException as err:
            print('[OBS WS REQ SIMPLE ERR]', err)
            self.connected = False
            return False

    def get_stream_status(self) -> dict:
        if self.check_connection():
            stream_status = self.call_get(requests.GetStreamStatus())
            if stream_status is not None:
                return stream_status
        return {"outputActive": False}

    # https://github.com/obsproject/obs-websocket/blob/master/docs/generated/protocol.md#getstreamstatus
    # outputActive outputReconnecting outputTimecode outputDuration outputCongestion outputBytes outputSkippedFrames outputTotalFrames

    def set_scene(self, scene_name: str) -> bool:  # NetTyanChat NetTyan NetTyan NetTyanDisclaimer
        if self.check_connection():
            return self.call(requests.SetCurrentProgramScene(sceneName=scene_name))

    def set_record(self, enable: bool) -> bool:
        if self.check_connection():
            if enable:
                return self.call(requests.StartRecord())
            else:
                return self.call(requests.StopRecord())

    def set_stream(self, enable: bool) -> bool:
        if self.check_connection():
            if enable:
                return self.call(requests.StartStream())
            else:
                return self.call(requests.StopStream())


# timecode = stream_status.get("outputTimecode", "00:00:-00")
# active = stream_status.get("outputActive", False)
if __name__ == "__main__":
    import sys
    import time

    import logging

    # logging.basicConfig(level=logging.DEBUG)

    # sys.path.append('../')

    host = "localhost"
    port = 4455
    password = "h4sXG6mppje0wll4"


    def on_event(message):
        print("Got message: {}".format(message))


    def on_switch(message):
        print("You changed the scene to {}".format(message.getSceneName()))


    # {'d': {'eventData': {'sceneName': 'NetTyanDisclaimer'}, 'eventIntent': 4, 'eventType': 'CurrentProgramSceneChanged'}, 'op': 5}
    # Got message: <CurrentProgramSceneChanged event ({'sceneName': 'NetTyanDisclaimer'})>
    def connec():
        print('connec thread started')
        new_obs_exemplar = OBS_Websocket()
        print(new_obs_exemplar.get_stream_status())
        time.sleep(10)
        # new_obs_exemplar.set_record(True)
        time.sleep(5)
        # new_obs_exemplar.set_record(False)
        time.sleep(10)
        print(new_obs_exemplar.get_stream_status())
        time.sleep(10)
        print(new_obs_exemplar.get_stream_status())
        time.sleep(10)
        print(new_obs_exemplar.get_stream_status())
        time.sleep(10)
        print(new_obs_exemplar.get_stream_status())
        time.sleep(10)
        print(new_obs_exemplar.get_stream_status())


    #
    import threading

    thr = threading.Thread(target=connec, daemon=True)
    thr.start()
    # ws = obsws(host, port, password)
    #
    # ws.register(on_event)
    # ws.register(on_switch, events.SwitchScenes)
    # ws.register(on_switch, events.CurrentProgramSceneChanged)
    # ws.connect()
    # Got message: <RecordStateChanged event ({'outputActive': True, 'outputPath': 'C:/Users/Onix/Downloads/2023-07-09 01-15-20.mp4', 'outputState': 'OBS_WEBSOCKET_OUTPUT_STARTED'})>
    # Got message: <RecordStateChanged event ({'outputActive': False, 'outputPath': 'C:/Users/Onix/Downloads/2023-07-09 01-15-20.mp4', 'outputState': 'OBS_WEBSOCKET_OUTPUT_STOPPED'})>

    # Хоррор версия NetTyan

    # Got message: <SourceFilterEnableStateChanged event ({'filterEnabled': False, 'filterName': 'HorrorNetTyan', 'sourceName': 'VTube Studio'})>
    # DEBUG:obswebsocket.core:Got event: {'d': {'eventData': {'filterEnabled': False, 'filterName': 'HorrorNetTyan', 'sourceName': 'VTube Studio'}, 'eventIntent': 32, 'eventType': 'SourceFilterEnableStateChanged'}, 'op': 5}
    # DEBUG:obswebsocket.core:Got event: {'d': {'eventData': {'filterEnabled': True, 'filterName': 'HorrorNetTyan', 'sourceName': 'VTube Studio'}, 'eventIntent': 32, 'eventType': 'SourceFilterEnableStateChanged'}, 'op': 5}
    # Got message: <SourceFilterEnableStateChanged event ({'filterEnabled': True, 'filterName': 'HorrorNetTyan', 'sourceName': 'VTube Studio'})>

    # Нужное!

    ####def set_scare_filter(value: bool = True):
    ####    ws.call(requests.SetSourceFilterEnabled(sourceName="VTube Studio",filterName="HorrorNetTyan", filterEnabled=value))
    ####
    ####def scare_transition():
    ####    currentSceneData = ws.call(requests.GetCurrentProgramScene())
    ####    print("currentSceneData", currentSceneData, currentSceneData.data(),currentSceneData.datain,currentSceneData.dataout)
    ####    currentScene = currentSceneData.datain.get("currentProgramSceneName", "NetTyan")
    ####    print('got cur scene: '+currentScene+" "+str(currentSceneData))
    ####    ws.call(requests.SetSceneSceneTransitionOverride(sceneName=currentScene, transitionName=None,
    ####                                                     transitionDuration=None))
    ####    #ws.call(requests.SetSceneSceneTransitionOverride(sceneName=currentScene, transitionName="NetTyanTransition", transitionDuration=1250))
    ####def get_stream_time():
    ####    stream_status = ws.call(requests.GetStreamStatus())
    ####    timecode = stream_status.datain.get("outputTimecode", "00:00:-00")
    ####    active = stream_status.datain.get("outputActive", False)
    ####    print(timecode,active)

    # https://github.com/obsproject/obs-websocket/blob/master/docs/generated/protocol.md#stopstream
    obss = OBS_Websocket()
    try:
        print("OK")
        time.sleep(1)
        print('1')
        # ws.call(requests.SetCurrentProgramScene(sceneName='NetTyan'))
        # ws.call(requests.StartRecord())
        # ws.call(requests.StartStream())
        # ws.call(requests.StopStream())
        # time.sleep(2)
        # scare_transition()
        time.sleep(0.4)
        print(obss.get_stream_status())
        # set_scare_filter(True)
        # ws.call(requests.SetCurrentProgramScene(sceneName='NetTyan'))
        time.sleep(1)
        # set_scare_filter(False)
        # ws.call(requests.SetCurrentProgramScene(sceneName='NetTyanDisclaimer'))
        # ws.call(requests.StopRecord())
        # https://github.com/obsproject/obs-websocket/blob/master/docs/generated/protocol.md#setpersistentdata
        # https://pub.dev/documentation/obs_websocket/latest/#sending-commands-to-obs---low-level
        # ws.call(requests.GetPersistentData(realm='OBS_WEBSOCKET_DATA_REALM_PROFILE',slotName='port'))
        time.sleep(100)
        print("END")

    except KeyboardInterrupt:
        pass

# import obspython as obs
#
## The name of# the scene containing the text source
# scene_name = "NetTyan"
#
## The name of the text source to modify
# text_source_name = "TextForChange"
#
## The new text value to set
# new_text_value = "Hello, world!"
#
# def script_description():
#    return "Changes the text value of a text source in an OBS sc3ene"
#
# def script_update(settings):
#    # Get the current scene and text source objects
#    current_scene = obs.obs_frontend_get_current_scene()
#    scene = obs.obs_scene_from_source(current_scene)
#    text_source = obs.obs_scene_find_source(scene, text_source_name)
#
#    ###if scene is not None and text_source is not None:
#    ###    # Set the new text value
#    ###    settings_value = obs.obs_data_get_string(settings, "text_value")
#    ###    obs.obs_data_set_string(obs.obs_data_create(), "text", settings_value)
#    ###    obs.obs_source_update(text_source, obs.obs_data_create())
#    ###
#    ###    # Release the objects
#    ###    obs.obs_source_release(text_source)
#    ###    obs.obs_scene_release(scene)
#    ###    obs.obs_sceneitem_release(current_scene)
#
# def script_properties():
#    props = obs.obs_properties_create()
#
#    # Add a text property for the new text value
#    obs.obs_properties_add_text(props, "text_value", "New Text Value", obs.OBS_TEXT_DEFAULT)
#
#    return props
