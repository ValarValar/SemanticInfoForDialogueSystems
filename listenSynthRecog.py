# машинное обучения для реализации возможности угадывания намерений
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from vosk import Model, KaldiRecognizer  # оффлайн-распознавание от Vosk
import speech_recognition  # распознавание пользовательской речи (Speech-To-Text)
import pyttsx3  # синтез речи (Text-To-Speech)
import wave  # создание и чтение аудиофайлов формата wav
import json  # работа с json-файлами и json-строками
import os  # работа с файловой системой
import random
from termcolor import colored
from dotenv import load_dotenv
import googletrans

class VoiceAssistant:
    """
    Настройки голосового ассистента, включающие имя, пол, язык речи
    """
    name = ""
    sex = ""
    speech_language = ""
    recognition_language = ""


class OwnerPerson:
    """
    Информация о владельце, включающие имя, город проживания, родной язык речи, изучаемый язык (для переводов текста)
    """
    name = ""
    home_city = ""
    native_language = ""
    target_language = ""

def setup_assistant_voice():
    """
    Установка голоса по умолчанию (индекс может меняться в
    зависимости от настроек операционной системы)
    """
    voices = ttsEngine.getProperty("voices")

    if assistant.speech_language == "en":
        assistant.recognition_language = "en-US"
        if assistant.sex == "female":
            # Microsoft Zira Desktop - English (United States)
            ttsEngine.setProperty("voice", voices[1].id)
        else:
            # Microsoft David Desktop - English (United States)
            ttsEngine.setProperty("voice", voices[2].id)
    else:
        assistant.recognition_language = "ru-RU"
        # Microsoft Irina Desktop - Russian
        ttsEngine.setProperty("voice", voices[0].id)


def play_voice_assistant_speech(text_to_speech):
    """

    """
    ttsEngine.say(str(text_to_speech))
    ttsEngine.runAndWait()

def play_failure_phrase(*args: tuple):
    """
    Проигрывание случайной фразы при неудачном распознавании
    """
    failure_phrases = [
        "Это команда?Хотите ее добавить?"]
    play_voice_assistant_speech(failure_phrases[random.randint(0, len(failure_phrases) - 1)])

def intentAdd(*args: tuple):
    st= "Хорошо, добавляем интент"
    play_voice_assistant_speech(st)
    #TODO ДОБАВЕНИЕ ИНТЕНТА В КОНФИГ

def play_greetings(*args: tuple):
    """
    Проигрывание случайной приветственной речи
    """
    greetings = [
        "Привет, {}! Чем могу помочь?".format(person.name),
        "Здравствуй, {}! Какой план на сегодня?".format(person.name)
    ]
    play_voice_assistant_speech(greetings[random.randint(0, len(greetings) - 1)])


def play_farewell_and_quit(*args: tuple):
    """
    Проигрывание прощательной речи и выход
    """
    farewells = [
        "До связи, {}! Хорошего тебе дня!".format(person.name),
        "До скорого, {}!".format(person.name)
    ]
    play_voice_assistant_speech(farewells[random.randint(0, len(farewells) - 1)])
    ttsEngine.stop()
    quit()



def record_and_recognize_audio(*args: tuple):
    """
    Запись и распознавание аудио
    """
    with microphone:
        recognized_data = ""

        # регулирование уровня окружающего шума
        recognizer.adjust_for_ambient_noise(microphone, duration=2)

        try:
            print("Listening...")
            audio = recognizer.listen(microphone, 5, 5)

            with open("microphone-results.wav", "wb") as file:
                file.write(audio.get_wav_data())

        except speech_recognition.WaitTimeoutError:
            print("Пожалуйста, проверь, что микрофон включен")
            return

        # использование online-распознавания через Google
        # (высокое качество распознавания)
        try:
            print("Started recognition...")
            recognized_data = recognizer.recognize_google(audio, language="ru").lower()

        except speech_recognition.UnknownValueError:
            pass

        # в случае проблем с доступом в Интернет происходит
        # попытка использовать offline-распознавание через Vosk
        except speech_recognition.RequestError:
            print("Trying to use offline recognition...")
            recognized_data = use_offline_recognition()

        return recognized_data


def use_offline_recognition():
    """
    Переключение на оффлайн-распознавание речи
    :return: распознанная фраза
    """
    recognized_data = ""
    try:
        # проверка наличия модели на нужном языке в каталоге приложения
        if not os.path.exists("models/vosk-model-small-ru-0.4"):
            print("Please download the model from:\n"
                  "https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
            exit(1)

        # анализ записанного в микрофон аудио (чтобы избежать повторов фразы)
        wave_audio_file = wave.open("microphone-results.wav", "rb")
        model = Model("models/vosk-model-small-ru-0.4")
        offline_recognizer = KaldiRecognizer(model, wave_audio_file.getframerate())

        data = wave_audio_file.readframes(wave_audio_file.getnframes())
        if len(data) > 0:
            if offline_recognizer.AcceptWaveform(data):
                recognized_data = offline_recognizer.Result()

                # получение данных распознанного текста из JSON-строки
                # (чтобы можно было выдать по ней ответ)
                recognized_data = json.loads(recognized_data)
                recognized_data = recognized_data["text"]
    except:
        print("Sorry, speech service is unavailable. Try again later")

    return recognized_data


# перечень команд для использования в виде JSON-объекта
conf = """config = {
    "intents": {
        "greeting": {
            "examples": ["привет", "здравствуй", "добрый день"],
            "responses": "play_greetings"
        },
        "farewell": {
            "examples": ["пока", "до свидания", "увидимся", "до встречи"],
            "responses": "play_farewell_and_quit"
        },
        "умнаяЛампочка": {
            #  Есть сущ. - наш объект. Сканируем тексты на предмет глаголов в повелит. наклонении, располож. рядом с таким сущ.
            #  формируем Examples
            "examples": ["лампочку включи","лампочку выключи"],
            "responses": ""
            # предложить список для выбора - что работоспособно

        }

    },
    "failure_phrases": "play_failure_phrase"
}"""
#with open("config.json", 'w', encoding='utf-8') as file:
    #json.dump(config, file, indent=3, ensure_ascii=False) #запист в json


with open("config.json", 'r', encoding='utf-8') as file:
    config = json.load(file)
functions = {"play_greetings":play_greetings, 'play_farewell_and_quit':play_farewell_and_quit,'play_failure_phrase':play_failure_phrase }
#Соответственно вызов будет functions['ваша строка']()
for intent_name, intent_data in config["intents"].items():
    print("intent_name ", intent_name, "intent_data", intent_data)

print(type(config))
def prepare_corpus():
    """
    Подготовка модели для угадывания намерения пользователя
    """
    corpus = []
    target_vector = []
    for intent_name, intent_data in config["intents"].items():
        for example in intent_data["examples"]:
            corpus.append(example)
            target_vector.append(intent_name)

    training_vector = vectorizer.fit_transform(corpus)
    classifier_probability.fit(training_vector, target_vector)
    classifier.fit(training_vector, target_vector)


def get_intent(request):
    """
    Получение наиболее вероятного намерения в зависимости от запроса пользователя
    :param request: запрос пользователя
    :return: наиболее вероятное намерение
    """
    #best_intent = classifier.predict(vectorizer.transform([request]))[0]

    #index_of_best_intent = list(classifier_probability.classes_).index(best_intent)
    #probabilities = classifier_probability.predict_proba(vectorizer.transform([request]))[0]

    #best_intent_probability = probabilities[index_of_best_intent]

    # при добавлении новых намерений стоит уменьшать этот показатель
    #for a in probabilities:
        #print(a)
    #print(best_intent_probability)
    #if best_intent_probability > 0.45:
        #return best_intent
    best_intent = 0
    for intent_name, intent_data in config["intents"].items():
        if request in intent_data["examples"]:
            best_intent = intent_name
    print (best_intent)
    return best_intent






def make_preparations():
    """
    Подготовка глобальных переменных к запуску приложения
    """
    global recognizer, microphone, ttsEngine, person, assistant, translator, vectorizer, classifier_probability, classifier

    # инициализация инструментов распознавания и ввода речи
    recognizer = speech_recognition.Recognizer()
    microphone = speech_recognition.Microphone()

    # инициализация инструмента синтеза речи
    ttsEngine = pyttsx3.init()

    # настройка данных пользователя
    person = OwnerPerson()
    person.name = "Даниил"
    person.home_city = "Rostov-on-Don"
    person.native_language = "ru"
    person.target_language = "ru"

    # настройка данных голосового помощника
    assistant = VoiceAssistant()
    assistant.name = "Alice"
    assistant.sex = "female"
    assistant.speech_language = "ru"

    # установка голоса по умолчанию
    setup_assistant_voice()

    # добавление возможностей перевода фраз (из заготовленного файла)
    #translator = Translation()

    # загрузка информации из .env-файла (там лежит API-ключ для OpenWeatherMap)
    load_dotenv()

    # подготовка корпуса для распознавания запросов пользователя с некоторой вероятностью (поиск похожих)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
    classifier_probability = LogisticRegression()
    classifier = LinearSVC()
    #prepare_corpus()


if __name__ == "__main__":
    make_preparations()

    while True:
        # старт записи речи с последующим выводом распознанной речи и удалением записанного в микрофон аудио
        voice_input = record_and_recognize_audio()

        if os.path.exists("microphone-results.wav"):
            os.remove("microphone-results.wav")

        print(colored(voice_input, "blue"))

        # отделение комманд от дополнительной информации (аргументов)
        if voice_input:
            voice_input_parts = voice_input.split(" ")

            # если было сказано одно слово - выполняем команду сразу без дополнительных аргументов
            if len(voice_input_parts):
                intent = get_intent(voice_input)
                if intent:
                    functions[config["intents"][intent]["responses"]]()
                else:
                    functions[config["failure_phrases"]]()
                    tmp=voice_input
                    voice_input = record_and_recognize_audio()
                    voice_input_parts = voice_input.split(" ")
                    if voice_input_parts[0] == "да":
                        intentAdd(voice_input)



            # в случае длинной фразы - выполняется поиск ключевой фразы и аргументов через каждое слово,
            # пока не будет найдено совпадение
            #if len(voice_input_parts) > 1:
                #for guess in range(len(voice_input_parts)):
                    #intent = get_intent((" ".join(voice_input_parts[0:guess])).strip())
                    #print(intent)
                    #if intent:
                        #command_options = [voice_input_parts[guess:len(voice_input_parts)]]
                        #print(command_options)
                        #functions[config["intents"][intent]["responses"]](*command_options)
                        #break
                    #if not intent and guess == len(voice_input_parts)-1:
                        #functions[config["failure_phrases"]]()