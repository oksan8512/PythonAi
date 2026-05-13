
import os

from gtts import gTTS

# ''' ''' - використовується трьо лапки для тексту типу віршика

text = '''
    Кіт-рибалка у човні
    Мріє на світанні:
    – От якби зловить мені
    Карася в сметані!

'''

tts = gTTS(text=text, lang='uk')

fileName = 'sheva.mp3'

tts.save(fileName)
os.system(f'start {fileName}')
