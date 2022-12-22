import pyautogui as pag
import time
while True:
    try:
        time.sleep(1)
        x, y = pag.position()
        print(x, y)
    except KeyboardInterrupt:
        break