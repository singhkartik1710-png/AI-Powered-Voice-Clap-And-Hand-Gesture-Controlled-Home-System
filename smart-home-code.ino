import cv2
import mediapipe as mp
import time
import threading
import queue
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import pyttsx3

# ================= सेटिंग्स =================
USE_ARDUINO = False    # ESP32/Arduino जुड़ने पर True कर देना
COM_PORT = "COM3"      # तब अपना पोर्ट डालना
HOLD_TIME = 1.0        # हाथ का जेस्चर इतने सेकंड पकड़ना है
CLAP_THRESHOLD = 0.5   # ताली की सेंसिटिविटी (नीचे टिप देखो)
COOLDOWN = 0.3         # एक कमांड के बाद इतने सेकंड दूसरी कमांड नहीं मानेगा
USE_CLAP = True
USE_VOICE = True

ser = None
if USE_ARDUINO:
    import serial
    ser = serial.Serial(COM_PORT, 9600, timeout=1)
    time.sleep(2)

# ============ Jarvis की आवाज़ (बोलने वाला हिस्सा) ============
speech_queue = queue.Queue()

def speaker_loop():
    while True:
        text = speech_queue.get()
        try:
            engine = pyttsx3.init()   # हर बार नई engine
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
        except Exception as e:
            print("Speak error:", e)

threading.Thread(target=speaker_loop, daemon=True).start()

def speak(text):
    speech_queue.put(text)

# ============ कमांड भेजना (तीनों तरीके यहीं से गुज़रते हैं) ============
lock = threading.Lock()
last_action_time = 0
light_on = False
last_status = "Ready"

def send(cmd, reply):
    """कमांड भेजता है। अगर अभी cooldown चल रहा है तो False लौटाता है।"""
    global last_action_time, light_on, last_status
    with lock:
        now = time.time()
        if now - last_action_time < COOLDOWN:
            return False
        last_action_time = now
        if cmd == "A":
            light_on = True
        if cmd == "B":
            light_on = False
        last_status = reply
    print("Command:", cmd, "|", reply)
    if ser:
        ser.write(cmd.encode())
    speak(reply)
    return True

# ============ 1) हाथ का जेस्चर ============
# 5 उंगलियां = Light ON | 0 (मुट्ठी) = Light OFF
# 2 उंगलियां = Fan ON   | 1 उंगली  = Fan OFF
ACTIONS = {
    5: ("A", "okh shikhar , light on"),
    0: ("B", "Light off, boss"),
    2: ("C", "Fan on, boss"),
    1: ("D", "Fan off, boss"),
}

def count_fingers(hand_landmarks, label):
    lm = hand_landmarks.landmark
    fingers = []
    if label == "Right":
        fingers.append(1 if lm[4].x < lm[3].x else 0)
    else:
        fingers.append(1 if lm[4].x > lm[3].x else 0)
    for tip in [8, 12, 16, 20]:
        fingers.append(1 if lm[tip].y < lm[tip - 2].y else 0)
    return sum(fingers)

def hand_gesture_loop():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)
    last_count = -1
    hold_start = time.time()
    last_sent = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            label = result.multi_handedness[0].classification[0].label
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            count = count_fingers(hand, label)

            if count != last_count:
                last_count = count
                hold_start = time.time()

            held = time.time() - hold_start
            if held >= HOLD_TIME and count in ACTIONS and count != last_sent:
                cmd, reply = ACTIONS[count]
                if send(cmd, reply):      # cooldown में रुका तो अगली बार फिर कोशिश होगी
                    last_sent = count

            cv2.putText(frame, f"Fingers: {count}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            last_count = -1
            last_sent = -1

        cv2.putText(frame, last_status, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)
        cv2.putText(frame, "Light: " + ("ON" if light_on else "OFF"), (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
        cv2.imshow("Gesture Control", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ============ 2) ताली (एक ताली = light ON/OFF बदलेगी) ============
def clap_loop():
    def callback(indata, frames, time_info, status):
        if np.max(np.abs(indata)) > CLAP_THRESHOLD:
            if light_on:
                send("B", "Light off, boss")
            else:
                send("A", "Yes boss, light on")
    try:
        with sd.InputStream(channels=1, callback=callback):
            while True:
                time.sleep(0.1)
    except Exception as e:
        print("Clap वाले माइक में दिक्कत:", e)

# ============ 3) Jarvis वॉइस कमांड ============
def handle_voice(text):
    words = text.replace(",", " ").split()
    if "jarvis" not in words:
        return
    turn_on = "on" in words or "open" in words
    turn_off = "off" in words or "close" in words

    if "light" in words:
        if turn_off:
            send("B", "Light off, boss")
        elif turn_on:
            send("A", "Yes boss, light on")
    elif "fan" in words:
        if turn_off:
            send("D", "Fan off, boss")
        elif turn_on:
            send("C", "Fan on, boss")

def voice_loop():
    recognizer = sr.Recognizer()
    try:
        mic = sr.Microphone()
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
    except Exception as e:
        print("Voice वाले माइक में दिक्कत:", e)
        return
    print("Voice तैयार है। बोलो: 'Jarvis light on'")

    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
            text = recognizer.recognize_google(audio, language="en-IN").lower()
            print("सुना:", text)
            handle_voice(text)
        except sr.WaitTimeoutError:
            pass
        except sr.UnknownValueError:
            pass
        except sr.RequestError:
            print("Voice पहचानने के लिए internet चाहिए")
            time.sleep(2)
        except Exception as e:
            print("Voice error:", e)

# ============ तीनों को एक साथ चलाना ============
if __name__ == "__main__":
    if USE_CLAP:
        threading.Thread(target=clap_loop, daemon=True).start()
    if USE_VOICE:
        threading.Thread(target=voice_loop, daemon=True).start()
    hand_gesture_loop()   # कैमरा window मुख्य हिस्से में चलती है
