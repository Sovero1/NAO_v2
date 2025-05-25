import cv2
import mediapipe as mp
import socket
import json
import time
import json
from holistic_data import HolisticData, JointType
from body_angles_copy import getBodyAngles

# Función para limitar valores
def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)

def clamp_angles_for_nao(angles):
    clamped = {}
    for key, value in angles.items():
        if key.endswith("KNEE_PITCH"):
            clamped[key] = clamp(value, 0, 120)
        elif key.endswith("HIP_PITCH"):
            clamped[key] = clamp(value, -45, 30)
        elif key.endswith("ELBOW_ROLL"):
            clamped[key] = clamp(value, -88, -2)
        elif key.endswith("SHOULDER_PITCH"):
            clamped[key] = clamp(value, -90, 90)
        elif key == "HEAD_PITCH":
            clamped[key] = clamp(value, -40, 30)
        elif key == "HEAD_YAW":
            clamped[key] = clamp(value, -120, 120)
        elif key == "VAREPSILON":
            clamped[key] = clamp(value, -60, 40)
        else:
            clamped[key] = value
    return clamped

# Suavizado exponencial
class AngleSmoother:
    def __init__(self, alpha=0.2):
        self.previous = {}
        self.alpha = alpha

    def smooth(self, angles):
        smoothed = {}
        for key, new_val in angles.items():
            if key in self.previous:
                prev_val = self.previous[key]
                smoothed_val = self.alpha * new_val + (1 - self.alpha) * prev_val
            else:
                smoothed_val = new_val
            smoothed[key] = smoothed_val
            self.previous[key] = smoothed_val
        return smoothed

# Validación
def is_body_fully_detected(data):
    joints = data.bodyJointsArray
    required = [
        JointType.LeftShoulder, JointType.LeftElbow, JointType.LeftWrist,
        JointType.RightShoulder, JointType.RightElbow, JointType.RightWrist,
        JointType.LeftHip, JointType.LeftKnee, JointType.LeftAnkle,
        JointType.RightHip, JointType.RightKnee, JointType.RightAnkle,
        JointType.SpineMid, JointType.SpineShoulder, JointType.Nose
    ]
    return all(j in joints for j in required)

# Conexión al socket del NAO
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("127.0.0.1", 6000))  # IP del servidor NAO/controlador

# Inicialización
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture("prueba2.mp4")
smoother = AngleSmoother(alpha=0.2)
start_time = time.time()
session_data=[]

with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        data = HolisticData(results)  

        annotated = frame.copy()
        mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(annotated, results.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(annotated, results.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        if results.pose_landmarks:
            data = HolisticData(results)

            if is_body_fully_detected(data) and time.time() - start_time > 2:
                angles = getBodyAngles(data)
                angles = clamp_angles_for_nao(angles)
                angles = smoother.smooth(angles)
                
                frame_time = time.time() - start_time
                session_data.append({
                    "timestamp": round(frame_time, 3),
                    "angles": angles
                })

                try:# justo antes de sock.sendall(...)
                    payload = json.dumps(angles) +'\n'
                    print("Enviando al NAO:", payload)
                    sock.sendall(payload.encode("utf-8"))
                    print("-->>> Ángulos enviados.")
                except Exception as e:
                    print("Error al enviar ángulos:", e)
            else:
                cv2.putText(annotated, "Cuerpo no detectado", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            cv2.putText(annotated, "No se detecta cuerpo", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("NAO Tracker", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
sock.close()
# Guardar sesión completa
with open("output_session.json", "w") as f:
    json.dump(session_data, f, indent=2)
print("✅ Sesión guardada en output_session.json")



def replay_session_from_file(path, delay=0.01):
    print("🔁 Reproduciendo sesión desde", path)
    try:
        with open(path) as f:
            session = json.load(f)
    except Exception as e:
        print("❌ Error al cargar sesión:", e)
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", 6000))
    print("✅ Reconectado al NAO para reproducción")

    start = time.time()
    for frame in session:
        t_target = frame["timestamp"]
        delta = t_target - (time.time() - start)
        if delta > 0:
            time.sleep(delta)
        payload = json.dumps(frame["angles"]) + "\n"
        sock.sendall(payload.encode("utf-8"))
        print("⏩ Enviado:", frame["angles"])

    sock.close()
    print("✅ Reproducción completa.")
