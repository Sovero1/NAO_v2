import numpy as np
import math
from holistic_data import JointType

# --- Basic Vector Operations ---

def vector(p, q):
    """
    Devuelve el vector 3D de p a q.
    p, q tienen atributos x,y,z.
    """
    return np.array([q.x - p.x, q.y - p.y, q.z - p.z])


def normalize(v):
    """
    Normaliza el vector v.
    """
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-8 else v


def project_onto_plane(v, normal):
    """
    Proyecta v en el plano perpendicular a normal.
    normal debe estar normalizado.
    """
    return v - np.dot(v, normal) * normal


def angle_between(v1, v2):
    """
    Retorna el ángulo en radianes entre v1 y v2.
    """
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    cosang = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return math.acos(cosang)

# --- Shoulder Angles (Pitch, Roll) following GA-LVVJ ---

def get_right_shoulder_angles(joints):
    S = joints[JointType.RightShoulder]
    E = joints[JointType.RightElbow]
    LS = joints[JointType.LeftShoulder]
    RS = joints[JointType.RightShoulder]
    SM = joints[JointType.SpineMid]

    # Brazo
    v_upper = vector(S, E)

    # Vector que define plano frontal (horizontal entre hombros)
    shoulder_line = vector(LS, RS)
    vertical_torso = vector(SM, S)

    # Normal del plano frontal del torso
    normal = normalize(np.cross(shoulder_line, vertical_torso))

    # --- Pitch ---
    z = normalize(vertical_torso)
    y = normalize(np.cross(normal, z))
    proj_pitch = project_onto_plane(v_upper, np.cross(z, y))
    theta_pitch = angle_between(proj_pitch, y)
    if np.dot(v_upper, z) < 0:
        theta_pitch = -theta_pitch

    # --- Roll ---
    # Usamos plano frontal del torso
    proj_roll = project_onto_plane(v_upper, normal)
    theta_roll = angle_between(proj_roll, shoulder_line)
    if np.dot(v_upper, normal) > 0:
        theta_roll = -theta_roll

    return math.degrees(theta_pitch), math.degrees(theta_roll)

def get_left_shoulder_angles(joints):
    S = joints[JointType.LeftShoulder]
    E = joints[JointType.LeftElbow]
    LS = joints[JointType.LeftShoulder]
    RS = joints[JointType.RightShoulder]
    SM = joints[JointType.SpineMid]

    v_upper = vector(S, E)
    shoulder_line = vector(LS, RS)
    vertical_torso = vector(SM, S)

    normal = normalize(np.cross(shoulder_line, vertical_torso))

    # --- Pitch ---
    z = normalize(vertical_torso)
    y = normalize(np.cross(normal, z))
    proj_pitch = project_onto_plane(v_upper, np.cross(z, y))
    theta_pitch = angle_between(proj_pitch, y)
    if np.dot(v_upper, z) < 0:
        theta_pitch = -theta_pitch
    theta_pitch = -theta_pitch  # invertimos para brazo izquierdo

    # --- Roll ---
    proj_roll = project_onto_plane(v_upper, normal)
    theta_roll = angle_between(proj_roll, shoulder_line)
    if np.dot(v_upper, normal) < 0:
        theta_roll = -theta_roll

    return math.degrees(theta_pitch), math.degrees(theta_roll)

# --- Elbow Angles (Flexión codo) ---

def get_right_elbow_angle(joints):
    """
    Calcula RightElbowRoll (flexión) en grados.
    Solo un DOF en el codo del NAO.
    """
    # Puntos
    S = joints[JointType.RightShoulder]
    E = joints[JointType.RightElbow]
    W = joints[JointType.RightWrist]

    # Vectores
    v_upper = normalize(vector(S, E))     # eje de rotación
    v_forearm = vector(E, W)              # antebrazo

    # Proyectar antebrazo en plano perpendicular al eje del brazo
    proj = v_forearm - np.dot(v_forearm, v_upper) * v_upper

    ref = normalize(np.cross(v_upper, np.array([0, 1, 0])))
    if np.linalg.norm(ref) < 1e-3:
        # Manejo de caso degenerado (brazo vertical): elegimos otro eje
        ref = normalize(np.cross(v_upper, np.array([1, 0, 0])))

    # Ahora medimos el ángulo entre proj y ref en el plano del codo:
    theta = angle_between(proj, ref)

    # Determinamos el signo: si el antebrazo se dobla hacia adelante o atrás:
    # miramos la componente en Y global (o usando otro eje local del torso):
    if np.dot(v_forearm, np.array([0, -1, 0])) < 0:
        theta = -theta

    return math.degrees(theta)

def get_left_elbow_angle(joints):
    """
    Calcula LeftElbowRoll (flexión) en grados:
    un solo DOF en la articulación del codo del NAO.
    """
    # Puntos del esqueleto
    S = joints[JointType.LeftShoulder]
    E = joints[JointType.LeftElbow]
    W = joints[JointType.LeftWrist]

    # Vector eje de rotación (brazo)
    v_upper = normalize(vector(S, E))
    # Vector antebrazo
    v_forearm = vector(E, W)

    # Proyectar antebrazo en el plano perpendicular al eje del brazo
    proj = v_forearm - np.dot(v_forearm, v_upper) * v_upper

    # Vector de referencia en el mismo plano:
    # cruz de v_upper con el eje Y global (arbitrario consistente)
    ref = normalize(np.cross(v_upper, np.array([0, 1, 0])))
    if np.linalg.norm(ref) < 1e-3:
        # Caso degenerado (brazo muy vertical): usar X global
        ref = normalize(np.cross(v_upper, np.array([1, 0, 0])))

    # Ángulo entre la proyección y el vector de referencia
    theta = angle_between(proj, ref)

    # para el izquierdo invertimos la condición para mantener consistencia
    if np.dot(v_forearm, np.array([0, -1, 0])) > 0:
        theta = -theta

    return math.degrees(theta)

# --- Clamp en grados para límites NAO ---

def clamp_deg(val, lo, hi):
    return max(lo, min(hi, val))

# --- Función pública ---

def getBodyAngles(joints):
    """
    Devuelve un diccionario con los ángulos de hombros y codos,
    listos para enviar al NAO.
    """
    # Hombros
    rsp, rsr = get_right_shoulder_angles(joints)
    lsp, lsr = get_left_shoulder_angles(joints)

    # Codos
    rer = get_right_elbow_angle(joints)
    ler = get_left_elbow_angle(joints)

    # Aplicar límites
    angles = {
        "RShoulderPitch": clamp_deg(rsp, -119.5, 119.5),
        "RShoulderRoll":  clamp_deg(rsr, -18.0, 76.0),
        "RElbowRoll":     clamp_deg(rer, -88.5, -2.0),
        "LShoulderPitch": clamp_deg(lsp, -119.5, 119.5),
        "LShoulderRoll":  clamp_deg(lsr, -76.0, 18.0),
        "LElbowRoll":     clamp_deg(ler, 2.0, 88.5)
    }
    return angles
