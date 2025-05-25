import numpy as np
import math
from holistic_data import JointType, HolisticData

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

def get_right_shoulder_angles_ga_lvjj(joints):
    S = joints[JointType.RightShoulder]
    E = joints[JointType.RightElbow]
    W = joints[JointType.RightWrist] 
    LS = joints[JointType.LeftShoulder]
    RS = joints[JointType.RightShoulder]
    SM = joints[JointType.SpineMid]

    v_upper = vector(S, E)
    v_forearm = vector(E, W)
    v_shoulder_line = vector(LS, RS)
    v_torso_up = vector(SM, S)

    x_axis = -normalize(v_shoulder_line)    
    z_axis = normalize(v_torso_up)
    y_axis = normalize(np.cross(z_axis, x_axis))

    proj_pitch = project_onto_plane(v_upper, x_axis)
    pitch_angle = angle_between(proj_pitch, y_axis)
    if np.dot(v_upper, z_axis) < 0:
        pitch_angle = -pitch_angle

    proj_roll = project_onto_plane(v_upper, y_axis)
    roll_angle = angle_between(proj_roll, x_axis)
    if np.dot(v_upper, y_axis) > 0:
        roll_angle = -roll_angle

    return math.degrees(-pitch_angle), math.degrees(roll_angle)


def get_left_shoulder_angles_ga_lvjj(joints):
    S = joints[JointType.LeftShoulder]
    E = joints[JointType.LeftElbow]
    LS = joints[JointType.LeftShoulder]
    RS = joints[JointType.RightShoulder]
    SM = joints[JointType.SpineMid]

    v_upper = vector(S, E)
    v_shoulder_line = vector(LS, RS)
    v_torso_up = vector(SM, S)

    x_axis = -normalize(v_shoulder_line)    
    z_axis = normalize(v_torso_up)
    y_axis = normalize(np.cross(z_axis, x_axis))

    proj_pitch = project_onto_plane(v_upper, x_axis)
    pitch_angle = angle_between(proj_pitch, y_axis)
    if np.dot(v_upper, z_axis) < 0:
        pitch_angle = -pitch_angle
  

    proj_roll = project_onto_plane(v_upper, y_axis)
    roll_angle = angle_between(proj_roll, x_axis)
    if np.dot(v_upper, y_axis) < 0:
        roll_angle = -roll_angle

    return math.degrees(-pitch_angle), math.degrees(roll_angle)


# --- Elbow Angles (Flexión codo) ---


# --- MANOS ---


# --- Clamp en grados para límites NAO ---

def clamp_deg(val, lo, hi):
    return max(lo, min(hi, val))

# --- Función pública ---

def getBodyAngles(data):
    """
    Devuelve un diccionario con los ángulos de hombros y codos,
    listos para enviar al NAO.
    """
    joints= data.bodyJointsArray
    # Hombros
    rsp, rsr = get_right_shoulder_angles_ga_lvjj(joints)
    lsp, lsr = get_left_shoulder_angles_ga_lvjj(joints)
    
    right_hand_open = data.handState.get("RIGHT_HAND", False)
    left_hand_open = data.handState.get("LEFT_HAND", False)


    # Codos
  

    # Aplicar límites
    angles = {
        "RShoulderPitch": clamp_deg(rsp, -119.5, 119.5),
        "RShoulderRoll":  clamp_deg(rsr, -18.0, 76.0),
        #"RElbowRoll":     clamp_deg( -88.5, -2.0),
        "LShoulderPitch": clamp_deg(lsp, -119.5, 119.5),
        "LShoulderRoll":  clamp_deg(lsr, -76.0, 18.0),
        #"LElbowRoll":     clamp_deg( 2.0, 88.5)
        "LHand": 1 if data.handState["LEFT_HAND"] else 0,
        "RHand": 1 if data.handState["RIGHT_HAND"] else 0,
    }
    return angles
