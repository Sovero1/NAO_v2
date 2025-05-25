import numpy as np
import mediapipe as mp

class JointPoint:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

class JointType:
    # MediaPipe Pose Landmarks 
    Nose = 0
    LeftEye = 1
    RightEye = 2
    LeftEar = 3
    RightEar = 4
    LeftShoulder = 11
    RightShoulder = 12
    LeftElbow = 13
    RightElbow = 14
    LeftWrist = 15
    RightWrist = 16
    LeftHip = 23 
    RightHip = 24
    LeftKnee = 25
    RightKnee = 26
    LeftAnkle = 27
    RightAnkle = 28
    LeftFootIndex = 31
    RightFootIndex = 32
    SpineBase = 23  # Aproximamos con LeftHip
    SpineMid = 0    # Se calcula como promedio
    SpineShoulder = 100  # Se calcula como promedio

class HolisticData:
    def __init__(self, holistic_result):
        self.bodyJointsArray = {}
        self.handState = { "LEFT_HAND": False, "RIGHT_HAND": False }
        self.headRotationAngle = JointPoint()

        pose = holistic_result.pose_landmarks
        if pose:
            self._load_pose_data(pose)
            self._estimate_spine_points()
        
        # Detectar si las manos están abiertas
        self.handState["LEFT_HAND"] = self._is_hand_open(holistic_result.left_hand_landmarks)
        self.handState["RIGHT_HAND"] = self._is_hand_open(holistic_result.right_hand_landmarks)

    def _load_pose_data(self, pose_landmarks):
        for idx, lm in enumerate(pose_landmarks.landmark):
            self.bodyJointsArray[idx] = JointPoint(lm.x, lm.y, lm.z)

    def _estimate_spine_points(self):
        # Promedio de hombros para SpineShoulder
        ls = self.bodyJointsArray.get(JointType.LeftShoulder)
        rs = self.bodyJointsArray.get(JointType.RightShoulder)
        if ls and rs:
            self.bodyJointsArray[JointType.SpineShoulder] = self._average_points([ls, rs])
        
        # Promedio de caderas para SpineMid
        lh = self.bodyJointsArray.get(JointType.LeftHip)
        rh = self.bodyJointsArray.get(JointType.RightHip)
        if lh and rh:
            self.bodyJointsArray[JointType.SpineMid] = self._average_points([lh, rh])

    def _average_points(self, points):
        x = sum(p.x for p in points) / len(points)
        y = sum(p.y for p in points) / len(points)
        z = sum(p.z for p in points) / len(points)
        return JointPoint(x, y, z)

    def _is_hand_open(self, hand_landmarks):
        if hand_landmarks is None:
            return False
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        dist = ((thumb_tip.x - index_tip.x) ** 2 +
                (thumb_tip.y - index_tip.y) ** 2 +
                (thumb_tip.z - index_tip.z) ** 2) ** 0.5
        return dist > 0.05