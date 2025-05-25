#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
from naoqi import ALProxy
import time

IP   = "127.0.0.1"   # Cambia si tu NAO tiene otra IP
PORT = 9559         # Puerto NAOqi

motion  = ALProxy("ALMotion",      IP, PORT)
posture = ALProxy("ALRobotPosture",IP, PORT)

motion.setStiffnesses("Body", 1.0)
posture.goToPosture("StandInit", 0.5)
time.sleep(1.0)

# Prueba mover hombro derecho +30°
print("Moviendo RShoulderPitch a +30°...")
motion.angleInterpolation(["RShoulderPitch"],
                          [30*3.1416/180.0],
                          [1.0], True)
time.sleep(1.0)

# Volver a 0°
print("Volviendo RShoulderPitch a 0°...")
motion.angleInterpolation(["RShoulderPitch"],
                          [0], [1.0], True)
time.sleep(1.0)

motion.setStiffnesses("Body", 0.0)
print("✅ test_move.py completado")
