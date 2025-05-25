#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
nao_control.py

Script para controlar un robot NAO mediante NAOqi (Python 2.7).
Escucha ángulos corporales enviados en JSON por socket y mueve las articulaciones del robot.
"""
from __future__ import print_function
import sys
import socket
import json
import math
from naoqi import ALProxy

# Mapeo de nombres de ángulos a nombres de articulaciones de NAO
ANGLE_MAP = {
    "LShoulderPitch": "LShoulderPitch",
    "RShoulderPitch": "RShoulderPitch",
    "LElbowRoll":     "LElbowRoll",
    "RElbowRoll":     "RElbowRoll",
    "LHipPitch":      "LHipPitch",
    "RHipPitch":      "RHipPitch",
    "LKneePitch":     "LKneePitch",
    "RKneePitch":     "RKneePitch",
    "HeadPitch":      "HeadPitch",
    "HeadYaw":        "HeadYaw",
    "HipYawPitch":    "HipYawPitch",
    "LWristYaw": "LWristYaw",
    "RWristYaw": "RWristYaw",
    "LHand": "LHand",
    "RHand": "RHand",

}

NAO_IP = "localhost"  # Cambiar a la IP de tu robot NAO
NAO_PORT = 64156
SOCK_IP = "127.0.0.1"
SOCK_PORT = 6000       # Puerto para recibir ángulos JSON
BUFFER_SIZE = 4096


def deg2rad(deg):
    return deg * math.pi / 180.0


def main(robot_ip, robot_port, sock_ip, sock_port):
    # Conectar a los proxies de NAO
    try:
        motion = ALProxy("ALMotion", robot_ip, robot_port)
        posture = ALProxy("ALRobotPosture", robot_ip, robot_port)
    except Exception as e:
        print("Error al conectar con NAOqi:", e)
        sys.exit(1)

    # Poner rigidez a las articulaciones
    motion.setStiffnesses("Body", 1.0)
    # Postura inicial
    posture.goToPosture("StandInit", 0.5)

    # Configurar socket TCP para escuchar ángulos
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((sock_ip, sock_port))
    s.listen(1)
    print("Esperando conexión en {}:{}...".format(sock_ip, sock_port))
    conn, addr = s.accept()
    print("Conectado desde", addr)
    conn_file = conn.makefile('r')  # permite leer línea por línea
    try:
        while True:
            line = conn_file.readline()
            if not line:
                break
            try:
                angles_dict = json.loads(line.strip())
                print(">>>>> Recibido:", angles_dict)

            except ValueError:
                print("Datos JSON inválidos recibidos: {}".format(line))
                continue

            # Preparar listas para NAOqi
            joint_names = []
            target_angles = []
            for key, joint in ANGLE_MAP.items():
                if key in angles_dict and key not in ('LHand','RHand'):
                    joint_names.append(joint)
                    # Convertir grados a radianes
                    target_angles.append(deg2rad(angles_dict[key]))

            if joint_names:
                try:
                    motion.setAngles(joint_names, target_angles, 0.3)  # velocidad ajustable
                except Exception as e:
                    print("Error moviendo articulaciones:", e)
    finally:
        conn.close()
        s.close()
        # Bajar rigidez
        motion.setStiffnesses("Body", 0.0)
        print("Conexión cerrada, robot en reposo.")


if __name__ == '__main__':
    # Parámetros por defecto o desde línea de comandos
    ip = NAO_IP
    port = NAO_PORT
    if len(sys.argv) >= 2:
        ip = sys.argv[1]
    if len(sys.argv) >= 3:
        port = int(sys.argv[2])

    print("NAO Control: Robot {}:{}".format(ip, port))
    main(ip, port, SOCK_IP, SOCK_PORT)
