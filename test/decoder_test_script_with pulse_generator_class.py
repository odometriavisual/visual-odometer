#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 8 2023

@author: matheusfdario,danielsantin e rossato
"""

import time
import RPi.GPIO as gpio  # import especifico para trabalhar com GPIO

class PulseGenerator:

    def __init__(self): 
        # --- defininindo portas e variáveis para trabalhar com gpio ---#

        # define que as referências de pinagem do raspberry pi devem ser referentes a lista BCM
        gpio.setmode(gpio.BCM)

        self.Encoder1PinPhaseA = 13
        self.Encoder1PinPhaseB = 19

        self.Encoder2PinPhaseA = 26
        self.Encoder2PinPhaseB = 21

        self.Encoder3PinPhaseA = 5
        self.Encoder3PinPhaseB = 6

        self.encoderPrimary = [self.Encoder1PinPhaseA, self.Encoder2PinPhaseA, self.Encoder3PinPhaseA]
        self.encoderSecondary = [self.Encoder1PinPhaseB, self.Encoder2PinPhaseB, self.Encoder3PinPhaseB]


        # definindo os pinos como saída e nível lógico baixo no estado inicial.
        for i in range(3):
            gpio.setup(self.encoderPrimary[i], gpio.OUT)
            gpio.output(self.encoderPrimary[i],gpio.LOW)
            gpio.setup(self.encoderSecondary[i], gpio.OUT)
            gpio.output(self.encoderSecondary[i],gpio.LOW)

        self.coordenadasAtuaisNoPanther = [0, 0]
        self.passoDirecaoPanther = [0, 0]
        self.escalaPanther = 0.1

        self.escalaPyCamX = 0.02151467169232321
        self.escalaPyCamY = 0.027715058926976663

        self.arrayUtilitario = [[gpio.LOW, gpio.LOW], [gpio.HIGH, gpio.LOW] , [gpio.HIGH, gpio.HIGH], [gpio.LOW, gpio.HIGH]] #representa o estado do encoder. 
        # cada valor representar um estado: 0 -> 00; 1 -> 10; 2 -> 11; 3 -> 01;
        # --- fim do setup GPIO ---#

        self.run = True
        self.encoders_state_var = [0,0,0] # define o estado do canal de encoder para os eixos x, y e z. pode receber valores de 0 a 3.


    def sendPulses(self, x, y, z=0):
        des_delta_xyz = [x,y,z]         # variável para o deslocamento em x,y,z em número de pulsos.
        state_change_dir_xyz = [0, 0, 0]    # variável que muda o estado dos canais de encoder  nos eixos xyz conforme a direção de deslocamento.

        for i in range(3):
            if des_delta_xyz[i] > 0:
                state_change_dir_xyz[i] = -1
            elif des_delta_xyz[i] < 0:
                state_change_dir_xyz[i] = 1

        for i in range(3):
            while des_delta_xyz[i] != 0:
                self.encoders_state_var[i] = (self.encoders_state_var[i] + state_change_dir_xyz[i]) % 4 #Garante que a variável  terá os valores 0,1,2,ou 3, que correnspondem as possíveis posições do encoder
                des_delta_xyz[i] = des_delta_xyz[i] + state_change_dir_xyz[i]
                gpio.output(self.encoderPrimary[i], self.arrayUtilitario[self.encoders_state_var[i]][0])
                gpio.output(self.encoderSecondary[i], self.arrayUtilitario[self.encoders_state_var[i]][1])
                time.sleep(1e-10)
            state_change_dir_xyz[i] = 0
        return 0


pg = PulseGenerator()

N = 1
des_x = 0   # deslocamento total no eixo x
des_y = 0   # deslocamento total no eixo y
while(N > 0):
    N = int(input("N:"))
    if(N > 0):
        var_x = int(input("X:"))    # variação no eixo x
        var_y = int(input("Y:"))    # variação no eixo y
        for i in range(N):
            des_x += var_x
            des_y += var_y
            pg.sendPulses(var_x,var_y)
        print(des_x,des_y)
    print("end",des_x,des_y)


# TODO: Turn off all used GPIO pin in startup.
# TODO: Test with 2 encoders.
# TODO: Reorganize code.
# TODO: Rename variables.    

# for i in range(100):
#     deltax = -1
#     cont += deltax
#     print(encoderManager.atualizarPos(deltax,deltay))
#     #time.sleep(0.01)
#     print(cont)

# def pos_mov(state_now,mov):
#     # considera um valor positivo de mov para movimentação  
#     if(mov>0):
    
#     else:
#         if(mov>0):
        
#         else:
#             return 0