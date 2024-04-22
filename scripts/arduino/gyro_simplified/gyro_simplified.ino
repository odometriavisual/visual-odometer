#include <Arduino.h>
#include "MPU9250.h"
#define LED 2

MPU9250 mpu;

float q0 = 0;
float q1 = 0;
float q2 = 0;
float q3 = 0;

void print_quaternion() {
    digitalWrite(LED,HIGH);
    q0 = mpu.getQuaternionX();
    q1 = mpu.getQuaternionY();
    q2 = mpu.getQuaternionZ();
    q3 = mpu.getQuaternionW();
    Serial.print(q0, 2);
    Serial.print(",");
    Serial.print(q1, 2);
    Serial.print(",");
    Serial.print(q2, 2);
    Serial.print(",");
    Serial.println(q3, 2);
    digitalWrite(LED,LOW);
}

void setup() {
    pinMode(LED,OUTPUT);
    digitalWrite(LED,LOW);                                      
    Serial.begin(115200);
    Serial.setTimeout(40);
    Wire.begin();
    delay(2000);

    if (!mpu.setup(0x68)) {  // change to your own address                                                                                                                                                                
        while (1) {
            Serial.println("MPU connection failed. Please check your connection with connection_check example.");
            delay(5000);
        }
    }
}

void loop() {
    if (mpu.update()) { 
        static uint32_t prev_ms = millis();
        if (millis() > prev_ms + 40) {
            print_quaternion();
            prev_ms = millis();
        }
    }
}