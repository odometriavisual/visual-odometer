#include "MPU9250.h"

MPU9250 mpu; // You can also use MPU9255 as is

void setup() {
    Serial.begin(115200);
    Wire.begin();
    delay(2000);
    mpu.setup(0x68);  // change to your own address
}

void loop() {
  if (mpu.update()) {
      Serial.print(mpu.getQuaternionX()); Serial.print(", ");
      Serial.print(mpu.getQuaternionY()); Serial.print(", ");
      Serial.print(mpu.getQuaternionZ()); Serial.print(", ");
      Serial.println(mpu.getQuaternionW());
  }
} 