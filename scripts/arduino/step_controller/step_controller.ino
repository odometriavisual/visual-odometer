#include <Arduino.h>

#define DIR_PIN 4 // Pino conectado ao DIR+ do driver de motor de passo;
#define STEP_PIN 5 // Pino conectado ao STEP+ do driver de motor de passo;

int steps_per_second = 100; 
int current_position = 0;
int step_direction = 1;
int target_position = 0;
int total_steps = 0;
int steps_to_go = 0;
int steps_perfomed = 0;
float last_step_time = 0;
float step_interval = 1000000.0 / steps_per_second;
float desired_step_interval = step_interval;
bool is_moving = false;
bool is_motor_working = false; // Motor está funcionando ou não;
int acceleration_steps = 10; // Número de passos para aceleração/desaceleração
int initial_slow_steps = 0;
int final_slow_steps = 0;
float desired_speed, current_speed;

void setup() {
  Serial.flush();
  pinMode(DIR_PIN, OUTPUT);
  pinMode(STEP_PIN, OUTPUT);
  Serial.begin(115200);
  digitalWrite(DIR_PIN, HIGH);
  Serial.println("Setup Completed.");
}

void loop() {
  if (Serial.available() > 0) {
    // Obtém o texto da serial:
    String incoming_text = Serial.readStringUntil('\n');
    char *token = strtok(const_cast<char*>(incoming_text.c_str()), ",");
    Serial.println("Token: ");
    Serial.print(token);

    if (token != NULL) {
      target_position = atoi(token);
      total_steps = abs(target_position - current_position);
      
      // A rampa de subida e descida é 10 % do número total de passos:
      initial_slow_steps = total_steps * .1;
      final_slow_steps = total_steps * .1;

      steps_to_go = total_steps;
      token = strtok(NULL, ",");
      if (token != NULL) {
        desired_step_interval = 1000000.0 / atof(token); // Convertendo de micro segundos para micro segundos;
        desired_speed = atof(token);
        step_interval = desired_step_interval;
      }
      Serial.println("Numero de passos:");
      Serial.print(desired_step_interval);

      if (total_steps > 0) {
        step_direction = (target_position > current_position) ? 1 : -1;
        digitalWrite(DIR_PIN, (step_direction == 1) ? LOW : HIGH);
        is_moving = true;
      }
    }
  }

  while (is_moving) {
    if (steps_to_go > 0) {
      float current_time = micros();
      if (current_time - last_step_time >= step_interval) {
        digitalWrite(STEP_PIN, HIGH);
        digitalWrite(13, HIGH);  
        digitalWrite(STEP_PIN, LOW);
        digitalWrite(13, LOW);  
        last_step_time = current_time;
        current_position += step_direction;
        steps_to_go--;
        
        // Determine the current step position relative to the total steps
        steps_perfomed = total_steps - steps_to_go;
        current_speed = 1 / (1000000.0) * step_interval;
      }

      // Apply soft start (ramp-up) at the beginning
      if (steps_perfomed > 0 && steps_perfomed <= initial_slow_steps) {
        float ramp_up_factor  = float(desired_speed - 0)/initial_slow_steps; // Transform speed to period;
        step_interval = 1000000.0 / (steps_perfomed * ramp_up_factor);
      }
      // Apply soft stop (ramp-down) at the end
      else if (steps_perfomed > (total_steps - final_slow_steps) && steps_perfomed <= total_steps) {
        int steps_in = steps_perfomed - (total_steps - final_slow_steps);
        float ramp_down_factor  = desired_speed/final_slow_steps; // Transform speed to period;
        step_interval = 1000000.0 / (desired_speed - steps_in * ramp_down_factor);
      }
      // Continue with regular speed in between
      else {
        //Serial.println("Fast");
        step_interval = desired_step_interval;
      }
      
    } else {
      Serial.println("Parou de andar.");
      is_moving = false;
    }
  }
}
