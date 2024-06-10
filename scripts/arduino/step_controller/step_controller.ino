#define DIR_PIN 4
#define STEP_PIN 5

int steps_per_second = 100; 
int current_position = 0;
int step_direction = 1;
int target_position = 0;
int total_steps = 0;
int steps_to_go = 0;
float last_step_time = 0;
float step_interval = 1000000.0 / steps_per_second;
float desired_step_interval = step_interval
bool is_moving = false;
int acceleration_steps = 10; // Número de passos para aceleração/desaceleração
int initial_slow_steps = 0;
int final_slow_steps = 0;

void setup() {
  pinMode(DIR_PIN, OUTPUT);
  pinMode(STEP_PIN, OUTPUT);
  Serial.begin(115200);
  digitalWrite(DIR_PIN, HIGH);
}

void loop() {
  if (Serial.available() > 0) {
    String incoming_text = Serial.readStringUntil('\n');
    char *token = strtok(const_cast<char*>(incoming_text.c_str()), ",");
    if (token != NULL) {
      target_position = atoi(token);
      total_steps = abs(target_position - current_position);
      initial_slow_steps = total_steps * 0.1;
      final_slow_steps = total_steps * 0.1;
      steps_to_go = total_steps;
      token = strtok(NULL, ",");
      if (token != NULL) {
        desired_step_interval = 1000000.0 / atof(token);
      }
    }
    if (total_steps > 0) {
      step_direction = (target_position > current_position) ? 1 : -1;
      digitalWrite(DIR_PIN, (step_direction == 1) ? LOW : HIGH);
      is_moving = true;
    }
  }

  while (is_moving) {
    if (steps_to_go > 0) {
      float current_time = micros();
      if (current_time - last_step_time >= step_interval) {
        digitalWrite(STEP_PIN, HIGH);
        digitalWrite(STEP_PIN, LOW);
        last_step_time = current_time;
        current_position += step_direction;
        steps_to_go--;

        // Ajuste da velocidade
        if (steps_to_go > total_steps - initial_slow_steps || steps_to_go < final_slow_steps) {
          // Acelerando ou desacelerando
          step_interval = map(steps_to_go, total_steps - initial_slow_steps, total_steps, 200000, desired_step_interval);
        } else {
          // Velocidade constante
          step_interval = step_interval;
        }
      }
    } else {
      Serial.println(".");
      is_moving = false;
    }
  }
}
