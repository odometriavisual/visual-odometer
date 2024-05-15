#define DIR_PIN 4
#define STEP_PIN 5

int steps_per_second = 100; 
int current_position = 0;
int step_direction = 1;
int target_position = 0;
float last_step_time = 0;
float step_interval = 1000/steps_per_second;
bool is_moving = false;


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
      token = strtok(NULL, ",");
      if (token != NULL) {
        //Serial.print("Token: ");
        //Serial.println(token);
        //Serial.print("ATOF TOKEN: ");
        //Serial.println(atof(token));
        //Serial.print("Step: ");

        step_interval = 1000000/atof(token);
        //Serial.println(step_interval);
      }
    }
    if (target_position - current_position > 0) {
      step_direction = 1;
      digitalWrite(DIR_PIN, LOW);
      is_moving = true;
    } else if (target_position - current_position < 0) {
      step_direction = -1;
      digitalWrite(DIR_PIN, HIGH);
      is_moving = true;
    }
  }
  while (is_moving == true){
    if (current_position != target_position) {
      float current_time = micros();
      if (current_time - last_step_time >= step_interval) {
        digitalWrite(STEP_PIN, HIGH);
        digitalWrite(STEP_PIN, LOW);
        last_step_time = current_time;
        current_position = current_position + step_direction;
      }
    } else if (is_moving == true) {
      Serial.println(".");
      is_moving = false;
    }
  }
}


