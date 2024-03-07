#define DIR_PIN 3
#define STEP_PIN 4
#define END_GATE_PIN 10

int steps_per_second = 100; 
int current_position = 0;
int step_direction = 1;
int target_position = 0;
unsigned long last_step_time = 0;
unsigned long step_interval = 1000/steps_per_second;
bool is_moving = false;


void setup() {
  pinMode(DIR_PIN, OUTPUT);
  pinMode(STEP_PIN, OUTPUT);
  pinMode(END_GATE_PIN, INPUT_PULLUP);
  Serial.begin(115200);
  digitalWrite(DIR_PIN, HIGH);
  while(digitalRead(END_GATE_PIN) == 0){
      delayMicroseconds(5000);
      digitalWrite(STEP_PIN, HIGH);
      delayMicroseconds(2); 
      digitalWrite(STEP_PIN, LOW);
  }

  digitalWrite(DIR_PIN, LOW);
  for (int i = 0; i <= 50; i++) {
      delayMicroseconds(5000);
      digitalWrite(STEP_PIN, HIGH);
      delayMicroseconds(2); 
      digitalWrite(STEP_PIN, LOW);
  }
  delay(1);
  Serial.println(".");
}

void loop() {
  if (Serial.available() > 0) {
    String incoming_text = Serial.readStringUntil('\n');
    char *token = strtok(const_cast<char*>(incoming_text.c_str()), ",");
    if (token != NULL) {
      target_position = atoi(token); 
      token = strtok(NULL, ",");
      if (token != NULL) {
        step_interval = 1000/atoi(token);
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

  if (current_position != target_position) {
    unsigned long current_time = millis();
    if (current_time - last_step_time >= step_interval) {
      digitalWrite(STEP_PIN, HIGH);
      delayMicroseconds(2); 
      digitalWrite(STEP_PIN, LOW);
      last_step_time = current_time;
      current_position = current_position + step_direction;
    }
  } else if (is_moving == true) {
    Serial.println(".");
    is_moving = false;
  }
  if(digitalRead(END_GATE_PIN)){
    digitalWrite(DIR_PIN, LOW);
    for (int i = 0; i <= 50; i++) {
        delayMicroseconds(5000);
        digitalWrite(STEP_PIN, HIGH);
        delayMicroseconds(2); 
        digitalWrite(STEP_PIN, LOW);
    }
    exit(0);
  }
}


