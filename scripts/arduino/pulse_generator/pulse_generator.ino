
const int encoderPrimary[3] = {33,14,26};  // gpio5  -> D1   //passo do encoder 1
const int encoderSecondary[3] = {32,27,25}; // gpio14 -> D5   //direção do encoder 1


const byte numChars = 32;
char receivedChars[numChars];
boolean newData = false;

int received_delta[] = {0,0,0};
bool arrayUtilitario[4][2] = {{true,true},{true,false},{false,false},{false,true}};
int contador[3] = {0,0,0};
int multiplicador[3] = {0,0,0};

void setup()
{
  Serial.begin(115200);
  for (int i = 0; i < 3; i++) {
    pinMode(encoderPrimary[i], OUTPUT);
    pinMode(encoderSecondary[i],OUTPUT);
  }
}

void loop()
{
   recvWithEndMarker();
   if (newData){
      parseData();
      generatePulses();
   }
}

void recvWithEndMarker()
{
   static byte ndx = 0;
   char endMarker = '\n';
   char rc;

   while (Serial.available() > 0 && newData == false)
   {
      rc = Serial.read();

      if (rc != endMarker)
      {
         receivedChars[ndx] = rc;
         ndx++;
         if (ndx >= numChars)
         {
            ndx = numChars - 1;
         }
      }
      else
      {
         receivedChars[ndx] = '\0'; // terminate the string
         ndx = 0;
         newData = true;
      }
   }
}

void parseData()
{
   char *strings[8]; // an array of pointers to the pieces of the above array after strtok()
   char *ptr = NULL; byte index = 0;
   ptr = strtok(receivedChars, ",");  // delimiters, semicolon
   while (ptr != NULL)
   {
      strings[index] = ptr;
      index++;
      ptr = strtok(NULL, ",");
   }

   received_delta[0] = atoi(strings[0]);
   received_delta[1] = atoi(strings[1]);
   received_delta[2] = atoi(strings[2]);

   newData = false;

   /*

   Serial.print(received_delta[0]); Serial.print(",");
   Serial.print(received_delta[1]); Serial.print(",");
   Serial.print(received_delta[2]); Serial.println();
   */

}


void generatePulses(){
  for (int i = 0; i < 3; i++) {
    if(received_delta[i]>0){
      multiplicador[i] = 1;
    } else if (received_delta[i] < 0){
      multiplicador[i] = -1;
    } else {
      multiplicador[i] = 0;
    }
  }

  while(multiplicador[0] != 0 || multiplicador[1] != 0 || multiplicador[2] != 0){
    for (int i = 0; i < 3; i++) {
      if(multiplicador[i] != 0){
        received_delta[i] = received_delta[i] - multiplicador[i];
        contador[i] = (4 + contador[i] - multiplicador[i])%4;
        digitalWrite(encoderPrimary[i],arrayUtilitario[contador[i]][0]);
        digitalWrite(encoderSecondary[i],arrayUtilitario[contador[i]][1]);
        if(received_delta[i] == 0){
          multiplicador[i] = 0;
          //Serial.println("=");
        }
      }
    } 
  }
}

