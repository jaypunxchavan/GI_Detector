/*
  GI Sensor — TIA Signal Reader
  =============================
  Reads the transimpedance amplifier (TIA) output from the MCP6002
  op-amp circuit on analog pin A0. The TIA converts the microamp-level
  current from the glucometer strip's electrochemical reaction into a
  0-5V signal that the Arduino's built-in 10-bit ADC can resolve.

  No external libraries required — this firmware uses only the
  built-in analogRead() function. The INA219 current sensor is not
  used in this revision of the circuit.

  Circuit summary:
    - MCP6002 pin 7 (VDD) -> Arduino 5V
    - MCP6002 pin 4 (VSS) -> Arduino GND
    - MCP6002 pin 3 (non-inverting input) -> Arduino GND (true 0V reference)
    - MCP6002 pin 6 (output) -> Arduino A0
    - 1M ohm feedback resistor between pin 6 (output) and pin 2 (inverting input)
    - 4.7k ohm series resistor between strip electrode 1 and pin 2
    - Strip electrode 2 -> bias divider node (~0.45V from 5V through
      100k to node, node through 10k to GND)

  Serial output format (CSV, one line per reading):
    timestamp_ms,adc_raw,voltage

    timestamp_ms : milliseconds since Arduino powered on
    adc_raw      : raw 10-bit ADC value, 0-1023
    voltage      : adc_raw converted to volts (0.000 - 5.000)

  calibration.py and auc_calc.py expect adc_raw as the value recorded
  in calibration_readings.csv and digestion_results.csv.

  Usage:
    Upload this sketch, then open Serial Monitor at 9600 baud.
    Readings print every 500ms.
*/

const int SENSOR_PIN = A0;
const float VREF = 5.0;
const int ADC_MAX = 1023;
const unsigned long READ_INTERVAL_MS = 500;

void setup() {
  Serial.begin(9600);

  // Wait for serial connection (needed on some boards, harmless on others)
  while (!Serial) {
    ; // wait
  }

  // Print header row so the CSV is self-describing if logged directly
  Serial.println("timestamp_ms,adc_raw,voltage");
}

void loop() {
  int raw = analogRead(SENSOR_PIN);
  float voltage = (raw / (float)ADC_MAX) * VREF;

  Serial.print(millis());
  Serial.print(",");
  Serial.print(raw);
  Serial.print(",");
  Serial.println(voltage, 4);

  delay(READ_INTERVAL_MS);
}
