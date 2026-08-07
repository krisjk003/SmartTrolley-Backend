#include <WiFi.h>
#include <WebServer.h>

// ==========================
// WiFi Credentials
// ==========================
const char* ssid = "realme C75 5G 149b";
const char* password = "12345678";

// HTTP Server Port
const int SERVER_PORT = 80;

WebServer server(SERVER_PORT);

// ==========================
// HTTP Handlers
// ==========================
void handleForward() {
  Serial.println("PERSON POSITION : CENTER");
  server.send(200, "text/plain", "OK");
}

void handleLeft() {
  Serial.println("PERSON POSITION : LEFT");
  server.send(200, "text/plain", "OK");
}

void handleRight() {
  Serial.println("PERSON POSITION : RIGHT");
  server.send(200, "text/plain", "OK");
}

void handleStop() {
  Serial.println("PERSON LOST / STOP");
  server.send(200, "text/plain", "OK");
}

// ==========================
// Setup
// ==========================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("======================================");
  Serial.println(" ESP32 Robot HTTP Server");
  Serial.println("======================================");

  // Connect to your WiFi
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  Serial.print("Connecting");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\n");
  Serial.println("WiFi Connected Successfully!");
  Serial.println("--------------------------------------");
  Serial.print("WiFi Name : ");
  Serial.println(ssid);

  Serial.print("ESP32 IP  : ");
  Serial.println(WiFi.localIP());

  Serial.print("HTTP Port : ");
  Serial.println(SERVER_PORT);
  Serial.println("--------------------------------------");

  // Register Endpoints
  server.on("/F", HTTP_GET, handleForward);
  server.on("/L", HTTP_GET, handleLeft);
  server.on("/R", HTTP_GET, handleRight);
  server.on("/S", HTTP_GET, handleStop);

  server.begin();

  Serial.println("HTTP Server Started");
  Serial.println("Available Endpoints:");
  Serial.println("  /F  -> CENTER");
  Serial.println("  /L  -> LEFT");
  Serial.println("  /R  -> RIGHT");
  Serial.println("  /S  -> STOP");
  Serial.println("======================================");
}

void loop() {
  server.handleClient();
}