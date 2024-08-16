from flask import Flask, Response, request, jsonify
import cv2
import requests

app = Flask(__name__)

def generate_frames():
    vid = cv2.VideoCapture('http://raspberrypi00.local:7123/stream.mjpg')
    while True:
        success, frame = vid.read()
        if not success:
            break
        else:
            # Converte o frame para JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Retorna o frame em formato de bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html>
    <head>
        <style>
            .video-frame {
                width: 640px;
                height: 480px;
            }
        </style>
        <script>
            async function setFocus() {
                const focusValue = document.getElementById('focus_value').value;
                if (focusValue) {
                    try {
                        const response = await fetch('/set_focus', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ focus_value: focusValue }),
                        });
                        const result = await response.text();
                        document.getElementById('status').innerText = result;
                    } catch (error) {
                        document.getElementById('status').innerText = 'Erro ao ajustar o foco.';
                    }
                }
            }
        </script>
    </head>
    <body>
        <h1>Streaming de Vídeo</h1>
        <img src='/video_feed' class='video-frame'/>
        <br>
        <label for="focus_value">Valor do Foco:</label>
        <input type="number" id="focus_value" name="focus_value" min="0" max="100" required>
        <button onclick="setFocus()">Definir Foco</button>
        <p id="status"></p>
    </body>
    </html>
    """

@app.route('/set_focus', methods=['POST'])
def set_focus():
    data = request.get_json()
    focus_value = data.get('focus_value')
    if focus_value:
        try:
            # Faz a requisição para ajustar o foco na câmera
            print("Requisição iniciada")
            requests.get(f'http://raspberrypi00.local:7123/focus.html/{focus_value}')
            print("Requisição finalizada")
            return "Foco ajustado com sucesso!"
        except requests.RequestException as e:
            return f"Erro ao ajustar o foco: {e}"
    return "Nenhum valor de foco fornecido."

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
