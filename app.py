from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def home():
    """
    Serve a webpage with JavaScript to access webcam and send frames via WebSocket.
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Webcam Stream</title>
    </head>
    <body>
        <h1>Webcam Stream</h1>
        <video id="video" autoplay playsinline></video>
        <canvas id="canvas" style="display: none;"></canvas>
        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');

            // Access webcam
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    video.srcObject = stream;
                })
                .catch(err => {
                    console.error("Error accessing webcam:", err);
                });

            // WebSocket connection
            const ws = new WebSocket("ws://localhost:8000/ws");

            ws.onopen = () => {
                console.log("WebSocket connection established");
                setInterval(() => {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                    // Send the frame as a Blob
                    canvas.toBlob(blob => {
                        if (blob) {
                            ws.send(blob);
                        }
                    }, "image/jpeg");
                }, 100); // Send every 100ms
            };

            ws.onmessage = event => {
                console.log("Server response:", event.data);
            };

            ws.onclose = () => {
                console.log("WebSocket connection closed");
            };
        </script>
    </body>
    </html>
    """

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Receive webcam frames from the client and process them.
    """
    await websocket.accept()
    try:
        while True:
            # Receive binary frame (image data)
            frame_data = await websocket.receive_bytes()

            # Decode and process frame
            np_frame = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

            if frame is not None:
                # Perform inference or any processing (dummy example: add text)
                cv2.putText(frame, "Processed by Server", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Encode processed frame back to JPEG
                _, buffer = cv2.imencode(".jpg", frame)

                # Send the processed frame back to client
                await websocket.send_bytes(buffer.tobytes())
    except WebSocketDisconnect:
        print("WebSocket disconnected")
