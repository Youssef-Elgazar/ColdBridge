import json
import socket
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # suppress default access logs

    def do_GET(self):
        body = json.dumps({
            "status": "ok",
            "runtime": "python",
            "host": socket.gethostname(),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        _ = self.rfile.read(length)
        result = float(np.sum(np.random.rand(1000)))
        body = json.dumps({"result": result, "runtime": "python"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
