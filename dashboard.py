"""Flask-based surveillance dashboard."""

from __future__ import annotations

import logging
from threading import Thread
from typing import Generator

import cv2
from flask import Flask, Response, jsonify, render_template

import config
from src.database import DatabaseManager
from src.detector import CampusSurveillance


LOGGER = logging.getLogger(__name__)

app = Flask(__name__)

config.configure_logging()

surveillance = CampusSurveillance(show_windows=False)
db = DatabaseManager()


def start_surveillance() -> None:
    LOGGER.info("Starting background surveillance thread")
    surveillance.run()


thread = Thread(target=start_surveillance, daemon=True)
thread.start()


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/alerts")
def api_alerts():
    events = [dict(row) for row in db.get_recent_events(20)]
    return jsonify({"alerts": events})


def _generate_frames() -> Generator[bytes, None, None]:
    while True:
        frame = surveillance.get_current_frame()
        if frame is None:
            continue
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(_generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/health")
def health():
    return {"status": "ok", "models": surveillance.model_manager.get_available_models()}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=config.DEBUG_MODE)

