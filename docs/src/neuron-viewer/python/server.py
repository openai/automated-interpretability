# %%
import logging

from flask import Flask, request
from flask_cors import CORS

import json

import urllib.request

def load_az_json(url):
    with urllib.request.urlopen(url) as f:
        return json.load(f)

def start(
    dev: bool = False,
    host_name: str = "0.0.0.0",
    port: int = 80,
):
    app = Flask("interpretability chat")
    app.logger.setLevel(logging.INFO)
    # app.logger.disabled = True
    CORS(app)

    @app.after_request
    def after_request(response):
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add(
            "Access-Control-Allow-Headers", "Content-Type,Authorization"
        )
        response.headers.add(
            "Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS"
        )
        return response

    @app.route("/load_az", methods=["GET", "POST"])
    async def load_az():
        args = request.get_json()
        path = args["path"]
        result = load_az_json(path)
        return result

    app.run(debug=dev, host=host_name, port=port, use_reloader=False)


def main(dev: bool = True, host_name: str = "0.0.0.0", port: int = 8000):
    start(dev=dev, host_name=host_name, port=port)


if __name__ == "__main__":
    main()
