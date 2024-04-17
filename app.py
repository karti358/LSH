from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from recommender_algos.predictions import *

webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")
r_matrix = load_matrix()
print(r_matrix.shape)

app = Flask(__name__, static_folder=static_dir,template_folder=template_dir)

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        try:
            if request.form:
                dict_req = dict(request.form)
                response = form_response(r_matrix, dict_req)
                return render_template("index.html", response=response)
            elif request.json:
                response = api_response(r_matrix, request.json)
                return jsonify({
                    "predicted rating" : response
                    }
                )

        except Exception as e:
            print(e)
            error = {"error": "Something went wrong!! Try again later!"}
            error = {"error": e}

            return render_template("404.html", error=error)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)