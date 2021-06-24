import importlib
import os

print("Checking if HPA_Cell_Segmentation is downloaded...", end="")
if importlib.util.find_spec("HPA_Cell_Segmentation.hpacellseg") is None:
    from git import Repo

    print("Downloading HPA-Cell-Segmentation from GitHub...", end="")
    Repo.clone_from("https://github.com/CellProfiling/HPA-Cell-Segmentation.git", "HPA_Cell_Segmentation")
    os.system(os.path.join("HPA_Cell_Segmentation", "hpacellseg", 'script.sh'))
    print("done !")
else:
    print("already downloaded. Proceeding.")
if importlib.util.find_spec("hpacellseg") is None:
    print("Installing HPA-Cell-Segmentation...", end="")
    os.system(os.path.join("HPA_Cell_Segmentation", "hpacellseg", 'script.sh'))
    print("done !")

from base64 import b64encode
import io
from PIL import Image

from flask import Flask, render_template, request
import imageio

from prediction import classifier
from tools_app import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["FLASK_APP"] = "app.py"

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = os.path.join("static", "images")


@app.route("/")
def use():
    return render_template("upload_image.html")


@app.route("/uploader", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        image_bleue = request.files["file1"]
        image_verte = request.files["file2"]
        image_rouge = request.files["file3"]
        image_jaune = request.files["file4"]

        voids = tuple(map(lambda x: x.filename == "", (image_bleue, image_verte, image_rouge, image_jaune)))

        if sum(voids) > 0:
            comments1 = " image bleue" + ("," if sum(voids[1:]) > 0 else ".") if voids[0] else ""
            comments2 = " image verte" + ("," if sum(voids[2:]) > 0 else ".") if voids[1] else ""
            comments3 = " image rouge" + ("," if sum(voids[3:]) > 0 else ".") if voids[2] else ""
            comments4 = " image jaune." if voids[3] else ""

            return render_template(
                "response.html",
                comments="Veuillez sélectionner toutes les images pour la prédiction."\
                         f" Image{'s' if sum(voids) > 1 else ''} manquante{'s' if sum(voids) > 1 else ''} : ",
                comments1=comments1,
                comments2=comments2,
                comments3=comments3,
                comments4=comments4,
            )
        else:
            print("Reading image...", end="")
            green_channel = imageio.imread(image_verte.read())
            red_channel = imageio.imread(image_rouge.read())
            blue_channel = imageio.imread(image_bleue.read())
            yellow_channel = imageio.imread(image_jaune.read())
            print("done.")

            # Generation of colored image for the vizualisation
            print("Coloring images...", end="")
            zero_channel = np.zeros_like(blue_channel, dtype=np.uint8)
            colored_red = np.stack([red_channel, zero_channel, zero_channel], -1).astype(np.uint8)
            colored_green = np.stack([zero_channel, green_channel, zero_channel], -1).astype(np.uint8)
            colored_blue = np.stack([zero_channel, zero_channel, blue_channel], -1).astype(np.uint8)
            colored_yellow = np.stack([yellow_channel, yellow_channel, zero_channel], -1).astype(np.uint8)
            print("done.")

            red_array, green_array, blue_array, yellow_array = \
                map(np.array, (red_channel, green_channel, blue_channel, yellow_channel))

            print("Segmenting image by cell...", end="")
            nucl_mask, cell_mask = segmentCell([[red_array], [yellow_array], [blue_array]])
            print("done.")

            print("Generating composite images...", end="")
            composite_mask = get_composite_mask(green_array, cell_mask, nucl_mask)
            composite = np.dstack((red_channel, yellow_channel, blue_channel))

            print("done.")

            # prediction with organelle_classifier
            print("Predicting organelles...", end="")
            prediction = classifier.predict(np.array(green_channel), cell_mask, nucl_mask)
            print("done.")
            print("Predictions =", prediction[2])
            print(f"Choix de la réponse: {prediction[0]} = {prediction[1]*100} %")

            r, g, b, y, c, c_m = io.BytesIO(), io.BytesIO(), io.BytesIO(), io.BytesIO(), io.BytesIO(), io.BytesIO()

            print("Saving Images to buffer...", end="")
            Image.fromarray(check_and_convert_to_8_bits(colored_red), "RGB").save(r, "JPEG")
            Image.fromarray(check_and_convert_to_8_bits(colored_green), "RGB").save(g, "JPEG")
            Image.fromarray(check_and_convert_to_8_bits(colored_blue), "RGB").save(b, "JPEG")
            Image.fromarray(check_and_convert_to_8_bits(colored_yellow), "RGB").save(y, "JPEG")
            Image.fromarray(composite.astype(np.uint8), "RGB").save(c, "JPEG")
            Image.fromarray(composite_mask.astype(np.uint8), "RGB").save(c_m, "JPEG")
            print("done.")

            return render_template(
                "result.html",
                prediction=prediction[0],
                confidence=f"{(prediction[1] * 100):.2f}",
                results=prediction[2],
                image_bleue=b64encode(b.getvalue()).decode("utf-8"),
                image_verte=b64encode(g.getvalue()).decode("utf-8"),
                image_rouge=b64encode(r.getvalue()).decode("utf-8"),
                image_jaune=b64encode(y.getvalue()).decode("utf-8"),
                composite=b64encode(c.getvalue()).decode("utf-8"),
                composite_mask=b64encode(c_m.getvalue()).decode("utf-8")
            )


if __name__ == "__main__":
    app.run(port=5000, debug=True)
