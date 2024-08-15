from flask import Flask
from flask import request, send_file
from flask import render_template
from PIL import Image
from flaskwebgui import FlaskUI

import io
import base64

from image_process import process

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('paint.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    data_url = request.get_json()['image']
    # Remove the data URL prefix
    data_url = data_url.split(',')[1]
    # Decode the base64 encoded string
    image_data = base64.b64decode(data_url)
    # Create an image file
    image = Image.open(io.BytesIO(image_data))
    # Remove transparency by pasting the image onto a white background
    image = image.convert("RGBA")
    white_bg = Image.new('RGBA', image.size, (255, 255, 255))
    image = Image.alpha_composite(white_bg, image)
    # Save the image to a file
    image.convert("RGB").save('static/temp/out.png')

    process('static/temp/out.png')
    return send_file('static/temp/output.jpg', mimetype='image/jpeg')


if __name__ == '__main__':
    # app.run(debug=True)
    FlaskUI(app=app, server="flask", width=800, height=600).run()