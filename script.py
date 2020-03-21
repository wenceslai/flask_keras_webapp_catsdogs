from flask import Flask, render_template, url_for, request, redirect
import os
from keras.models import load_model, model_from_json
from PIL import Image
import json
import tensorflow as tf
import numpy as np

app = Flask(__name__)

app.config['IMAGE_UPLOADS'] = 'static/img' #img dir
important_imgs = ["simplenn.png", "visualizing_cnn_activations.png"] #images that are used in templates

def loading_model(): 
	json_file = open('model_cats_dogs_VGG16.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	#load woeights into new model
	loaded_model.load_weights("model_cats_dogs_VGG16.h5")
	print("Loaded Model from disk")

	#compile and evaluate loaded model
	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	#loss,accuracy = model.evaluate(X_test,y_test)
	#print('loss:', loss)
	#print('accuracy:', accuracy)
	graph = tf.get_default_graph()

	return loaded_model,graph


model, graph = loading_model()

@app.route("/", methods=['GET', 'POST'])
def home():
    for file in os.listdir(app.config['IMAGE_UPLOADS']): #deleting images loaded from user before except from important ones, dont know how to display withou saving the image

        if file in important_imgs: continue

        file_path = os.path.join(app.config['IMAGE_UPLOADS'], file)

        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    return render_template("index.html")

    
@app.route("/image_upload", methods=['GET', 'POST'])
def image_upload():

     if request.method == 'POST':
        if request.files:

            image = request.files["image"]  #getting image from form
            newpath = os.path.join(app.config['IMAGE_UPLOADS'], image.filename)
            image.save(newpath)
         
            x = Image.open(newpath) #preprocessing the template
            x = x.resize((200, 200))
            x = np.asarray(x)
            try:        
                x = np.reshape(x, [1, 200, 200, 3])
            except:
                return render_template("error.html") #some image cannot be reshaped to this shape by numpy
                
            with graph.as_default():
                #perform the prediction
                prediction = model.predict_classes(x)
                #convert the response to a string
                #response = np.array_str(np.argmax(prediction,axis=1))
                response = np.array_str(prediction)
           
            return render_template("index.html", user_image=os.path.join('img',image.filename), model_prediction=response) #giving feedback to the user


@app.route("/about_project")
def about_project():
    return render_template('about_project.html')

        
@app.route("/about_cnn")
def about_cnn():
    return render_template('about_cnn.html')


if __name__ == "__main__":
    app.run()