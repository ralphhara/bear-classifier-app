import gradio as gr
from fastai.vision.all import *
import skimage

learn = load_learner('export.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Bear Classifier"
description = "This bear classifier was trained on internet photos of grizzly, black and teddy bears with fastai. Created as a demo for Gradio and HuggingFace Spaces."
article="<p style='text-align: center'><a href='https://course.fast.ai/Lessons/lesson2.html' target='_blank'>Blog post</a></p>"
examples = ['teddy.jpg', 'grizzly.jpg', 'black.jpg']
interpretation='default'
enable_queue=True

gr.Interface(
    fn=predict,inputs=gr.inputs.Image(shape=(512, 512)),
    outputs=gr.outputs.Label(num_top_classes=3),
    title=title,
    description=description,
    article=article,
    examples=examples,
    interpretation=interpretation,
    enable_queue=enable_queue
    ).launch(share=True)