from flask import Flask
from src.logger import logging

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    logging.info('we are testing our second method of logging')
    return "welcome to end to end machine learning project pipeline session"

if __name__=="__main__":
    app.run(debug=True)