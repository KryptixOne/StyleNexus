from flask import Flask, render_template

app = Flask(__name__)
# Configuring static file route
app.static_folder = 'static'
@app.route('/')
def homepage():
    return render_template('index.html')
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/examples')
def examples():
    return render_template('examples.html')



if __name__ == '__main__':
    # to run:  python ./webApp_Flask/webApp_flask.py in cmd prompt
    app.run()
