from flask import Flask, render_template, url_for, request, redirect

app = Flask(__name__)

# global variables
data = {}

@app.route("/")
def index():
    global data
    data = {}
    return render_template("index.html")


@app.route("/generate")
def generate():
    global data
    data = {}
    return render_template("generate.html")


@app.route("/getKeyWords", methods=['POST'])
def getKeyWords():    
    key1 = request.form.get('key1')
    key2 = request.form.get('key2')
    key3 = request.form.get('key3')
    key4 = request.form.get('key4')
    key5 = request.form.get('key5')
    numParas = request.form.get('numParas')
    
    global data

    data['key1'] = key1
    data['key2'] = key2
    data['key3'] = key3
    data['key4'] = key4
    data['key5'] = key5

    data['numParas'] = [str(i) for i in range(1, int(numParas) + 1)]

    return redirect(url_for('paragraphs'))


@app.route("/paragraphs")
def paragraphs():

    global data

    return render_template("paragraphs.html",data=data)


@app.route("/getFeedback", methods=['POST'])
def getFeedback():
    global data
    feedback = {}
    coherency = 0
    relevance = 0
    grammar = 0

    for i in range(1, len(data['numParas']) + 1):
        coherency = request.form.get('p'+str(i)+'-rate_coherency')
        relevance = request.form.get('p'+str(i)+'-rate_relevance')
        grammar = request.form.get('p'+str(i)+'-rate_grammar')
        feedback['para'+str(i)] = {'c': (0 if coherency == None else int(coherency) ),'r': (0 if relevance == None else int(relevance) ),'g': (0 if grammar == None else int(grammar) )}

    print(feedback)
    print(data)

    return redirect(url_for('generate'))


if __name__ == '__main__':
   app.run(debug=True)