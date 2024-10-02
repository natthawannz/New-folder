from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# โหลดโมเดลที่ฝึกมาแล้วจากไฟล์ model.pkl
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # รับข้อมูลจากฟอร์ม
    age = int(request.form['age'])
    gender = int(request.form['gender'])
    education = int(request.form['education'])
    job_title = int(request.form['job_title'])
    experience = int(request.form['experience'])

    # ใส่ข้อมูลที่ได้รับมาในรูปแบบของ numpy array
    input_data = np.array([[age, gender, education, job_title, experience]])

    # ทำนายเงินเดือนโดยใช้โมเดล
    prediction = model.predict(input_data)[0]

    # ส่งผลลัพธ์กลับไปแสดงบนหน้า HTML
    return render_template('index.html', prediction_text=f'Predicted Salary: ${prediction:,.2f}')

if __name__ == '__main__':
    app.run(debug=True)
