from flask import Flask, render_template_string, request, jsonify, Response
import cv2
import os
import numpy as np
import datetime
import mysql.connector
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import date

app = Flask(__name__)

camera_active = False

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="online_attendance"
)
cursor = conn.cursor()

cursor.execute("""
               CREATE TABLE IF NOT EXISTS users
               (rollno INT PRIMARY KEY,name VARCHAR(100),email VARCHAR(200),password VARCHAR(100),date DATE)""")
conn.commit()

cursor.execute("""
               CREATE TABLE IF NOT EXISTS atten
               (rollno INT, name VARCHAR(100),date DATE)""")
conn.commit()

cursor.execute("""
               CREATE TABLE IF NOT EXISTS admins
               (  username VARCHAR(100)PRIMARY KEY, password VARCHAR(100),email VARCHAR(200))""")
conn.commit()


def training():
    """Train the face recognition model with all dataset images"""
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        faces = []
        labels = []
        label_dict = {}
        label_id = 0
        dataset_path = "dataset"

        print("\n🔄 Starting training process...")

        if not os.path.exists(dataset_path):
            print("❌ Dataset folder not found!")
            return False

        rollno_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

        if not rollno_folders:
            print("❌ No student folders found in dataset!")
            return False

        print(f"✅ Found {len(rollno_folders)} students")

        for person_name in rollno_folders:
            person_path = os.path.join(dataset_path, person_name)
            label_dict[label_id] = int(person_name)

            print(f"📸 Processing Roll No: {person_name} (Label ID: {label_id})")

            image_count = 0
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is not None:
                    faces.append(img)
                    labels.append(label_id)
                    image_count += 1

            print(f"   ✓ Loaded {image_count} images")
            label_id += 1

        print(f"\n🎯 Total images: {len(faces)}")
        print("🔄 Training model... Please wait...")

        recognizer.train(faces, np.array(labels))
        recognizer.save("lbph_trained.yml")
        print("✅ Model saved: lbph_trained.yml")

        with open("labels.pkl", "wb") as f:
            pickle.dump(label_dict, f)
        print("✅ Labels saved: labels.pkl")

        print("\n📋 Label Mapping:")
        print("=" * 30)
        for lid, rollno in label_dict.items():
            print(f"Label ID {lid} → Roll No {rollno}")
        print("=" * 30)
        print("\n✅ Training completed successfully!\n")
        return True

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False



def send_email(receiver_email, name):
    sender = ""
    password = ""

    subject = "Attendance Marked"
    body = f"Hello {name},\nYour attendance is marked on {datetime.date.today()}.\nRegards,\nAttendance System"

    message = f"Subject: {subject}\n\n{body}"

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver_email, message)
        server.quit()
    except Exception as e:
        print(f"Email sending failed: {e}")

def take_attendance(rollno):
    today = datetime.date.today()

    cursor.execute("SELECT * FROM atten WHERE rollno=%s AND date=%s", (rollno, today))
    if cursor.fetchone():
        print(f"⚠️ Roll No {rollno} already marked.")
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Attendance Status</title>
        </head>
        <body>
            <h1>⚠️ Roll No {rollno} already marked today!</h1>
        </body>
        </html>
        """

    cursor.execute("SELECT name, email FROM users WHERE rollno=%s", (rollno,))
    user = cursor.fetchone()

    if not user:
        print(f"⚠️ Roll No {rollno} NOT FOUND in users table.")
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Attendance Status</title>
        </head>
        <body>
            <h1>⚠️ Roll No {rollno} NOT FOUND in database!</h1>
        </body>
        </html>
        """

    name, email = user

    cursor.execute("INSERT INTO atten (rollno, name, date) VALUES (%s, %s, %s)",
                   (rollno, name, today))
    conn.commit()

    print(f"✅ Attendance marked for Roll No {rollno} ({name})")

    if email:
        send_email(email, name)
    else:
        print("⚠️ No email stored for this rollno.")

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Attendance Confirmation</title>
    </head>
    <body>
        <h1>✅ Attendance marked for Roll No {rollno} ({name})</h1>
        <p>Date: {today}</p>
    </body>
    </html>
    """
def generate_frames():
    global camera_active

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("lbph_trained.yml")

        with open("labels.pkl", "rb") as f:
            label_dict = pickle.load(f)
    except Exception as e:
        print(f"Error loading model files: {e}")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    last_roll = None
    detection_start = None

    while camera_active:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        current_rollno = None

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            label_id, confidence = recognizer.predict(roi_gray)

            if confidence < 70:
                current_rollno = label_dict.get(label_id)
            else:
                current_rollno = None

            if current_rollno != last_roll and current_rollno is not None:
                detection_start = time.time()
                last_roll = current_rollno

            if current_rollno is not None and detection_start:
                elapsed = time.time() - detection_start
                remaining = int(7 - elapsed)

                if elapsed >= 7:
                    take_attendance(current_rollno)
                    detection_start = None
                else:
                    cv2.putText(frame, f"Marking in {remaining}s", (x, y - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            display_text = f"Roll: {current_rollno}" if current_rollno else "Unknown"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, display_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Attendance System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }

        h1 {
            text-align: center;
            color: white;
            font-size: 3em;
            margin: 30px 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            letter-spacing: 2px;
        }

        h2 {
            color: #667eea;
            margin: 30px 0 20px;
            font-size: 2em;
            text-align: center;
        }

        h3 {
            color: #555;
            margin: 20px 0;
            font-size: 1.5em;
            text-align: center;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }

        #mainButtons {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 50px 0;
            flex-wrap: wrap;
        }

        button {
            padding: 15px 40px;
            font-size: 18px;
            cursor: pointer;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 50px;
            transition: all 0.3s ease;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }

        button:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }

        button:active {
            transform: translateY(-1px);
        }

        button.delete-btn {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
        }

        button.delete-btn:hover {
            box-shadow: 0 6px 20px rgba(245, 87, 108, 0.6);
        }

        button.train-btn {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
        }

        button.train-btn:hover {
            box-shadow: 0 6px 20px rgba(79, 172, 254, 0.6);
        }

        input {
            padding: 15px 20px;
            margin: 10px 0;
            width: 100%;
            max-width: 400px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            transition: all 0.3s ease;
        }

        input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 10px rgba(102, 126, 234, 0.2);
        }

        form {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 30px 0;
        }

        #adminSection, #studentSection {
            animation: fadeIn 0.5s ease;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        table {
            border-collapse: collapse;
            width: 100%;
            margin: 30px 0;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        th, td {
            padding: 15px;
            text-align: center;
            border-bottom: 1px solid #f0f0f0;
        }

        th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        tr:hover {
            background-color: #f8f9ff;
        }

        tr:last-child td {
            border-bottom: none;
        }

        .delete-link {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 25px;
            display: inline-block;
            transition: all 0.3s ease;
            font-weight: 600;
        }

        .delete-link:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
        }

        #videoFeed {
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background: #f8f9ff;
            border-radius: 15px;
        }

        #videoStream {
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            max-width: 100%;
            height: auto;
        }

        .back-btn {
            background: linear-gradient(135deg, #a8a8a8 0%, #6c6c6c 100%);
            margin-top: 30px;
        }

        .message {
            text-align: center;
            padding: 15px;
            margin: 20px 0;
            border-radius: 10px;
            font-weight: 600;
        }

        .success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }

        .error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }

        .button-group {
            display: flex;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
            margin: 20px 0;
        }

        #stopCameraBtn {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        }

        #collectBtn {
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        }

        .section-card {
            background: #f8f9ff;
            padding: 30px;
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            h1 {
                font-size: 2em;
            }

            .container {
                padding: 20px;
            }

            button {
                padding: 12px 30px;
                font-size: 16px;
            }

            #mainButtons {
                flex-direction: column;
                align-items: center;
            }

            table {
                font-size: 14px;
            }

            th, td {
                padding: 10px;
            }
        }

        /* Loading Animation */
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Status Messages */
        #status1, #message, #adminMessage {
            margin: 20px 0;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-weight: 600;
        }
    </style>
</head>
<body>

<h1>WELCOME</h1>
<h1>🎓AI-POWERED AUTOMATIC ATTENDANCE SYSTEM USING DEEP LEARNING</h1>

<div class="container">
    <div id="mainButtons">
        <button onclick="showAdmin()">👨‍💼 ADMIN</button>
        <button onclick="showStudent()">👨‍🎓 STUDENT</button>
    </div>

    <div id="adminSection" style="display:none;">
        <h2>ADMIN PANEL</h2>
        

        <div id="adminAuthButtons" class="button-group">
            <button onclick="ShowAdminLogin()">🔑 LOG IN</button>
            <button onclick="ShowAdminSignup()">📝 SIGN UP</button>
        </div>

        <div id="adminSignupDiv" class="section-card" style="display:none;">
            <h3>Admin Sign Up</h3>
            <form onsubmit="event.preventDefault(); adminSignup()">
                <input type="text" id="adminUsername" placeholder="Username" required>
                <input type="password" id="adminPassword" placeholder="Password" required>
                <input type="email" id="adminEmail" placeholder="Email" required>
                <button type="submit">Submit</button>
            </form>
        </div>

        <div id="adminLoginDiv" class="section-card" style="display:none;">
            <h3>Admin Login</h3>
            <form onsubmit="event.preventDefault(); adminLogin()">
                <input type="text" id="adminLoginUsername" placeholder="Username" required>
                <input type="password" id="adminLoginPassword" placeholder="Password" required>
                <button type="submit">Login</button>
            </form>
            <p id="adminMessage"></p>
        </div>

        <div id="adminDashboard" style="display:none;">
            <h3>Welcome Admin! 👋</h3>
            <div class="button-group">
                <button onclick="startRecognition()">📸 TAKE ATTENDANCE</button>
                <button id="stopCameraBtn" onclick="stopRecognition()" style="display:none;">⏹️ STOP CAMERA</button>
                <button class="delete-btn" onclick="showDeleteUsers()">🗑️ DELETE USERS</button>
            </div>

            <div id="videoFeed" style="display:none;">
                <img id="videoStream" src="" width="640" height="480">
            </div>

            <div id="deleteUsersSection" class="section-card" style="display:none;">
                <h3>Registered Users</h3>
                <button onclick="loadUsers()">🔄 Refresh List</button>
                <div id="usersTableDiv"></div>
            </div>

            <div class="button-group">
                <button onclick="adminLogout()">🚪 Logout</button>
            </div>
        </div>

        <button class="back-btn" onclick="goBack()">⬅️ Back to Main</button>
    </div>

    <div id="studentSection" style="display:none;">
        <h2>WELCOME STUDENT</h2>
        <div id="studentAuthButtons" class="button-group">
            <button onclick="ShowLogin()">🔑 LOG IN</button>
            <button onclick="ShowForm()">📝 SIGN UP</button>
        </div>

        <div id="formdiv" class="section-card" style="display:none;">
            <h3>Sign Up</h3>
            <form onsubmit="event.preventDefault(); savePassword()">
                <input type="text" id="setNameInput" placeholder="Name" required>
                <input type="text" id="setRollInput" placeholder="Roll No" required>
                <input type="password" id="setPasswordInput" placeholder="Password" required>
                <input type="email" id="setEmailInput" placeholder="Email" required>
                <button type="submit">Submit</button>
            </form>
        </div>

        <div id="logindiv" class="section-card" style="display:none;">
            <h3>Login</h3>
            <form onsubmit="event.preventDefault(); checkPassword()">
                <input type="text" id="loginNameInput" placeholder="Name" required>
                <input type="password" id="loginPasswordInput" placeholder="Password" required>
                <button type="submit">Login</button>
            </form>
            <p id="message"></p>
        </div>

        <div id="studentDashboard" style="display:none;">
            <h3>Welcome Student! 👋</h3>
            <div class="button-group">
                <button id="collectBtn" onclick="showForm1()">📷 Collect Dataset</button>
                <button onclick="showAttendancePercentage()">📊 Attendance Percentage</button>
            </div>

            <div id="formDiv1" class="section-card" style="display:none;">
                <h3>Dataset Collection</h3>
                <form onsubmit="submitForm1(event)">
                    <input type="text" id="nameInput1" placeholder="Enter your name" required>
                    <button type="submit">Start</button>
                </form>
                <p id="status1"></p>
            </div>

            <div class="button-group">
                <button onclick="studentLogout()">🚪 Logout</button>
            </div>
        </div>

        <button class="back-btn" onclick="goBack()">⬅️ Back to Main</button>
    </div>
</div>

<script>
function showAdmin() {
    document.getElementById('mainButtons').style.display = 'none';
    document.getElementById('adminSection').style.display = 'block';
    document.getElementById('studentSection').style.display = 'none';
}

function showStudent() {
    document.getElementById('mainButtons').style.display = 'none';
    document.getElementById('adminSection').style.display = 'none';
    document.getElementById('studentSection').style.display = 'block';
}

function goBack() {
    document.getElementById('mainButtons').style.display = 'flex';
    document.getElementById('adminSection').style.display = 'none';
    document.getElementById('studentSection').style.display = 'none';
    document.getElementById('formdiv').style.display = 'none';
    document.getElementById('logindiv').style.display = 'none';
    document.getElementById('studentDashboard').style.display = 'none';
    document.getElementById('studentAuthButtons').style.display = 'flex';
    document.getElementById('formDiv1').style.display = 'none';
    document.getElementById('adminSignupDiv').style.display = 'none';
    document.getElementById('adminLoginDiv').style.display = 'none';
    document.getElementById('adminAuthButtons').style.display = 'flex';
    document.getElementById('adminDashboard').style.display = 'none';
    document.getElementById('deleteUsersSection').style.display = 'none';
}

function ShowAdminSignup() {
    document.getElementById('adminSignupDiv').style.display = 'block';
    document.getElementById('adminLoginDiv').style.display = 'none';
    document.getElementById('adminAuthButtons').style.display = 'none';
}

function ShowAdminLogin() {
    document.getElementById('adminLoginDiv').style.display = 'block';
    document.getElementById('adminSignupDiv').style.display = 'none';
    document.getElementById('adminAuthButtons').style.display = 'none';
}

function adminSignup() {
    const username = document.getElementById('adminUsername').value.trim();
    const password = document.getElementById('adminPassword').value.trim();
    const email = document.getElementById('adminEmail').value.trim();

    fetch('/admin/signup', {
        method: 'POST',
        headers: {'Content-Type':'application/x-www-form-urlencoded'},
        body:`username=${username}&password=${password}&email=${email}`
    }).then(r=>r.json()).then(data=>{
        if(data.success){
            alert("✅ Admin signup successful! Please login.");
            ShowAdminLogin();
        }else{
            alert("❌ " + data.message);
        }
    })
}

function adminLogin() {
    const username = document.getElementById('adminLoginUsername').value.trim();
    const password = document.getElementById('adminLoginPassword').value.trim();

    fetch('/admin/login', {
        method:"POST",
        headers:{'Content-Type':'application/x-www-form-urlencoded'},
        body:`username=${username}&password=${password}`
    }).then(r=>r.json()).then(data=>{
        if(data.success){
            document.getElementById('adminMessage').innerHTML = "✅ Login Successful";
            document.getElementById('adminMessage').className = "message success";
            document.getElementById('adminLoginDiv').style.display = 'none';
            document.getElementById('adminAuthButtons').style.display = 'none';
            document.getElementById('adminDashboard').style.display = 'block';
        } else {
            document.getElementById('adminMessage').innerHTML = "❌ Invalid Credentials";
            document.getElementById('adminMessage').className = "message error";
        }
    })
}

function adminLogout() {
    document.getElementById('adminDashboard').style.display = 'none';
    document.getElementById('adminLoginDiv').style.display = 'none';
    document.getElementById('adminSignupDiv').style.display = 'none';
    document.getElementById('adminAuthButtons').style.display = 'flex';
    document.getElementById('deleteUsersSection').style.display = 'none';
    stopRecognition();
}

function startRecognition() {
    document.getElementById('deleteUsersSection').style.display = 'none';
    fetch('/start_camera')
    .then(r=>r.json())
    .then(data=>{
        if(data.success){
            document.getElementById('videoFeed').style.display = 'block';
            document.getElementById('videoStream').src = '/video_feed';
            document.getElementById('stopCameraBtn').style.display = 'inline-block';
        } else {
            alert('❌ ' + (data.message || 'Failed to start camera. Train model first!'));
        }
    })
}

function stopRecognition() {
    fetch('/stop_camera')
    .then(r=>r.json())
    .then(data=>{
        document.getElementById('videoFeed').style.display = 'none';
        document.getElementById('videoStream').src = '';
        document.getElementById('stopCameraBtn').style.display = 'none';
    })
}

function showDeleteUsers() {
    document.getElementById('videoFeed').style.display = 'none';
    document.getElementById('videoStream').src = '';
    document.getElementById('stopCameraBtn').style.display = 'none';
    document.getElementById('deleteUsersSection').style.display = 'block';
    loadUsers();
}

function loadUsers() {
    fetch('/get_users')
    .then(r=>r.json())
    .then(data=>{
        if(data.users && data.users.length > 0) {
            let html = '<table><tr><th>Roll No</th><th>Name</th><th>Email</th><th>Action</th></tr>';
            data.users.forEach(user => {
                html += `<tr>
                    <td>${user.rollno}</td>
                    <td>${user.name}</td>
                    <td>${user.email}</td>
                    <td><a class="delete-link" href="#" onclick="deleteUser(${user.rollno}); return false;">Delete</a></td>
                </tr>`;
            });
            html += '</table>';
            document.getElementById('usersTableDiv').innerHTML = html;
        } else {
            document.getElementById('usersTableDiv').innerHTML = '<p class="message">No users found.</p>';
        }
    })
}

function deleteUser(rollno) {
    if(confirm('⚠️ Delete Roll No ' + rollno + '? This will also retrain the model.')) {
        fetch('/delete_user/' + rollno, {method: 'POST'})
        .then(r=>r.json())
        .then(data=>{
            if(data.success) {
                alert('✅ User deleted & model retrained successfully!');
                loadUsers();
            } else {
                alert('❌ Error: ' + data.message);
            }
        })
    }
}

function ShowForm() {
    document.getElementById('formdiv').style.display = 'block';
    document.getElementById('logindiv').style.display = 'none';
    document.getElementById('studentAuthButtons').style.display = 'none';
}

function ShowLogin() {
    document.getElementById('logindiv').style.display = 'block';
    document.getElementById('formdiv').style.display = 'none';
    document.getElementById('studentAuthButtons').style.display = 'none';
}

function savePassword() {
    const name = setNameInput.value.trim();
    const roll = setRollInput.value.trim();
    const pass = setPasswordInput.value.trim();
    const email = setEmailInput.value.trim();

    fetch('/signup', {
        method: 'POST',
        headers: {'Content-Type':'application/x-www-form-urlencoded'},
        body:`name=${name}&rollno=${roll}&password=${pass}&email=${email}`
    }).then(r=>r.json()).then(data=>{
        if(data.success){
            alert("✅ Signup successful! Please login.");
            ShowLogin();
        }else{
            alert("❌ " + data.message);
        }
    })
}

function checkPassword() {
    const name = loginNameInput.value.trim();
    const password = loginPasswordInput.value.trim();

    fetch('/login', {
        method:"POST",
        headers:{'Content-Type':'application/x-www-form-urlencoded'},
        body:`name=${name}&password=${password}`
    }).then(r=>r.json()).then(data=>{
        if(data.success){
            message.innerHTML = "✅ Login Successful";
            message.className = "message success";
            logindiv.style.display = "none";
            document.getElementById('studentAuthButtons').style.display = 'none';
            document.getElementById('studentDashboard').style.display = 'block';
        } else {
            message.innerHTML = "❌ Invalid Credentials";
            message.className = "message error";
        }
    })
}

function studentLogout() {
    document.getElementById('studentDashboard').style.display = 'none';
    document.getElementById('studentAuthButtons').style.display = 'flex';
    document.getElementById('formDiv1').style.display = 'none';
}

function showAttendancePercentage() {
    alert('📊 Attendance Percentage feature - Backend integration pending');
}

function showForm1() {
    formDiv1.style.display = "block";
}

function submitForm1(e){
    e.preventDefault();
    let name = nameInput1.value.trim();
    status1.innerText = "📸 Collecting images...";
    status1.className = "message";

    fetch('/submit', {
        method:"POST",
        headers:{'Content-Type':'application/x-www-form-urlencoded'},
        body:"name="+name
    }).then(r=>r.json()).then(data=>{
        if(data.success) {
            status1.innerText = "🎯 Dataset collected! Training model...";
            fetch('/train_model', {method: 'POST'})
            .then(r=>r.json())
            .then(trainData=>{
                if(trainData.success) {
                    status1.innerText = "✅ Dataset collected & model trained!";
                    status1.className = "message success";
                } else {
                    status1.innerText = "⚠️ Dataset collected but training failed. Ask admin to train.";
                    status1.className = "message error";
                }
            })
        } else {
            status1.innerText = "❌ " + data.message;
            status1.className = "message error";
        }
    })
}
</script>

</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(html_template)


@app.route('/signup', methods=['POST'])
def signup():
    name = request.form.get('name').strip()
    rollno = request.form.get('rollno').strip()
    password = request.form.get('password')
    email = request.form.get('email')
    today = datetime.date.today()

    try:
        cursor.execute("INSERT INTO users (rollno, name, email, password, date) VALUES (%s, %s, %s, %s, %s)",
                       (int(rollno), name, email, password, today))
        conn.commit()
        return jsonify({"success": True})
    except mysql.connector.IntegrityError:
        return jsonify({"success": False, "message": "Roll No already exists!"})


@app.route('/login', methods=['POST'])
def login():
    name = request.form.get('name')
    password = request.form.get('password')
    cursor.execute("SELECT * FROM users WHERE name=%s AND password=%s", (name, password))
    return jsonify({"success": bool(cursor.fetchone())})


@app.route('/admin/signup', methods=['POST'])
def admin_signup():
    username = request.form.get('username').strip()
    password = request.form.get('password')
    email = request.form.get('email')

    try:
        cursor.execute("INSERT INTO admins (username, password, email) VALUES (%s, %s, %s)",
                       (username, password, email))
        conn.commit()
        return jsonify({"success": True})
    except mysql.connector.IntegrityError:
        return jsonify({"success": False, "message": "Username already exists!"})


@app.route('/admin/login', methods=['POST'])
def admin_login():
    username = request.form.get('username')
    password = request.form.get('password')
    cursor.execute("SELECT * FROM admins WHERE username=%s AND password=%s", (username, password))
    return jsonify({"success": bool(cursor.fetchone())})


@app.route('/train_model', methods=['POST'])
def train_model_route():
    try:
        success = training()
        if success:
            return jsonify({"success": True, "message": "Model trained successfully!"})
        else:
            return jsonify({"success": False, "message": "Training failed. Check console."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route('/get_users')
def get_users():
    cursor.execute("SELECT rollno, name, email FROM users")
    users = cursor.fetchall()
    users_list = [{"rollno": u[0], "name": u[1], "email": u[2]} for u in users]
    return jsonify({"users": users_list})


@app.route('/delete_user/<int:rollno>', methods=['POST'])
def delete_user(rollno):
    try:
        dataset_folder = os.path.join("dataset", str(rollno))
        if os.path.exists(dataset_folder):
            shutil.rmtree(dataset_folder)

        cursor.execute("DELETE FROM atten WHERE rollno=%s", (rollno,))
        cursor.execute("DELETE FROM users WHERE rollno=%s", (rollno,))
        conn.commit()

        training()

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route('/start_camera')
def start_camera():
    global camera_active

    if not os.path.exists("lbph_trained.yml") or not os.path.exists("labels.pkl"):
        return jsonify({
            "success": False,
            "message": "Model not trained! Click TRAIN MODEL first."
        })

    camera_active = True
    return jsonify({"success": True})


@app.route('/stop_camera')
def stop_camera():
    global camera_active
    camera_active = False
    return jsonify({"success": True})


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/submit', methods=['POST'])
def submit():
    name = request.form.get('name').strip()

    cursor.execute("SELECT rollno FROM users WHERE name=%s", (name,))
    user = cursor.fetchone()

    if not user:
        return jsonify({"success": False, "message": "User not found!"})

    rollno = user[0]
    save_path = os.path.join("dataset", str(rollno))
    os.makedirs(save_path, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    count = 0

    while count < 100:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(save_path, f"{count}.jpg"), roi)
            count += 1

    cap.release()
    cv2.destroyAllWindows()

    return jsonify({"success": True, "message": f"Dataset collected for {name}"})


def send_low_attendance_mail(to_email, rollno, percentage):
    sender_email = ""
    sender_pass = ""

    subject = f"Low Attendance Warning - Roll No {rollno}"
    body = f"Your attendance is below 50%. Attendance: {percentage:.2f}%"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_pass)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
    except Exception as e:
        print(f"Email sending failed: {e}")


def check_attendance():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="1234",
        database="online_attendance"
    )
    cur = connection.cursor()

    cur.execute("SELECT rollno, date, email FROM users")
    students = cur.fetchall()

    today = date.today()

    for rollno, admission_date, email in students:
        cur.execute("SELECT COUNT(DISTINCT date) FROM atten WHERE rollno=%s", (rollno,))
        present = cur.fetchone()[0]

        total_days = (today - admission_date).days
        percentage = (present / total_days) * 100 if total_days > 0 else 0

        if percentage < 50:
            send_low_attendance_mail(email, rollno, percentage)

    cur.close()
    connection.close()


def schedule_thread():
    schedule.every().day.at("17:59").do(check_attendance)
    while True:
        schedule.run_pending()
        time.sleep(2)


if __name__ == '__main__':
    threading.Thread(target=schedule_thread, daemon=True).start()
    app.run(debug=True)