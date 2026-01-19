##  Introduction

* The "AI-Powered Automatic Attendance System Using Deep Learning" is an automated attendance tracking application built using Python and MySQL. Instead of taking attendance manually, this system uses advanced face recognition technology to identify students in real time through a webcam or camera feed and automatically logs their attendance in the database. This saves time, eliminates errors, and ensures a reliable attendance record.

* The system provides a secure Admin Panel where administrators or teachers can manage student profiles, train the face recognition model, and review attendance records with intuitive dashboards and reports. The model is trained using the students’ face images, and every time a student appears in front of the camera, the system recognizes the face and marks them as present.

* A key enhancement in this system is the Low Attendance Email Alert Feature. The system continuously calculates attendance percentages for all students and, if a student’s attendance falls below a predefined threshold (for example 75%), it automatically sends them an email warning about their low attendance. This email alert is generated using SMTP (such as Gmail or institutional email) and contains a personalized message that notifies the student about their current attendance percentage and urges them to improve before it affects their academic standing.

* Administrators can configure the attendance threshold, schedule automatic checks, and view sent alerts within the admin dashboard. Overall, this project integrates computer vision for real-time recognition, backend logic for attendance management, and automated communication to improve student accountability, making it a comprehensive attendance solution for educational institutions.

![image-alt](https://github.com/tejendra123/AI-based-Attendance-system-/blob/d6313f7570919309394fe2a3f17310ca14aa25c8/opencv/Home%20page.jpg)
Figure 1: Home Page

---

## 🚀 Features

### 👨🎓 Student Panel

* Student **Sign Up & Login**
* **Dataset Collection** using webcam
* Email notification when attendance is marked

![image-alt](https://github.com/tejendra123/AI-based-Attendance-system-/blob/d6313f7570919309394fe2a3f17310ca14aa25c8/opencv/student.jpeg)
Figure 2: Student Panel

---

![image-alt](https://github.com/tejendra123/AI-based-Attendance-system-/blob/d6313f7570919309394fe2a3f17310ca14aa25c8/opencv/studentLogin.jpeg)
Figure 3: Student Dashboard

---

![image-alt](https://github.com/tejendra123/AI-based-Attendance-system-/blob/d6313f7570919309394fe2a3f17310ca14aa25c8/opencv/Dataset.jpeg)
Figure 4: Dataset Collection

---

### 👨‍💼 Admin Panel

* Admin **Sign Up & Login**
* **Train Face Recognition Model** using stored datasets
* **Start / Stop Camera** for live face recognition
* **Automatic Attendance Marking** after face is detected for a fixed duration
* **View & Delete Registered Users**
* Auto retraining of model after user deletion

![image-alt](https://github.com/tejendra123/AI-based-Attendance-system-/blob/d6313f7570919309394fe2a3f17310ca14aa25c8/opencv/admin.jpeg)
Figure 5: Admin Panel

---

![image-alt](https://github.com/tejendra123/AI-based-Attendance-system-/blob/d6313f7570919309394fe2a3f17310ca14aa25c8/opencv/adminLogin.jpeg)
Figure 6: Admin Dashboard

---

![image-alt](https://github.com/tejendra123/AI-based-Attendance-system-/blob/d6313f7570919309394fe2a3f17310ca14aa25c8/opencv/records.png)
Figure 7: Admin Maintain and Delete Records

---

### 🤖 Face Recognition

* Attendance is marked only after continuous face detection for 7 seconds, ensuring authenticity and preventing false entries

![image-alt](https://github.com/tejendra123/AI-based-Attendance-system-/blob/d6313f7570919309394fe2a3f17310ca14aa25c8/opencv/recognize.jpeg)
Figure 8: Taking Attendance

---

### 📧 Email Notification

* Automatically sends an email notification to the student once attendance is successfully recorded
* Helps students stay informed and provides instant confirmation of attendance
* Sends a low attendance warning email to students whose attendance percentage falls below 50%, encouraging them to maintain the required attendance level

![image-alt](https://github.com/tejendra123/AI-based-Attendance-system-/blob/d6313f7570919309394fe2a3f17310ca14aa25c8/opencv/Email-notify.jpeg)
Figure 9: Notification for Marked Attendance

---

![image-alt](https://github.com/tejendra123/AI-based-Attendance-system-/blob/d6313f7570919309394fe2a3f17310ca14aa25c8/opencv/Alert-email.jpeg)
Figure 10: Notification for Low Attendance

---

## 🧠 Working Flow

![image-alt](https://github.com/tejendra123/AI-based-Attendance-system-/blob/d6313f7570919309394fe2a3f17310ca14aa25c8/opencv/Workflow.jpeg)
Figure 11: Block diagram of Workflow

---

The flow diagram represents the working of the Face Recognition–based Attendance System. The user interacts with the system through a web browser, which communicates with the Flask web server. The server handles user requests, manages face dataset collection using OpenCV, and stores attendance data in the MySQL database. The system automatically checks attendance records and sends email notifications using SMTP when required.
