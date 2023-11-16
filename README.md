# face_recognition

Face recognition system using python, insightface and redis

In this project you can register multiple person and define their role and use your webcam to detect faces and recognize them

### **Libraries and dependencies**
- insightface
- onnxruntime
- pandas
- numpy
- redis
- opencv
- *** C++ compiler and docker should be installed on your system

### **How to use**
1. docker-compose up (to run redis server)
2. python face_rec.py (run project)
3. after running project press 1 to register new person (webcam should be accessible)
4. press 2 to recognize faces and show their name and role (can be used to recognize multiple person at the same time)

![face_rec](https://github.com/mohaAmiri/face_recognition/assets/111754905/681f70ca-4883-4341-93cb-61f7a865232a)
