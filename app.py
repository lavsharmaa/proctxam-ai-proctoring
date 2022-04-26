from itertools import count
import os
import psycopg2
from dotenv import load_dotenv
from flask import Flask, render_template, request, abort, redirect, url_for, session, Response
from twilio.jwt.access_token import AccessToken
from twilio.jwt.access_token.grants import VideoGrant, ChatGrant
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
import datetime 
import os
import time, math
import asyncio
import cv2
import numpy as np
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks, draw_marks
from flask_session import Session
# import appscript
import datetime

load_dotenv()
twilio_account_sid = '<account_sid>'
twilio_api_key_sid = '<api_key_sid>'
twilio_api_key_secret = '<api_key_secret>'
twilio_client = Client(twilio_api_key_sid, twilio_api_key_secret,
                       twilio_account_sid)

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# azure database
# Update connection string information
host = "<host_name>"
dbname = "<db_name>"
user = "<username>"
password = "<password>"
# sslmode is required if you are using azure postgresql database
# read more on https://docs.microsoft.com/en-us/azure/postgresql/connect-python
# sslmode = "require"

# connection string
# if using azure postgresql just add sslmode={4} and sslmode in the conn_string
conn_string = "host={0} user={1} dbname={2} password={3}".format(host, user, dbname, password)

def get_chatroom(name):
    for conversation in twilio_client.conversations.conversations.stream():
        if conversation.friendly_name == name:
            return conversation

    # a conversation with the given name does not exist ==> create a new one
    return twilio_client.conversations.conversations.create(
        friendly_name=name)


@app.route('/')
def index():
    connection = psycopg2.connect(conn_string)
    cursor = connection.cursor()
    # Student table
    student_table = """CREATE TABLE IF NOT EXISTS "student"
    (
        "username" text NOT NULL,
        "firstname" text NOT NULL,
        "lastname" text NOT NULL,
        "emailid" text NOT NULL,
        "dateofbirth" date NOT NULL,
        "studentclass" text NOT NULL,
        "studentidno" integer NOT NULL,
        "rollno" integer NOT NULL,
        password text NOT NULL,
        approved boolean NOT NULL,
        "createtimestamp" date NOT NULL,
        PRIMARY KEY ("username"),
        CONSTRAINT "emailid" UNIQUE ("emailid"),
        CONSTRAINT "studentidno" UNIQUE ("studentidno"),
        CONSTRAINT "username" UNIQUE ("username")
    );
    """
    cursor.execute(student_table)
    connection.commit()

    admin_table = """CREATE TABLE IF NOT EXISTS "admin"
    (
        "username" text NOT NULL,
        "firstname" text NOT NULL,
        "lastname" text NOT NULL,
        "emailid" text NOT NULL,
        password text NOT NULL,
        "createtimeStamp" date NOT NULL,
        PRIMARY KEY ("username"),
        UNIQUE ("username"),
        UNIQUE ("emailid")
    );"""
    cursor.execute(admin_table)
    connection.commit()

    proctoring_table = """CREATE TABLE IF NOT EXISTS "proctoring"
    (
        "username" text NOT NULL,
        "emailid" text NOT NULL,
        percentage integer NOT NULL,
        count integer NOT NULL,
        "createtimeStamp" date NOT NULL,
        PRIMARY KEY ("username"),
        UNIQUE ("username"),
        UNIQUE ("emailid"),
        FOREIGN KEY ("username")
            REFERENCES "student" ("username") MATCH SIMPLE
            ON UPDATE NO ACTION
            ON DELETE NO ACTION
            NOT VALID,
        FOREIGN KEY ("emailid")
            REFERENCES "student" ("emailid") MATCH SIMPLE
            ON UPDATE NO ACTION
            ON DELETE NO ACTION
            NOT VALID
    );"""
    cursor.execute(proctoring_table)
    connection.commit()

    if session.get('logged_in') == True:
        connection = psycopg2.connect(conn_string)
        cursor = connection.cursor()
        # proctoring percentage
        sql_proctoring_percentage = """SELECT proctoring.percentage FROM proctoring WHERE username='lavsharma'"""
        cursor.execute(sql_proctoring_percentage)
        proctoring = cursor.fetchall()
        print("Proctoring=",proctoring)
        sql_proctoring_count = """SELECT proctoring.count FROM proctoring WHERE username='lavsharma'"""
        cursor.execute(sql_proctoring_count)
        count = cursor.fetchall()
        print("Count=", count)
        return render_template('index.html', data=proctoring)
    else:
        return render_template("login/login.html")

@app.route('/login', methods=['POST'])
def login():
    username = request.get_json(force=True).get('username')
    if not username:
        abort(401)

    conversation = get_chatroom('My Room')
    try:
        conversation.participants.create(identity=username)
    except TwilioRestException as exc:
        # do not error if the user is already in the conversation
        if exc.status != 409:
            raise

    token = AccessToken(twilio_account_sid, twilio_api_key_sid,
                        twilio_api_key_secret, identity=username)
    token.add_grant(VideoGrant(room='My Room'))
    token.add_grant(ChatGrant(service_sid=conversation.chat_service_sid))

        

    return {'token': token.to_jwt().decode(),
            'conversation_sid': conversation.sid}


@app.route('/signin', methods=['GET', 'POST'])
def signin():
    
    if request.method == "POST":
        
        username = request.form['username']
        password = request.form['password']
        print(username, password)
        try:
            connection = psycopg2.connect(conn_string)
            cursor = connection.cursor()

            username = request.form.get("username", False)
            password_entered = request.form.get("password", False)
            record_to_search = (username,)
            sql_login_query = """SELECT student.username, student.password, student.approved FROM student WHERE student.username = %s"""
            cursor.execute(sql_login_query, record_to_search)
            query = cursor.fetchone()
            username = query[0]
            password = query[1]
            approved = query[2]
            approvedS = str(approved)
            print("username=", username)
            print("password=", password)
            print("approved=", approvedS)
            connection.commit()
            
            if password_entered == password and approvedS == 'False':                
                return "Account not approved"  
            elif password_entered == password and approvedS == 'True':
                session["name"] = username
                session['logged_in'] = True

                a = datetime.datetime.now()
                b = datetime.datetime(2015,8,25,0,0,0,0)                 
                Sid = a - b
                Lid = Sid.seconds - 12900
                return render_template('index.html')  
            elif password_entered != password and approvedS == 'True':
                return "Email/password is wrong"
            else:
                return "Some other error!s"

        except (Exception, psycopg2.Error) as error:
            print("Error in insert operation", error)

        finally:
            # closing database connection.
            if (connection):
                #cursor.close()
                connection.close()
                print("PostgreSQL connection is closed")
                render_template("login/login.html")
    
    return render_template("login/login.html")


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    global cursor
    if request.method == "POST":
        username = request.form['username']
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        emailid = request.form['emailid']
        dateofbirth = request.form['dateofbirth']
        studentclass = request.form.get("studentclass", False)
        studentidno = request.form['studentidno']
        rollno = request.form['rollno']
        password = request.form['password']
        createtimestamp = '15-04-2022'
        approved = 'False'
        
        # try:
        connection = psycopg2.connect(conn_string)
        cursor = connection.cursor()

        sql_insert_query = """INSERT INTO student(
	username, firstname, lastname, emailid, dateofbirth, studentclass, studentidno, rollno, password, createtimestamp, approved)
	VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"""
        record_to_insert = (username, firstname, lastname, emailid, dateofbirth, studentclass, studentidno, rollno, password, createtimestamp, approved)
        print(record_to_insert)
        cursor.execute(sql_insert_query, record_to_insert)
        
        # proctorinig entry
        sql_proctoring_entry = """INSERT INTO proctoring(username, emailid, percentage, "createtimeStamp", count) VALUES(%s, %s, %s, %s, %s);"""
        record_to_insert_proctoring = (username, emailid, '100', '15-04-2022', '0')
        cursor.execute(sql_proctoring_entry, record_to_insert_proctoring)
        connection.commit()
        return redirect(url_for('signin'))
    return render_template('register/register.html')

@app.route('/proct')
def proct():

    # for mac
    # appscript.app('Terminal').do_script('cd /Users/samuelmonteiro/Desktop/Be-Project/Be-Project/ ; python proctoring.py') 
    
    #for windows
    # open proctoring file
    os.system("python3 /home/lav/Downloads/Be-Project/Be-Project/proctoring.py")
    return "View"

@app.route('/forgot-password')
def forgot_password():
    return render_template('forgot_password.html')

@app.route('/reset_password', methods=['GET', 'POST'])
def reset():
    if request.method == "POST":
        emailid = request.form['emailid']
        password = request.form['password']
        connection = psycopg2.connect(conn_string)
        cursor = connection.cursor()

        emailid = request.form.get("emailid", False)
        password = request.form.get("password", False)
        student_email = emailid 
        student_password = password
        sql = """ UPDATE student
                SET password = %s
                WHERE emailid = %s"""
        cursor.execute(sql, (student_password, student_email))
        connection.commit()
        render_template("reset_successful.html")
    return render_template("reset_successful.html")

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if session.get('logged_in_admin') == True:
        return redirect(url_for('admin_dashboard'))
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        try:
            connection = psycopg2.connect(conn_string)
            cursor = connection.cursor()

            username = request.form.get("username", False)
            password_entered = request.form.get("password", False)
            record_to_search = (username,)
            sql_login_query = """SELECT admin.username, admin.password FROM admin WHERE admin.username = %s"""
            cursor.execute(sql_login_query, record_to_search)
            query = cursor.fetchone()
            username = query[0]
            password = query[1]
            connection.commit()

            if password_entered == password: 
                session["name"] = username
                session['logged_in_admin'] = True              
                return redirect(url_for('admin_dashboard'))
            else:
                return "Some other error!s"

        except (Exception, psycopg2.Error) as error:
            print("Error in insert operation", error)

        finally:
            # closing database connection.
            if (connection):
                #cursor.close()
                connection.close()
                print("PostgreSQL connection is closed")
                render_template("login/admin_login.html")

    return render_template("login/admin_login.html")

@app.route('/admin_dashboard')
def admin_dashboard():
    connection = psycopg2.connect(conn_string)
    cursor = connection.cursor()

    sql_student_query = """SELECT student.firstname, student.lastname, student.emailid, student.studentidno, student.studentclass, student.approved FROM student"""
    cursor.execute(sql_student_query)
    query = cursor.fetchall()
    # count no. of students
    count_student = """SELECT COUNT(*) FROM student"""
    count = cursor.execute(count_student)
    print("Count total=", count)
    # count no. of approved account
    count_approved = """SELECT COUNT(*) FROM student"""
    countApproved = cursor.execute(count_approved)
    print("Count approved=", countApproved)
    # count no. of pending account
    count_pending = """SELECT COUNT(*) FROM student"""
    countPending = cursor.execute(count_pending)
    print("Count pending=", countPending)
    return render_template('admin_dashboard.html', data=query)

@app.route('/approve_student/<student_email>',methods = ['GET','POST'])
def approve_student(student_email):
    connection = psycopg2.connect(conn_string)
    cursor = connection.cursor()
    student_email = student_email
    sql = """ UPDATE student
            SET approved = %s
            WHERE emailid = %s"""
    cursor.execute(sql, (True, student_email))
    connection.commit()
    return redirect(url_for('admin_dashboard'))

@app.route('/block_student/<student_email>',methods = ['GET','POST'])
def block_student(student_email):
    connection = psycopg2.connect(conn_string)
    cursor = connection.cursor()
    student_email = student_email
    sql = """ UPDATE student
            SET approved = %s
            WHERE emailid = %s"""
    cursor.execute(sql, (False, student_email))
    connection.commit()
    return redirect(url_for('admin_dashboard'))

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('index'))

@app.route('/logout_admin')
def logout_admin():
    session.pop('logged_in_admin', None)
    return redirect(url_for('admin'))

if __name__ == '__main__':
    app.run(host='0.0.0.0')