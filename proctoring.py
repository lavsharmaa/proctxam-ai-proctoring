import cv2
import numpy as np
import math
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks, draw_marks
from _thread import *
import psycopg2
from flask import Flask, render_template, request, abort, redirect, url_for, session, Response
from flask_session import Session


__all__ = ("error", "LockType", "start_new_thread", "interrupt_main", "exit", "allocate_lock", "get_ident", "stack_size", "acquire", "release", "locked")
  

def eye_on_mask(mask, side, shape):
 
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    l = points[0][0]
    t = (points[1][1]+points[2][1])//2
    r = points[3][0]
    b = (points[4][1]+points[5][1])//2
    return mask, [l, t, r, b]

def find_eyeball_position(end_points, cx, cy):
    """Find and return the eyeball positions, i.e. left or right or top or normal"""
    x_ratio = (end_points[0] - cx)/(cx - end_points[2])
    y_ratio = (cy - end_points[1])/(end_points[3] - cy)
    if x_ratio > 3:
        return 1
    elif x_ratio < 0.33:
        return 2
    elif y_ratio < 0.33:
        return 3
    else:
        return 0

    
def contouring(thresh, mid, img, end_points, right=False):
 
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        pos = find_eyeball_position(end_points, cx, cy)
        return pos
    except:
        pass
    
def process_thresh(thresh):
 
    thresh = cv2.erode(thresh, None, iterations=2) 
    thresh = cv2.dilate(thresh, None, iterations=4) 
    thresh = cv2.medianBlur(thresh, 3) 
    thresh = cv2.bitwise_not(thresh)
    return thresh

def print_eye_pos(img, left, right):
 
    if left == right and left != 0:
        text = ''
        
        connection = psycopg2.connect(user="postgres",
                                          password="root",
                                          host="localhost",
                                          port="5432",
                                          database="proctoring")
        cursor = connection.cursor()
        
        if left == 1:
            print('Looking left')
            text = 'Looking left'
        elif left == 2:
            print('Looking right')
            text = 'Looking right'
        elif left == 3:
            print('Looking up')
            text = 'Looking up'
        else:
            print('Nothing')
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, text, (30, 30), font,  
                   1, (0, 255, 255), 2, cv2.LINE_AA) 
        
 
def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
    """Return the 3D points present as 2D for making annotation box"""
    point_3d = []
    dist_coeffs = np.zeros((4,1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))
    
    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)
    
    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d

def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix,
                        rear_size=300, rear_depth=0, front_size=500, front_depth=400,
                        color=(255, 255, 0), line_width=2):
 
    
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    # # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)
    
    
def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):
  
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    y = (point_2d[5] + point_2d[8])//2
    x = point_2d[2]
    
    return (x, y)

face_model = get_face_detector()
landmark_model = get_landmark_model()
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)
ret, img = cap.read()
thresh = img.copy()

cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)
 
size = img.shape
font = cv2.FONT_HERSHEY_SIMPLEX 

outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
d_outer = [0]*5
inner_points = [[61, 67], [62, 66], [63, 65]]
d_inner = [0]*3
 
d_outer[:] = [x / 100 for x in d_outer]
d_inner[:] = [x / 100 for x in d_inner] 

# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])

# Camera internals
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

def nothing(x):
    pass
cv2.createTrackbar('threshold', 'image', 75, 255, nothing)

countpr = 0
counth = 0
while(True):
    
    connection = psycopg2.connect(user="postgres",
                                          password="root",
                                          host="localhost",
                                          port="5432",
                                          database="proctoring")
    cursor = connection.cursor()
    # fetch the percentage
    sql_proctoring_count = """SELECT proctoring.percentage FROM proctoring WHERE username='lavsharma'"""
    cursor.execute(sql_proctoring_count)
    percentage = cursor.fetchall()
    sql_login_query = """SELECT latest id, latest.username, latest.percentage FROM latest ORDER BY 1 DESC LIMIT 1"""
    cursor.execute(sql_login_query)
    query = cursor.fetchone()
    username = query[1]
    percentage = query[2]

    connection.commit()
    ret, img = cap.read()
    rects = find_faces(img, face_model)
    
    for rect in rects:
        shape = detect_marks(img, landmark_model, rect)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask, end_points_left = eye_on_mask(mask, left, shape)
        mask, end_points_right = eye_on_mask(mask, right, shape)
        mask = cv2.dilate(mask, kernel, 5)
        
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = int((shape[42][0] + shape[39][0]) // 2)
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = cv2.getTrackbarPos('threshold', 'image')
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = process_thresh(thresh)
        
        eyeball_pos_left = contouring(thresh[:, 0:mid], mid, img, end_points_left)
        eyeball_pos_right = contouring(thresh[:, mid:], mid, img, end_points_right, True)
        print_eye_pos(img, eyeball_pos_left, eyeball_pos_right)
        # for (x, y) in shape[36:48]:
        #     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
        shape = detect_marks(img, landmark_model, rect)
        cnt_outer = 0
        cnt_inner = 0
        draw_marks(img, shape[48:])
        for i, (p1, p2) in enumerate(outer_points):
            if d_outer[i] + 3 < shape[p2][1] - shape[p1][1]:
                cnt_outer += 1 
        for i, (p1, p2) in enumerate(inner_points):
            if d_inner[i] + 2 <  shape[p2][1] - shape[p1][1]:
                cnt_inner += 1
        if cnt_outer > 3 and cnt_inner > 2:
            
            print('Mouth open')
            connection = psycopg2.connect(user="postgres",
                                          password="root",
                                          host="localhost",
                                          port="5432",
                                          database="proctoring")
            cursor = connection.cursor()
            sql_login_query = """SELECT username FROM latest ORDER BY id DESC LIMIT 1"""
            username = cursor.execute(sql_login_query)
            print(username)
            connection.commit()
            print("count = ",countpr)
            sql_update_query = """UPDATE proctoring SET count = count + 1 WHERE proctoring.username= 'innovex1';"""
            countpr = countpr + 1
            cursor.execute(sql_update_query)
            connection.commit()
            if countpr >  100 :
                countpr = 0
                sql_update_query = """UPDATE proctoring SET percentage = percentage - 1 WHERE proctoring.username= 'innovex1';"""
                cursor.execute(sql_update_query)
                sql_update_query = """UPDATE proctoring SET count = 0 WHERE proctoring.username= 'innovex1';"""
                cursor.execute(sql_update_query)
                connection.commit()
                
            cv2.putText(img, 'Mouth open', (30, 30), font,
                    1, (0, 255, 255), 2)

        else:
            print('Mouth close')
            # sql_decrease_query = """UPDATE proctoring SET count = count - 10 WHERE username=username;"""
            # cursor.execute(sql_decrease_query)
            # connection.commit()
            cv2.putText(img, 'Mouth close', (30, 30), font,
                    1, (0, 255, 255), 2)
        # show the output image with the face detections + facial landmarks
    
    if ret == True:
        faces = find_faces(img, face_model)
        for face in faces:
            marks = detect_marks(img, landmark_model, face)
            # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
            image_points = np.array([
                                    marks[30],     # Nose tip
                                    marks[8],     # Chin
                                    marks[36],     # Left eye left corner
                                    marks[45],     # Right eye right corne
                                    marks[48],     # Left Mouth corner
                                    marks[54]      # Right mouth corner
                                ], dtype="double")
            dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
            
            
            # Project a 3D point (0, 0, 1000.0) onto the image plane.
            # We use this to draw a line sticking out of the nose
            
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            
            for p in image_points:
                cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
            
            
            p1 = ( int(image_points[0][0]), int(image_points[0][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

            cv2.line(img, p1, p2, (0, 255, 255), 2)
            cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)
            # for (x, y) in marks:
            #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
            # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
            try:
                m = (p2[1] - p1[1])/(p2[0] - p1[0])
                ang1 = int(math.degrees(math.atan(m)))
            except:
                ang1 = 90
                
            try:
                m = (x2[1] - x1[1])/(x2[0] - x1[0])
                ang2 = int(math.degrees(math.atan(-1/m)))
            except:
                ang2 = 90
                
                # print('div by zero error')
            # database connection
            connection = psycopg2.connect(user="postgres",
                                          password="root",
                                          host="localhost",
                                          port="5432",
                                          database="proctoring")
            cursor = connection.cursor()
            if ang1 >= 48:
                print('Head down')
                sql_login_query = """SELECT latest.username FROM latest ORDER BY latest.id DESC LIMIT 1"""
                username = cursor.execute(sql_login_query)

                connection.commit()
                sql_update_query = """UPDATE proctoring SET count = count + 1 WHERE proctoring.username='innovex1';"""
                cursor.execute(sql_update_query)
                connection.commit()
                if counth >  50 :
                    counth = 0
                sql_update_query = """UPDATE proctoring SET percentage = percentage - 1 WHERE proctoring.username= 'innovex1';"""
                cursor.execute(sql_update_query)
                sql_update_query = """UPDATE proctoring SET count = 0 WHERE proctoring.username= 'innovex1';"""
                cursor.execute(sql_update_query)
                connection.commit()
                cv2.putText(img, 'Head down', (30, 30), font, 2, (255, 255, 128), 3)
                
            elif ang1 <= -48:
                print('Head up')
                sql_login_query = """SELECT latest.username FROM latest ORDER BY latest.id DESC LIMIT 1"""
                username = cursor.execute(sql_login_query)

                connection.commit()
                sql_update_query = """UPDATE proctoring SET count = count + 1 WHERE proctoring.username='innovex1';"""
                cursor.execute(sql_update_query)
                connection.commit()
                if counth >  50 :
                    counth = 0
                sql_update_query = """UPDATE proctoring SET percentage = percentage - 1 WHERE proctoring.username= 'innovex1';"""
                cursor.execute(sql_update_query)
                sql_update_query = """UPDATE proctoring SET count = 0 WHERE proctoring.username= 'innovex1';"""
                cursor.execute(sql_update_query)
                connection.commit()
                cv2.putText(img, 'Head up', (30, 30), font, 2, (255, 255, 128), 3)
             
            if ang2 >= 48:
                print('Head right')
                sql_login_query = """SELECT latest.username FROM latest ORDER BY latest.id DESC LIMIT 1"""
                username = cursor.execute(sql_login_query)

                connection.commit()
                sql_update_query = """UPDATE proctoring SET count = count + 1 WHERE proctoring.username='innovex1';"""
                cursor.execute(sql_update_query)
                connection.commit()
                if counth >  50 :
                    counth = 0
                sql_update_query = """UPDATE proctoring SET percentage = percentage - 1 WHERE proctoring.username= 'innovex1';"""
                cursor.execute(sql_update_query)
                sql_update_query = """UPDATE proctoring SET count = 0 WHERE proctoring.username= 'innovex1';"""
                cursor.execute(sql_update_query)
                connection.commit()
                cv2.putText(img, 'Head right', (90, 30), font, 2, (255, 255, 128), 3)
            elif ang2 <= -48:
                print('Head left')
                sql_login_query = """SELECT latest.username FROM latest ORDER BY latest.id DESC LIMIT 1"""
                username = cursor.execute(sql_login_query)

                connection.commit()
                sql_update_query = """UPDATE proctoring SET count = count + 1 WHERE proctoring.username='innovex1';"""
                cursor.execute(sql_update_query)
                connection.commit()
                if counth >  10 :
                    counth = 0
                sql_update_query = """UPDATE proctoring SET percentage = percentage - 1 WHERE proctoring.username= 'innovex1';"""
                cursor.execute(sql_update_query)
                sql_update_query = """UPDATE proctoring SET count = 0 WHERE proctoring.username= 'innovex1';"""
                cursor.execute(sql_update_query)
                connection.commit()
                cv2.putText(img, 'Head left', (90, 30), font, 2, (255, 255, 128), 3)
            else:
                print('Normal')
                cv2.putText(img, 'Normal', (90, 30), font, 2, (255, 255, 128), 3)
            cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
            cv2.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
 
  
        
    # cv2.imshow('eyes', img)
    # cv2.imshow("image", thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

 
