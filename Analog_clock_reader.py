"""
Created on Tue May  5 13:40:28 2020
@author: Balazs Horvath
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Funcție pentru calcul distanței euclidiene
def eucledian_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# Funcție folosit pentru transformare valorii de unghi din radiani în grade
def radius_to_degree(angle):
    return (angle * 180) / np.pi


# Funcție care elimină câteva liniile care nu sunt aproare de centrul cercului
# Se calculează distanța dintre centrul cercului și capetele liniilor, dacă această valoare este mai mic decât threshold linia
# va fi eliminită fiindcă nu poate fi limba ceasului
def filter_near_center(center_x, center_y, line, threshold):
    x1, y1, x2, y2 = line[0]
    d1 = eucledian_distance(center_x, center_y, x1, y1)
    d2 = eucledian_distance(center_x, center_y, x2, y2)

    if d1 < threshold or d2 < threshold:
        return True
    else:
        return False


# Funcția care elimină liniile dublicate pentru o limnbă: se calculează pentru un set de linii
# În cazul condiției adevărate, se elimină linii pe poziția i
def filter_near_lines(lines):
    for i, line in enumerate(lines):
        condition = compare_line(line, lines.pop(i))
        if condition:
            lines = lines.pop(i)


# Funcția folosit pentru condition în filter_near_lines, se compară o linie cu set de linii și în cazul în care
# delta d (diferența dintre lungimi) și delta angle (diferența de unghi) sunt mai mici decât threshold_distance și
# threshold_angle, înseamnă cu avem 2 linii cu orientare și lungimi aproape egale, astfel trebuie eliminate din lista liniile una dintre ele
def compare_line(instance_line, list_lines, threshold_angle, threshold_distance):
    x1, y1, x2, y2 = instance_line
    d_instance = eucledian_distance(x1, y1, x2, y2)
    slope = (y2 - y1) / (x2 - x1)
    angle_instance = radius_to_degree(np.arctan(slope))

    for line in list_lines:
        x1, y1, x2, y2 = line
        d = eucledian_distance(x1, y1, x2, y2)
        slope = (y2 - y1) / (x2 - x1)
        angle = radius_to_degree(np.arctan(slope))
        delta_d = abs(d_instance - d)
        delta_angle = abs(angle_instance - angle)
        if delta_d < threshold_distance and delta_angle < threshold_angle:
            return True
        else:
            return False


# detectare orar/minutar. În momentul în care avem numai două linii, se decide pe baza lungimiilor care este minutarul și orarul
def identify_lines(lines):
    lengths = []

    for line in lines:
        lengths.append(eucledian_distance(line[0], line[1], line[2], line[3]))

    if lengths[0] > lengths[1]:
        return (lines[1], lines[0])
    else:
        return (lines[0], lines[1])


# detectare quadranului
# se ia capătul liniei, care este mai departe de centrul cercului și se aplică check_point
def identify_quadran(line, center_x, center_y):
    quadran = -1
    (x1, y1, x2, y2) = line
    d1 = eucledian_distance(x1, y1, center_x, center_y)
    d2 = eucledian_distance(x2, y2, center_x, center_y)

    if d1 > d2:
        quadran = check_point(x1, y1, center_x, center_y)
    else:
        quadran = check_point(x2, y2, center_x, center_y)
    return quadran


# Check_point ia decizia referitor la qudran
# Se calculează diferența de coordonatele dintre punctul respectiv și centrul cercului
# S-a ales quadran 1 pentru orele 12-3, quadran 2 pentru 9-12, quadran 3 pentru 6-9 și quadran 4 pentru 3-6
def check_point(x, y, center_x, center_y):
    verify_x = x - center_x
    verify_y = y - center_y

    if verify_x > 0 and verify_y > 0:
        return 4
    if verify_x > 0 and verify_y < 0:
        return 1
    if verify_x < 0 and verify_y > 0:
        return 3
    if verify_x < 0 and verify_y < 0:
        return 2


# identificare orei pe baza unghiului și quadranului. Se schimbă ora la fiecare schimb de 30 de grade a valorii unghiului
def identify_hour(quadran, angle, hours):
    if (quadran == 1):
        index = int(2 - abs(angle) // 30)
        return hours[index - 1]
    elif (quadran == 2):
        index = int(9 + (abs(angle) // 30))
        return hours[index - 1]
    elif (quadran == 3):
        index = int(8 - (abs(angle) // 30))
        return hours[index - 1]
    else:
        index = int(3 + abs(angle) // 30)
        return hours[index - 1]


# identificare minutului pe baza unghiului și quadranului. Se schimbă minutul la fiecare schimb de 6 grade a valorii unghiului
def identify_minute(quadran, angle, minutes):
    if (quadran == 1):
        index = int(14 - abs(angle) // 6)
        return minutes[index]
    elif (quadran == 2):
        index = int(45 + (abs(angle) // 6))
        return minutes[index]
    elif (quadran == 3):
        index = minutes(44 - (abs(angle) // 6))
        return hours[index]
    else:
        index = minutes(15 + abs(angle) // 6)
        return hours[index]


# Citirea imaginii
img = cv2.imread('Analog_clock.jpg', 0)

img = cv2.resize(img, (460, 460), interpolation=cv2.INTER_AREA)  # e bun doar pentru poze patratice

img = cv2.medianBlur(img, 3)
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# detectarea ceasului folosind Circular Hough Transform
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 200,
                           param1=50, param2=40, minRadius=150, maxRadius=0)

# Desenarea cercului detectat
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    # marginea ceasului
    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # centrul ceasului
    cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow('detected circles', cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Extracția informațiilor privind ceasul: Coordonatele centrului și raza
radius = circles[0][0][2]
center_x = circles[0][0][0]
center_y = circles[0][0][1]
# cimg2 = cv2.cvtColor(cimg,cv2.COLOR_BGR2GRAY)
crop_image = cimg[center_x - radius:center_x + radius, center_y - radius:center_y + radius, :]

cv2.imshow('cropped_image', crop_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Detecție de contur, pentru a putea extrage limbiile
edges = cv2.Canny(crop_image, 150, 250, apertureSize=3)
cv2.imshow('detected_edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

minLineLength = 25
maxLineGap = 5

image_modified = crop_image.copy()
count = 0
lines_of_interest = []

# Detecția de linii folosind HoughLines și filtrarea lor folosind cele două funcții definite
lines = cv2.HoughLinesP(edges, 0.5, np.pi / 360, 60, minLineLength, maxLineGap)
[a, b, c] = np.shape(lines)
for i in range(0, a):
    for x1, y1, x2, y2 in lines[i]:
        condition = False
        condition = filter_near_center(center_x, center_y, lines[i], 25)
        if condition:
            count += 1
            cv2.line(image_modified, (x1, y1), (x2, y2), (0, 255, 0), 2)
            lines_of_interest.append([x1, y1, x2, y2])

cv2.imshow('detected lines', image_modified)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Detectare pentru orar și minutar pe baza lungimiilor liniilor
(orar, minutar) = identify_lines(lines_of_interest)

slope_orar = 0
slope_minutar = 0
angle_orar = 0
angle_minutar = 0

# Calcul unghiului dintre linia și orizontala, folosind arcustangenta pantei dreptei
slope_orar = (orar[3] - orar[1]) / (orar[2] - orar[0])
angle_orar = radius_to_degree(np.arctan(slope_orar))
slope_minutar = (minutar[3] - minutar[1]) / (minutar[2] - minutar[0])
angle_minutar = radius_to_degree(np.arctan(slope_minutar))

# Detecția quadranelor
quadran_minutar = identify_quadran(minutar, center_x, center_y)
quadran_orar = identify_quadran(orar, center_x, center_y)

# print(quadran_minutar)
# print(quadran_orar)
hours = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
minutes = np.arange(0, 60)

# identificarea orei și minutului, folosind posibilele valori, unghiul și quadranul
time_hour = identify_hour(quadran_orar, angle_orar, hours)
time_minute = identify_minute(quadran_minutar, angle_minutar, minutes)

# Afișare rezultatului
print('The time is {} hours {} minutes'.format(time_hour, time_minute))
