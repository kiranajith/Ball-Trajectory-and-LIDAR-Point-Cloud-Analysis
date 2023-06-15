import cv2
import numpy as np
import matplotlib.pyplot as plt

lower_red = np.array([0, 150, 120])
upper_red = np.array([10, 255, 255])
x_crd = []
y_crd = []
 
cap = cv2.VideoCapture('/Users/kiranajith/Documents/UMD/673/Project1/ball.mov')

while True:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
    mask = cv2.inRange(hsv, lower_red, upper_red)
 
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    ctrs, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for ctr in ctrs:
        area = cv2.contourArea(ctr)

        if area < 100:
            continue
        M = cv2.moments(ctr)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        x_crd.append(center[0])
        y_crd.append(center[1])
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        cv2.imshow('frame',frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


x = np.array(x_crd)
y = np.array(y_crd)

X = np.vstack((x**2, x, np.ones_like(x))).T




# Part 2. Using standard least squares to fit the curve
theta = np.linalg.lstsq(X, y, rcond=None)[0]

# Print the equation of the curve
print(f"y = {theta[0]}x^2 + {theta[1]}x + {theta[2]}")

# Plot the data and the best fit curve
plt.scatter(x, y, color='red', label='Data')
x_fit = np.linspace(x.min(), x.max(), 100)
y_fit = theta[0]*x_fit**2 + theta[1]*x_fit + theta[2]
plt.plot(x_fit, y_fit, color='blue', label='Best fit curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.gca().invert_yaxis()
plt.show()

# Part 3.
y_land = y_crd[0] + 300
a, b, c = theta
discriminant = b**2 - 4*a*(c - y_land)
x_land = (-b + np.sqrt(discriminant)) / (2*a)
print(f"The x-coordinate of the landing spot is {x_land:.2f} pixels.")