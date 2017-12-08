import cv2

def nothing(x):
    pass

cap = cv2.VideoCapture('data/india_driving_sample.mp4')
fourcc = cv2.VideoWriter_fourcc(*'H264') # For fourcc visit fourcc.org
cap_size = (int(cap.get(3)),int(cap.get(4)))
out = cv2.VideoWriter('outputs/edges_india_output.avi',fourcc,20,cap_size,False)

# Infinite loop until we hit the escape key on keyboard
while(1):
    ret, frame = cap.read()
    if not ret:
        break
    # get current positions of four trackbars
    lower = 140
    upper = 220
    edges = cv2.Canny(frame, lower, upper)

    # display images
    cv2.imshow('edges',edges)
    out.write(edges)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:   # hit escape to quit
        break

cap.release()
out.release()
cv2.destroyAllWindows()
