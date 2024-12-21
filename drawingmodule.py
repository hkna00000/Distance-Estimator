import cv2

# Initialize global variables
drawing = False
ix, iy = -1, -1
bbox = []

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, bbox

    if event == cv2.EVENT_LBUTTONDOWN:  # Start drawing
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:  # Update the rectangle while dragging
        if drawing:
            temp_img = param.copy()
            cv2.rectangle(temp_img, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Draw Bounding Box", temp_img)

    elif event == cv2.EVENT_LBUTTONUP:  # Finish drawing
        drawing = False
        bbox = [ix, iy, x, y]
        cv2.rectangle(param, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("Draw Bounding Box", param)

# Function to draw bounding box and return coordinates
def select_bounding_box(image_path):
    global bbox
    img = cv2.imread(image_path)
    temp_img = img.copy()

    cv2.imshow("Draw Bounding Box", temp_img)
    cv2.setMouseCallback("Draw Bounding Box", draw_rectangle, param=temp_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return bbox
