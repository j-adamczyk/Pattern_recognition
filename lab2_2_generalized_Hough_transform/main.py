import cv2
import numpy as np


def plot_template_edges():
    img = cv2.imread("template.png")
    edges = cv2.Canny(img, 200, 250)
    cv2.imwrite("template_edges.png", edges)


def generalized_Hough():
    img = cv2.imread("img.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    template = cv2.imread("template.png")
    height, width = template.shape[:2]

    edges = cv2.Canny(template, 200, 250)
    ght = cv2.createGeneralizedHoughGuil()
    ght.setTemplate(edges)

    ght.setMinDist(100)
    ght.setMinAngle(0)
    ght.setMaxAngle(360)
    ght.setAngleStep(1)
    ght.setLevels(360)
    ght.setMinScale(1)
    ght.setMaxScale(1.3)
    ght.setScaleStep(0.05)
    ght.setAngleThresh(30)
    ght.setScaleThresh(100)
    ght.setPosThresh(50)
    ght.setDp(0.8)

    positions = ght.detect(img_gray)[0][0]

    for position in positions:
        center_col = int(position[0])
        center_row = int(position[1])
        scale = position[2]
        angle = int(position[3])

        found_height = int(height * scale)
        found_width = int(width * scale)

        rectangle = ((center_col, center_row),
                     (found_width, found_height),
                     angle)

        box = cv2.boxPoints(rectangle)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

        for i in range(-2, 3):
            for j in range(-2, 3):
                img[center_row + i, center_col + j] = 0, 0, 255

    cv2.imwrite("results.png", img)


if __name__ == '__main__':
    #plot_template_edges()
    generalized_Hough()
