import cv2 as cv


file = "D:/Stacks/C224850-LH-D13-60X-05252022_STACKED/DAPI.jpg"

img = cv.imread(file, cv.IMREAD_GRAYSCALE)

blurred = img[:1000,:1000]
#blurred = img.copy()
cv.imwrite(f"D:/Stacks/Blurred/blur_{0}.jpg", blurred)


blurred1 = cv.blur(blurred, ksize=(25, 25), borderType= cv.BORDER_DEFAULT)

cv.imwrite(f"D:/Stacks/Blurred/blur_{1}.jpg", blurred1)

blurred2 = cv.blur(blurred1, ksize=(25, 25), borderType= cv.BORDER_DEFAULT)

cv.imwrite(f"D:/Stacks/Blurred/blur_{2}.jpg", blurred2)

blurred3 = cv.blur(blurred2, ksize=(25, 25), borderType= cv.BORDER_DEFAULT)
cv.imwrite(f"D:/Stacks/Blurred/blur_{3}.jpg", blurred3)

blurred4 = cv.blur(blurred3, ksize=(25, 25), borderType= cv.BORDER_DEFAULT)
cv.imwrite(f"D:/Stacks/Blurred/blur_{4}.jpg", blurred4)

blurred5 = cv.blur(blurred4, ksize=(25, 25), borderType= cv.BORDER_DEFAULT)
cv.imwrite(f"D:/Stacks/Blurred/blur_{5}.jpg", blurred5)

blurred6 = cv.blur(blurred5, ksize=(25, 25), borderType= cv.BORDER_DEFAULT)
cv.imwrite(f"D:/Stacks/Blurred/blur_{6}.jpg", blurred6)