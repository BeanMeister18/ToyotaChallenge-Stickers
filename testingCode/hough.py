def houghCircleDetector(img, og_image):
    
    img = cv2.medianBlur(img,3)
    img_edge = cv2.Canny(img,50,100)
    # plt.figure(figsize=(20,10))
    # #Note: matplotlib uses RGB format so had to convert BGR-to-RGB
    # plt.imshow(cv2.cvtColor(img_edge,cv2.COLOR_BGR2RGB))
    # plt.title('RGB Image',color='c')
    # plt.show()

    # circles = cv2.HoughCircles(img_edge,cv2.HOUGH_GRADIENT,1,minDist=20,param1=200,param2=70)
    circles = cv2.HoughCircles(img_edge, cv2.HOUGH_GRADIENT, dp=1.3, minDist=20, param1=100, param2=45, minRadius=0, maxRadius=50)
    try:
      circles = np.uint16(np.around(circles))
      for val in circles[0,:]:
          cv2.circle(og_image,(val[0],val[1]),val[2],(255,0,0),2)
    except:
       pass

    # plt.figure(figsize=(20,10))
    # plt.subplot(121),plt.imshow(cv2.cvtColor(img_edge,cv2.COLOR_BGR2RGB)),plt.title('Input',color='c')
    # plt.subplot(122),plt.imshow(cv2.cvtColor(og_image,cv2.COLOR_BGR2RGB)),plt.title('Result',color='c')
    # plt.show()