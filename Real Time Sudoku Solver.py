import cv2
import numpy as np
from tensorflow.keras.models import load_model


# load model
model = load_model(r'D:\CNN\Models\model2.h5')



############### FUNCTION TO CHECK WHETHER A NUMBER CAN BE WRITTEN OR NOT IN A BLANK SPACE #############

def check(num,row,col):
    global sudoku_grid
    for i in range(0,9):
        if(sudoku_grid[row][i]==num):  # CHECKING THE DIGIT IN THE SAME ROW
            return False
        if(sudoku_grid[i][col]==num):  # CHECKING THE DIGIT IN THE SAME COLUMN
            return False
    start_index_row=(row//3)*3
    start_index_col=(col//3)*3
    for i in range(start_index_row,start_index_row+3):    # CHECKING THE DIGIT IN THE SAME SUB GRID
        for j in range(start_index_col,start_index_col+3):
             if(sudoku_grid[i][j]==num):
                 return False
    return True

############################### FUNCTION TO SOLVE THE SUDOKU GRID ######################################

def solve():
    global sudoku_grid
    for row in range(0,9):
        for col in range(0,9):
            if(sudoku_grid[row][col]==0):          # CHECKING FOR BLANK SPACES
                for num in range(1,10):
                    if(check(num,row,col)): # IF A NUMBER CAN BE WRITTEN, WRITE IT FOR NOW
                        sudoku_grid[row][col]=num
                        if(solve()):# NOW CHECK FOR NEXT BLANK SPACE,KEEPING THE CONDITIONS FOR CURRENT NUMBER AT A HALT
                            return True
                        else:
                            sudoku_grid[row][col]=0     # IF WE AR STUCK THEN BACKTRACK, AND CONTINUE THE CONDITIONS KEPT AT HALT TO FIND THE NEXT SUITABLE NUMBER
                return False                # IF ALL THE NUMBERS FROM 1 TO 9 ARE TESTED AND NONE IS SUITED THEN IT MEANS THAT A NUMBER AT A PREVIOUS BLANK SPACE IS INCORRECT, HENCE WE NEED TO BACKTRACK
    return True    # IF WE END UP HERE THEN IT MEANS THAT THE SUDOKU HAS BEEN SOLVED

######################## FUNCTION TO FIND THE CONTOUR OF SUDOKU GRID ###########################################

def find_grid(img,img0):
    global grid_cnt
    contours,h=cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if(len(contours)!=0):
        max_cnt = max(contours, key=cv2.contourArea)
        area=cv2.contourArea(max_cnt)
        if(area>70000):
            grid_cnt=max_cnt # CONTOUR WITH AREA GREATER THAN 50000 IS DETECTED AND STORED AS POTENTIAL GRID
    cv2.imshow("r", img0)

######################### FUNCTION TO GET THE COORDINATES OF CORNER POINTS OF THE DETCTED GRID ##################
def get_corner_pts(grid_cnt,img0):
    peri = cv2.arcLength(grid_cnt, True)
    corner_pts = cv2.approxPolyDP(grid_cnt, 0.02 * peri, True)
    cv2.imshow("r", img0)
    return corner_pts

############################ FUNCTION TO REARRANGE THE ORDER OF CORNER POINTS DETECTED ###########################
def rearrange_corners(corner_pts,img0):
    corner_pts = corner_pts.reshape(4, 2)
    corner_pts_new = corner_pts.copy()
    sum = corner_pts.sum(axis=1)
    """"
    Sum of x and y                 y-x of this 
    of this point   ------------   point is 
    is minimum of  |            |  minimum
    the four       |            |
                   |            |
                   |            |     
    y-x of this     ------------   Sum of x and y of 
    point is                       this point is maximum 
    maximum                        of the four
    
    """
    corner_pts_new[0] = corner_pts[np.argmin(sum)]
    corner_pts_new[3] = corner_pts[np.argmax(sum)]
    diff = np.diff(corner_pts, axis=1)
    corner_pts_new[1] = corner_pts[np.argmin(diff)]
    corner_pts_new[2] = corner_pts[np.argmax(diff)]
    warp_pts = np.float32(corner_pts_new)
    cv2.imshow("r", img0)
    return warp_pts

######################## FUNCTION TO GET THE WARPED IMAGE OF SUDOKU GRID #######################################
def warp(img,warp_pts):
    width,height=360,360
    warp_coords=np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix=cv2.getPerspectiveTransform(warp_pts,warp_coords)
    warp_img=cv2.warpPerspective(img,matrix,(width,height))
    return warp_img,matrix

######################### FUNCTION TO SPLIT THE GRID INTO 81 BOXES ##############################################
def split_grid(warp_img,img0):
    warp_img1=warp_img.copy()
    #warp_img1=warp_img1[9:warp_img1.shape[0]-9,9:warp_img1.shape[1]-9] # Removing borders of the sudoku

    rows=np.vsplit(warp_img1,9)
    boxes=[]
    for r in rows:
        cols=np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    cv2.imshow("r", img0)
    return boxes

################################ FUNCTION TO SEPERATE DIGITS FROM ALL THE BOXES ###################################
def seperate_digits(boxes,img0):

    global frames,blank_space_original,blank_space_test
    blank_space_test=[]

    for i in range(0,81):
        img=np.asarray(boxes[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (32, 32))
        img = img[5:img.shape[0] - 5, 5:img.shape[1] - 5]

        img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,8)
        img1=cv2.bitwise_not(img)
        contours, h = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # THE LARGEST CONTOUR IS DETECTED
        if(len(contours)!=0):
            max_cnt = max(contours, key=cv2.contourArea)
            x,y,w,h=cv2.boundingRect(max_cnt)



            img2 =np.zeros((32,32),np.uint8) # CREATING A BLACK BACKGROUND TO PASTE DETECTED DIGIT ON IT

            if(w>5 and h>10): # LARGEST CONTOUR MUST SATISFY THESE CONDITIONS IF IT IS A DIGIT
                if(h<=20):
                    flag_h=1
                else:
                    flag_h=0
                if(w<=20):
                    flag_w=1
                else:
                    flag_w=0

                # POSITIONING THE DETECTED DIGIT IN CENTER
                if(flag_h==1 and flag_w==1):
                    img2[4:4+h,9:9+w]=img1[y:y+h,x:x+w].copy()
                if(flag_h==1 and flag_w==0):
                    img2[4:4 + h, x:x + w] = img1[y:y + h, x:x + w].copy()
                if(flag_h==0 and flag_w==1):
                    img2[y:y + h, 9:9 + w] = img1[y:y + h, x:x + w].copy()
                if(flag_h==0 and flag_w==0):
                    img2[y:y + h, x:x + w] = img1[y:y + h, x:x + w].copy()

                img2 = cv2.bitwise_not(img2)
                img2=cv2.erode(img2,np.ones((2,2),np.uint8),iterations=1)
                boxes[i]=img2


            # THERE IS NO DIGIT
            else:
                img2 = np.zeros((32, 32), np.uint8)
                boxes[i]=img2
                if(frames==15):
                    blank_space_original.append(i)
                if(frames>=30):
                    blank_space_test.append(i)

        # THERE IS NO DIGIT
        else:
            img2=np.zeros((32, 32), np.uint8)
            boxes[i]=img2
            if (frames == 15):
                blank_space_original.append(i)
            if (frames >= 30):
                blank_space_test.append(i)


        cv2.imshow("r", img0)

    return boxes

################################### READ THE DIGITS ####################################################################
def read_digits(boxes,img0):
    global index,digits,stop_reading,grid_constructed

    for i in range(0,3):
        if(boxes[index][0][0]==0):
            digits.append(0)
        else:
            boxes[index]=cv2.resize(boxes[index],(32,32))
            boxes[index]=boxes[index]/255.0
            img = boxes[index].reshape(1, 32, 32, 1)
            pred = np.argmax(model.predict(img), axis=-1)
            digits.append(pred[0])
        index=index+1
        if(index==81):
            digits_array=np.array(digits)
            sudoku_grid=np.reshape(digits_array,(9,9))
            stop_reading = 1
            grid_constructed=1
            return sudoku_grid

#################################### TO GET INDEX OF BLANK SPACES OF SUDOKU GRID ########################################

def get_index(sudoku_grid):
    index_lst=[]
    for i in range(0,9):
        for j in range(0,9):
            if(sudoku_grid[i][j]==0):
                index_lst.append([i,j])
    return index_lst



#################################### WRITES DIGIT ON THE SUDOKU GRID ####################################################

def write_digits(index_lst,warp_img):
    global blank_space_original,sudoku_grid
    x_error=-10
    y_error=10
    for i in range(0,len(index_lst)):
        X=index_lst[i][1]*40+20+x_error
        Y=index_lst[i][0]*40+20+y_error
        warp_img=cv2.putText(warp_img,str(sudoku_grid[index_lst[i][0]][index_lst[i][1]]),(X,Y),
                             cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2,cv2.LINE_AA)
    return warp_img


##################################### THIS FUNCTION CHECKS IF IT IS THE SAME SUDOKU GRID OR NOT #######################################################
def check_grid():
    global blank_space_original,blank_space_test,frames,grid_constructed,stop_reading,stop_solving,index,digits,sudoku_grid
    if(blank_space_original==blank_space_test):
        frames=frames
    else:
        frames=0
        blank_space_original=[]
        blank_space_test=[]
        grid_constructed=0
        stop_reading=0
        stop_solving=0
        index=0
        digits=[]
        print("Sudoku grid changed or it is not detected properly")


#######################################################################################################################################################

cap=cv2.VideoCapture(0)
z=1#
frames=0#
blank_space_original=[]#
blank_space_test=[]#
index=0#
digits=[]#
grid_constructed=0#
sudoku_grid=np.ones((9,9),dtype=np.uint8)#
stop_reading=0#
stop_solving=0#
while(z):
    grid_cnt=[] # THIS LIST IS USED TO STORE THE CONTOUR POINTS OF GRID DETECTED,AND THIS LIST IS REFRESHED AT EACH LOOP
    ret, img = cap.read()

    #img=cv2.imread(r"C:\Users\ashis\OneDrive\Desktop\sudoku.jpg",1)  # THIS IS USED WHEN WE HAVE TO DETECT SUDOKU FROM A PHOTO

    img0=img.copy() # img0 IS USED TO SHOW THE DETECTED GRID
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # img1 IS USED TO PROCESS THE IMAGE TO DETECT THE GRID
    img1 = cv2.Canny(img1, 80, 150)
    find_grid(img1,img0)

    if(len(grid_cnt)!=0):

        corner_pts=get_corner_pts(grid_cnt,img0)

        if(len(corner_pts)==4): # IF CORNERS ARE DETECTED
            frames=frames+1 # IT IS INCREMENTED WHENEVER THE SUDOKU GRID IS DETECTED
            warp_pts=rearrange_corners(corner_pts,img0) # GETTING THE WARP POINTS IN CORRECT ORDER
            warp_img,matrix=warp(img,warp_pts) # GETTING THE WARP IMAGE

            boxes=split_grid(warp_img,img0) # SPLITTING THE GRID INTO 81 BOXES
            boxes=seperate_digits(boxes,img0) # GETTING DIGITS READY FOR CNN MODEL

            if(frames>=30):
                check_grid() # CHECKING IF CURRENT SUDOKU IS SAME AS PREVIOUS SUDOKU
                if(grid_constructed==0 and stop_reading==0):
                    sudoku_grid=read_digits(boxes,img0)  # RECOGNISING THE DIGITS



                if(grid_constructed==1):
                    if(stop_solving==0):
                        index_lst=get_index(sudoku_grid) # GETTING INDEX OF BLANK SPACES IN ROW AND COLUMN FORM
                        if(solve()):
                            stop_solving=1
                    if(stop_solving==1):
                        warp_img=write_digits(index_lst,warp_img) # WRITING THE DIGIT ON WARPED GRID

            cv2.imshow("e", warp_img)

        if(frames < 30):
            img0 = cv2.drawContours(img0, [grid_cnt], 0, (0, 0, 255),4)  # GRID IS SHOWED WITH RED BORDER AS IT IS NOT KNOWN TILL NOW THAT IT IS SUDOKU GRID OR SOME OTHER GRID

        if((frames>=30) and (grid_constructed==0)):
            img0 = cv2.drawContours(img0, [grid_cnt], 0, (0, 255, 255), 4) # GRID IS SHOWED WITH YELLOW WHEN DIGITS ARE BEING RECOGNISED

        if((frames>=30) and (grid_constructed==1)):
            img0 = cv2.drawContours(img0, [grid_cnt], 0, (0, 255, 0), 4) # GRID IS SHOWED WITH GREEN WHEN SUDOKU IS SOLVED


    cv2.imshow("r", img0)

    if (cv2.waitKey(1) == ord("q")):
        cap.release()
        cv2.destroyAllWindows()
        Z = 0
        break