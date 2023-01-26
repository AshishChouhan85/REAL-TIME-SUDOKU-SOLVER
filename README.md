# REAL-TIME-SUDOKU-SOLVER

 This project detects and solves a sudoku grid in real time. 

![Screenshot (89)](https://user-images.githubusercontent.com/60431758/113689933-7b4aa280-96e8-11eb-838a-4726d47abf68.png)<br>


- # EDGE DETECTION

Edge is detected using Canny edge detector which gives a noiseless image.

![Screenshot (91)](https://user-images.githubusercontent.com/60431758/113690017-974e4400-96e8-11eb-94f9-eecd3a99fb93.png)<br>


- # CONTOUR DETECTION

From all the detected contours, the contour whose area is greater than a threshold is selected, that is generally our sudoku grid

![Screenshot (90)](https://user-images.githubusercontent.com/60431758/113690389-f613bd80-96e8-11eb-8bd7-90ed18f90c57.png)<br>


- # WARPING THE SUDOKU GRID

![Screenshot (92)](https://user-images.githubusercontent.com/60431758/113690716-5276dd00-96e9-11eb-9ee2-d6c633766ee1.png)<br>


- # SPLITTING THE GRID

The warped grid is splitted into 81 boxes.

![Screenshot (104)](https://user-images.githubusercontent.com/60431758/113702144-bfdd3a80-96f6-11eb-9381-4bbc8eb780c6.png)<br>


- # PREPROCESSING THE BOXES

All the 81 boxes are pre-processed which follows the followinng sequence:
- Border removal
- Adaptive thresholding
- Moving the number in the middle of the box

All the blank images are converted into black boxes.

![Screenshot (105)](https://user-images.githubusercontent.com/60431758/113702174-c5d31b80-96f6-11eb-9aef-1e336bdb8614.png)<br>


- # RECOGNISING THE DIGITS

The digits are recognisied using a trained CNN model. The model is trained using 10000 images. 25 epochs were done to train the model.

![Screenshot (96)](https://user-images.githubusercontent.com/60431758/113702391-1185c500-96f7-11eb-96f4-eed8c465317e.png)<br>

Images of all the numbers were present in almost equal number.

![Screenshot (98)](https://user-images.githubusercontent.com/60431758/113702559-4d208f00-96f7-11eb-8f6e-6802556e6875.png)<br>

## Accuracy vs Val_Accuracy Graph

![Screenshot (97)](https://user-images.githubusercontent.com/60431758/113704106-590d5080-96f9-11eb-9392-4fa71a1ffd40.png)

- # MAKING THE SUDOKU GRID

Blank spaces are represented by 0.

![Screenshot (99)](https://user-images.githubusercontent.com/60431758/113705764-73482e00-96fb-11eb-8b04-df58d3a49871.png)

- # SOLVING THE SUDOKU GRID

The sudoku grid is solved using backtracking.

![Screenshot (100)](https://user-images.githubusercontent.com/60431758/113705803-7fcc8680-96fb-11eb-8490-eb37f39519e9.png)

- # SHOWING THE DIGITS

The solved grid is showed by writing the numbers on blank spaces on the warped image.

![Screenshot (93)](https://user-images.githubusercontent.com/60431758/113706109-ea7dc200-96fb-11eb-85cd-d71b6bfecf7b.png)

# WORKING VIDEO

![ezgif com-gif-maker (7)](https://user-images.githubusercontent.com/60431758/113706748-ab9c3c00-96fc-11eb-9dc9-49baddc09edd.gif)

# SALIENT FEATURES

- Grid is shown with different colours showing different processes. Red border means it is storing index of blank spaces which will be used later to ensure that sudoku grid is not changed. Yellow border means digits are being recognised and grid is being made. Green border means sudoku is being solved and when it is solved it is shown.
- Index of blank spaces of current sudoku grid is compared with the index of blank spaces stored when grid was red. If at any time there is difference in index of blank spaces then whole process will begin from start. This helps to detect that the sudoku grid has been changed.
- 3 boxes are given at a time to recognise the digits, hence framerate remains constant even in systems which do not have GPU. 























