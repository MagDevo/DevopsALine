import joblib
import os
import numpy as np
import cv2
        
salary_model=joblib.load("salary_model.pk1")
cap = cv2.VideoCapture(0)
face_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

os.system('cls')
print("\n\n\n\t\t\tWelcome to future tools")
print("\t\t\t------------------------")

def display():
    print("\n\n\n\t\tPress 1: salary predict")
    print("\t\tPress 2: face ditection")
    print("\t\tPress 3: cal score for current emp")
    print("\t\tPress 4: classfy score for current emp")
    print("\t\tPress 5: to exit")

def Input():
    global ch
    ch = int(input('\n\n\n\t\tEnter your choice:'))

def main():
    if ch == 1:
        exp = float(input('\n\t\tEnter year Exp :'))
        data = np.array(exp)
        print( "predicted salary : " , salary_model.predict(data.reshape(1,1))[0] )
        input('\n\n\n\t\t\tpress enter to continue...........')
        os.system('cls')
        display()
        Input()
        main()

    elif ch == 2:
        while True:
            status , photo = cap.read()

            face_cor = face_model.detectMultiScale(photo)

            if len(face_cor) == 0:
                pass
            else:   
                x1  = face_cor[0][0]
                y1 = face_cor[0][1]
                x2 = x1 + face_cor[0][2]
                y2 = y1 + face_cor[0][3]

                photo = cv2.rectangle(photo , (x1,  y1) , (x2, y2), [0,255,0], 3)
                cv2.imshow('hi' , photo)
                if cv2.waitKey(5) == 13:
                    break
        
        cv2.destroyAllWindows()
        input('\n\n\n\t\t\tpress enter to continue...........')
        os.system('cls')
        display()
        Input()
        main()
        
    elif ch == 3:
        print("cal")

    elif ch == 4:
        print("cal")

    elif ch == 5:
        exit()

    else:
        print("\n\n\n\n\t\t\tInvalid input .......!!!\n\n")
        input('\t\tpress Enter to continue....')
        os.system('cls')
        display()
        Input()
        main()
display()
Input()
main()


