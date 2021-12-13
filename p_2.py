import tempfile
import cv2
import pytesseract
import numpy as np
from tkinter import messagebox
from matplotlib import pyplot as plt
import glob

import p_1
import p_3

class Second2(object):
    pytesseract.pytesseract.tesseract_cmd ='C:/Program Files/Tesseract-OCR/tesseract.exe'

    # ======================
    # Global Variable
    # ======================
    global cascPath
    # global inputFile

    cascPath = 'haarcascade_frontalface_default.xml'

    percentType = 5
    detectTypeResult = 0
    PersonalInfoType="미확인"
    patternDir=r'./Config/pattern'

    PatternIDCARD=glob.glob(patternDir+"/IDCARD/*.jpg")    #주민등록증
    PatternDRIVER=glob.glob(patternDir+"/DRIVER/*.jpg")    #운전면허증
    PatternFAMILLY=glob.glob(patternDir+"/FAMILLY/*.jpg")  #가족관계증명서
    PatternPASSPORT=glob.glob(patternDir+"/PASSPORT/*.jpg")#여권
    PatternRESIDENT=glob.glob(patternDir+"/RESIDENT/*.jpg")#주민등록표

    # ======================
    # Custom Functions
    # ======================

    # Read the image
    def readImg(self, filepath) :
        image = cv2.imread(filepath)
        return image

    def diffImg(self, img1, img2):
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw first 10 matches.
        # outImg = np.empty((1,1))
        # img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], outImg, flags=2)

        # plt.imshow(img3),plt.show()


        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = []

        for m, n in matches:
            if m.distance < 0.71 * n.distance:
                good.append([m])

        print('PersonalType Good Detected Count: %d'%(len(good)))
        if len(good) > self.percentType:
            knn_image = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

            #plt.imshow(knn_image)
            #plt.show()

        return len(good)


    def detectedType(self, OrgMat,PatternMat):
        #iPattermImg = cImg.readImg(pattern)
        #img2 = cImg.readImg(orgMat)
        if self.diffImg(OrgMat, PatternMat) > self.percentType:
            result = 1
        else:
            result = 0

        return result

    # ======================
    # Main function
    # ======================

    def start(self):
        self.detectTypeResult = 0
        self.PersonalInfoType="미확인"
        
        first_py = p_1.firs()
        file_path = first_py.inputFile
        inputImg = self.readImg(file_path)

        #주민등록증
        if self.detectTypeResult == 0:
            print('\n###DEBUG_LoadPatternData:[ Pattern_IDCARD ]')
            for i in range(0,len(self.PatternIDCARD)):
                _iPattermImg = self.readImg(self.PatternIDCARD[i])
                print('LoadPatternIMG: [%s]' %(self.PatternIDCARD[i]))
                if self.detectedType(inputImg,_iPattermImg) != 0:
                    self.detectTypeResult = 1
                    self.PersonalInfoType = '주민등록증'

                    p_3.de_idcard(file_path)
                    #print('PersonalType Detected: %s' % (PersonalInfoType))
                    # break
        #운전면허증
        elif self.detectTypeResult == 0:
            print('\n###DEBUG_LoadPatternData:[ Pattern_DRIVER ]')
            for i in range(0, len(self.PatternDRIVER)):
                _iPattermImg = self.readImg(self.PatternDRIVER[i])
                print('LoadPatternIMG: [%s]' %(self.PatternDRIVER[i]))
                if self.detectedType(inputImg, _iPattermImg) != 0:
                    self.detectTypeResult = 2
                    self.PersonalInfoType = '운전면허증'
                    #print('PersonalType Detected: %s' % (PersonalInfoType))
                    # break
        #가족관계증명서
        elif self.detectTypeResult == 0:
            print('\n###DEBUG_LoadPatternData:[ Pattern_FAMILLY ]')
            for i in range(0, len(self.PatternFAMILLY)):
                _iPattermImg = self.readImg(self.PatternFAMILLY[i])
                print('LoadPatternIMG: [%s]' %(self.PatternFAMILLY[i]))
                if self.detectedType(inputImg, _iPattermImg) != 0:
                    self.detectTypeResult = 3
                    self.PersonalInfoType = '가족관계증명서'
                    # print('PersonalType Detected: %s' % (PersonalInfoType))
                    # break
        #여권
        elif self.detectTypeResult == 0:
            print('\n###DEBUG_LoadPatternData:[ Pattern_PASSPORT ]')
            for i in range(0, len(self.PatternPASSPORT)):
                _iPattermImg = self.readImg(self.PatternPASSPORT[i])
                print('LoadPatternIMG: [%s]' %(self.PatternPASSPORT[i]))
                if self.detectedType(inputImg, _iPattermImg) != 0:
                    self.detectTypeResult = 4
                    self.PersonalInfoType = '여권'
                    # print('PersonalType Detected: %s' % (PersonalInfoType))
                    # break
        #주민등록표
        elif self.detectTypeResult == 0:
            print('\n###DEBUG_LoadPatternData:[ Pattern_RESIDENT ]')
            for i in range(0, len(self.PatternRESIDENT)):
                _iPattermImg = self.readImg(self.PatternRESIDENT[i])
                print('LoadPatternIMG: [%s]' %(self.PatternRESIDENT[i]))
                if self.detectedType(inputImg, _iPattermImg) != 0:
                    self.detectTypeResult = 5
                    self.PersonalInfoType = '주민등록표'
                    # print('PersonalType Detected: %s' % (PersonalInfoType))
                    # break
        else :
            print('no matching')
            no_matching = messagebox.showerror('Error', '주민등록증, 운전면허증, 가족관계증명서, 여권, 주민등록등본 혹은 초본만 가능합니다')
            # first_py = p_1.firs()
            # first_py.root.mainloop()


        
