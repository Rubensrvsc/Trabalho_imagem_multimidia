import numpy as np
import cv2 as cv
import matplotlib as mtl

def escreve(img, texto, cor=(255,0,0)):
    fonte = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img, texto, (10,20), fonte, 0.5, cor, 0, cv.LINE_AA)

def main():
    im = 'dados.PNG'
    ler_img=cv.imread(im)
    im_gray = cv.cvtColor(ler_img,cv.COLOR_BGR2GRAY)
    #cv.imshow("Imagem em cinza: ",im_gray) #tons de cinza
    bl = cv.blur(im_gray,(5,5))
    #cv.imshow("Resultado: ",bl) #filtro blur

    (T, bin) = cv.threshold(bl,160,255,cv.THRESH_BINARY)
    #(T, binI) = cv.threshold(bl, 160, 255, cv.THRESH_BINARY_INV)
    #cv.imshow("Resultado: ",bin) #binarização

    sobelX = cv.Sobel(bin, cv.CV_64F, 1, 0)
    sobelY = cv.Sobel(bin, cv.CV_64F, 0, 1)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))
    sobel = cv.bitwise_or(sobelX, sobelY)

    contours, hierarchy = cv.findContours(bin, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    #cv.imshow("Resultado: ",ler_img) #detecção de bordas
    cv.drawContours(ler_img, contours, -1, (255, 0, 0), 2)
    #escreve(ler_img, str(len(contours))+" objetos encontrados!")
    
    #print(str(len(contours)))
    cv.imshow("Total de contornos: "+str(len(contours)),ler_img) #qtd de objetos encontrados
    ret= cv.waitKey(0)

if __name__=='__main__':
    main()
