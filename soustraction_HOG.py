import numpy as np
import cv2
import matplotlib.pyplot as plt
import NMS


def soustraction_image(fond, image, seuil):
    '''
    Methode de soustraction image en utilisant la methode d'openCV puis création d'un mask
    :param seuil
    :return:
    '''

    diffImg1 = cv2.subtract(fond, image) #utilisation de la méthode d'openCV
    diffImg1 = cv2.cvtColor(diffImg1, cv2.COLOR_BGR2GRAY) #conversion en noir et blancs

    #Création d'un mask
    height, width = diffImg1.shape
    mask = np.zeros([height, width], np.uint8)
    for y in range(height):
        for x in range(width):
            if diffImg1[y][x] > seuil:
                mask[y][x] = 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=3)
    #cv2.imshow('subtract(img1,img2) with mask', mask)
    return mask


def superposition(background, overlay, x, y):
    '''
    Fonction qui superpose l'overlay à la position x,y sur le fond
    :param background:
    :param overlay:
    :param x:
    :param y:
    :return:
    '''
    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background



def image_de_fond():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()  # return a single frame in variable `frame`
    cap.release()
    return frame



def HOG(fond):
    # initialisation du HOG:
    cv2.imshow('fond', fond)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cv2.startWindowThread()

    # ouverture du flux vidéo de la webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # la sortie sera écrite dans le fichier output.avi
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 15., (640, 480))

    while (True):
        # capture image par image
        ret, frame = cap.read()

        # réduction de l'image pour une détection plus rapide
        frame = cv2.resize(frame, (640, 480))
        # passage en noir et blanc, également pour accélerer
        # la détection
        #print("frame.shape", frame.shape)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # détection des personnes dans l'image.
        # retourne les coordonnées de la boîte encadrant
        # les personnes détectées
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        boxes = NMS.non_max_suppression_fast(boxes, 0.3) #overlapThresh entre 0.3 et 0.5

        for (xA, yA, xB, yB) in boxes:
            #print("xA, xB, yA, yB : ", xA, xB, yA, yB)
            # affichages des boîtes sur l'image couleur
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

            #création d'une nouvelle image contenant uniquement le rectangle de l'image détecté
            background = np.zeros_like(frame, dtype=np.uint8)
            overlay = frame[yA:yB, xA:xB]
            zone_fond = fond[yA:yB, xA:xB]
            extrait_rectangle = superposition(background, overlay, xA, yA)
            #cv2.imshow('rectangle de detection', extrait_rectangle)
            background = np.zeros_like(frame, dtype=np.uint8)
            extrait_fond = superposition(background, zone_fond, xA, yA)
            #cv2.imshow('fond dans le rectangle', extrait_fond)
            inpainting = superposition(frame, zone_fond, xA, yA)
            cv2.imshow('essai inpainting', inpainting)
            soustraction = soustraction_image(extrait_fond, extrait_rectangle, 10)
            #cv2.imshow('Essai de soustraction ', soustraction)
        # écriture de la vidéo avec les boîtes
        out.write(frame.astype('uint8'))
        # affichage de l'image résultante
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # quand on a terminé:
    # on termine la capture
    cap.release()
    # on termine l'écriture
    out.release()
    # et on ferme la fenêtre
    cv2.destroyAllWindows()
    cv2.waitKey(1)



#Execution
image_fond = image_de_fond()
print('Image de fond capturée')
print('Execution HOG(image_fond)')
HOG(image_fond)