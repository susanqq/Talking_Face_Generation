import cv2
import dlib
import os
import sys
import glob
import numpy as np



PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 68))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (RIGHT_EYE_POINTS + RIGHT_BROW_POINTS + LEFT_EYE_POINTS + LEFT_BROW_POINTS +
                             NOSE_POINTS + MOUTH_POINTS )

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

REFERENCE_PATH="average_portrait/portraits/01a4462309f79052d1a480170ef3d7ca7bcbd564.jpg"

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def if_one_face(im):
    rects = detector(im, 1)
    if len(rects) > 1:
        return 2
    if len(rects) == 0:
        return 0
    return 1


def get_landmarks(im, bbox_gt):
    rects = detector(im, 0)

    # max_area = 0
    min_dist = 10000
    for i in range(len(rects)):
        # if dlib.rectangle.area(rects[i]) > max_area:
        #     max_area = dlib.rectangle.area(rects[i])
        #     max_rect = rects[i]

        bbox_dlib = np.array([rects[i].left(), rects[i].top(), rects[i].right(), rects[i].bottom()])
        dist = np.linalg.norm(bbox_gt-bbox_dlib)

        if dist < min_dist:
            min_dist = dist
            max_rect = rects[i]

    if min_dist > 200:
        return None

    # if len(rects) > 1:
    #     raise TooManyFaces
    if len(rects) == 0:
        return None
    s = np.matrix([[p.x, p.y] for p in predictor(im, max_rect).parts()])

    return s

def get_landmarks_ref(im):
    rects = detector(im, 0)

    max_area = 0
    min_dist = 10000
    for i in range(len(rects)):
        if dlib.rectangle.area(rects[i]) > max_area:
            max_area = dlib.rectangle.area(rects[i])
            max_rect = rects[i]

    # print(max_rect.left(), max_rect.top(), max_rect.right(), max_rect.bottom())

    # if len(rects) > 1:
    #     raise TooManyFaces
    if len(rects) == 0:
        return None
    s = np.matrix([[p.x, p.y] for p in predictor(im, max_rect).parts()])

    return s


def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T
    M = np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])

    return M



def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def warp_lip_landmark(M,lip_landmark):
    allone = np.ones((lip_landmark.shape[0],1), dtype=np.float32)

    lip_landmark_3d = np.column_stack((lip_landmark, allone))
    lip_region = np.around(np.matmul(M, lip_landmark_3d.T)[:2]).astype(int)
    list_lip_region= lip_region.T.tolist()
    return list_lip_region


def align_im_to_ref(img, img_ref, landmark, landmark_ref, is_only_front=True, is_only_lip=True):
    M = transformation_from_points(landmark_ref[ALIGN_POINTS],
                                   landmark[ALIGN_POINTS])

    M_2 = transformation_from_points(landmark[ALIGN_POINTS],landmark_ref[ALIGN_POINTS])

    warped_im = warp_im(img, M, img_ref.shape)
    c1=255
    r1=209
    # roi = warped_im[c1:c1+109,r1:r1+109]

    lip_landmark = warp_lip_landmark(M_2,landmark[MOUTH_POINTS])
    warp_landmark = warp_lip_landmark(M_2,landmark)

    # # draw landmark on face
    # for i in range(len(warp_landmark)):
    #    pos = tuple(warp_landmark[i])
    #    cv2.circle(warped_im, pos, 2, (255,255,255), -1)

    warped_face = warped_im[100:400, 115:415]
    mouth_roi = warped_im[255:364,209:318]
    # warped_face = warped_im[131:449, 332:670]

    # for i in range(len(landmark)):
    #    cv2.circle(img, (landmark[i][0,0],landmark[i][0,1]) , 2, (255,255,255), -1)
    # warped_face = img

    if is_only_lip:
        return_landmark = lip_landmark
    else:
        return_landmark = warp_landmark

    if is_only_front:
        #calculate the ratio between dist(40-28)/dist(43-28)
        dist_p40_p28 = np.linalg.norm(landmark[39]-landmark[27])
        dist_p43_p28 = np.linalg.norm(landmark[42]-landmark[27])
        eye_ratio = float(dist_p40_p28/dist_p43_p28)
        # dist_p65_p66 = np.linalg.norm(landmark[64]-landmark[65])
        # dist_p61_p68 = np.linalg.norm(landmark[60]-landmark[67])
        # lip_ratio = float(dist_p61_p68/dist_p65_p66)

        print('the distance eyeration is', eye_ratio)
        if 0.8 < eye_ratio and eye_ratio < 1.25:
            return warped_face, return_landmark
        else:
            return None, None
    else:
        print('the wrap face and whole face landmark')
        return warped_face, mouth_roi, return_landmark




def align_ref_to_im(img, img_ref, landmark, landmark_ref):
    M = transformation_from_points(landmark_ref[ALIGN_POINTS],
                                   landmark[ALIGN_POINTS])

    M_2 = transformation_from_points(landmark[ALIGN_POINTS],landmark_ref[ALIGN_POINTS])

    ref_mouth_coordinates = np.matrix([[209,255],[318,364]])
    src_mouth_coordinates = warp_lip_landmark(M, ref_mouth_coordinates)

    return src_mouth_coordinates


def get_mouth_region(img, landmark):
    landmark = np.array(landmark)
    left_x, left_y = landmark[48][0], landmark[48][1]
    right_x, right_y = landmark[54][0], landmark[54][1]

    height = width = 1.5*(right_x - left_x)
    print(height)
    mid_x = (left_x+right_x)/2.0
    mid_y = (left_y+right_y)/2.0

    roi_x1, roi_x2, roi_y1, roi_y2 =  mid_x-0.5*width, mid_x+0.5*width, mid_y-0.62*height, mid_y+0.38*height
    # return img[roi_y1:roi_y2, roi_x1:roi_x2]
    return roi_x1, roi_x2, roi_y1, roi_y2


def save_landmark_img(lip_landmark, img_shape, offset, filename, radius=2):
    imagename =filename
    img = np.zeros(img_shape)
    if offset is None:
        offset = [115,100]
    draw_landmark(img,lip_landmark,offset,imagename,radius)


def draw_landmark(img,landmarks, offset,imagename,radius=2):
    for i in range(len(landmarks)):
        # ref_landmark = lip_landmark[i]
        ref_landmark = (landmarks[i][0]-offset[0],landmarks[i][1]-offset[1])
        pos = tuple(ref_landmark)
        cv2.circle(img, pos, radius, (255,255,255), -1)
    # print('saving image to', imagename)
    cv2.imwrite(imagename, img)



def save_landmark_pos(landmark, offset, filename):
    # print('saving mouth region landmark into filename', filename)
    with open(filename,'w') as f:
        for i in range(len(landmark)):
            # it is writen as the image coordination space
            if offset is not None:
                ref_landmark = str(landmark[i][0]-offset[0]) +','+ str(landmark[i][1]-offset[1])+'\n'
            else:
                ref_landmark = str(landmark[i][0]-115) +','+ str(landmark[i][1]-100)+'\n'
            f.writelines(ref_landmark)
        f.close()


def drawlanmark(img, landmark):
    landmark_list = np.around(landmark).astype(int).tolist()
    for i in range(len(landmark_list)):
        pos = tuple(landmark_list[i])
        print(pos)
        cv2.circle(img, pos, 1, (0,255,0), -1)
    cv2.imwrite("square_circle_opencv.jpg", img)