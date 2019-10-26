import cv2
import numpy as np
import math
import camera as camera
import plot_helper as plot_helper
import matplotlib.pyplot as plt
from pyexcel_ods import get_data
from itertools import cycle

#https://github.com/eborboihuc/rotate_3d
#https://python-projective-camera-model.readthedocs.io/en/latest/api.html

def swap_rows(input_matrix, ix, iy, iz):
    swap_matrix = input_matrix.copy()
    swap_matrix[0, :] = input_matrix[ix, :]
    swap_matrix[1, :] = input_matrix[iy, :]
    swap_matrix[2, :] = input_matrix[iz, :]
    return swap_matrix

def get_2d_points(array):
    array_2d = []
    for point in array:
        array_2d.append(point[0:2])
    return np.array(array_2d, np.float32)

def set_roi( stitch_image, roi_image, all_height, all_width, image_height, height_offset):
    hA = int(all_height - height_offset - image_height)
    wA = int(0)
    hB = int(all_height - height_offset)
    wB = int(all_width)
    stitch_image[hA:hB, wA:wB] += roi_image

def line_intersection(x1_1, x1_2, y1_1, y1_2, x2_1, x2_2, y2_1, y2_2):
    A1 = y1_1 - y1_2
    B1 = x1_2 - x1_1
    C1 = x1_1 * y1_2 - x1_2 * y1_1
    A2 = y2_1 - y2_2
    B2 = x2_2 - x2_1
    C2 = x2_1 * y2_2 - x2_2 * y2_1

    x = None
    y = None
    if B1 * A2 - B2 * A1 and A1:
        y = (C2 * A1 - C1 * A2) / (B1 * A2 - B2 * A1)
        x = (-C1 - B1 * y) / A1
    elif B1 * A2 - B2 * A1 and A2:
        y = (C2 * A1 - C1 * A2) / (B1 * A2 - B2 * A1)
        x = (-C2 - B2 * y) / A2

    return np.array([x,y])

data = get_data("dataset/metadata-v1_manual.ods")

# Read Image
items = list(data.items())

input_rt = []
org_img = []
input_avg_rt = np.zeros((4,4))

index = 1
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

while index < len(items[0][1]):
    if(items[0][1][index]):
        img_path = "dataset/snapshots/"+items[0][1][index][0]
        im = cv2.imread(img_path)
        org_img.append(im)
        size = im.shape

        w = size[1]
        h = size[0]

        s = items[0][1][index][1]
        s = s.replace('[', '')
        s = s.replace(']', '')
        s = s.replace('\n', ' ')

        rt_matrix = np.fromstring(s, sep=' ').reshape(4,4)
        input_rt.append(rt_matrix)
        input_avg_rt += rt_matrix

    index+=1

input_avg_rt = np.array( ( 1 / len( input_rt ) ) * input_avg_rt )
print("Input avg rt:\n {0}".format(input_avg_rt))

ix = np.argmin(np.fabs(input_avg_rt[0:3,3]),axis=0)
iy = np.argmax(np.fabs(input_avg_rt[0:3,3]),axis=0)
iz = 3 - ( ix + iy )

avg_rt = swap_rows(input_avg_rt, ix, iy, iz)
print("Avg rt:\n {0}".format(avg_rt))

index = 0
max_z = input_rt[index][iz,3]
input_rt[index] = swap_rows(input_rt[index], ix, iy, iz)
print("rt({0}):\n {1}".format(index,input_rt[index]))

index = 1
while index < len(input_rt):
    max_z = np.min((max_z, input_rt[index][iz,3]))
    input_rt[index] = swap_rows(input_rt[index], ix, iy, iz)
    print("rt({0}):\n {1}".format(index,input_rt[index]))
    index += 1

print("Max z:\n {0}".format(max_z))

avg_rt_sign = np.sign(avg_rt)
basis = np.zeros(avg_rt.shape)

index = 0
while index < avg_rt.shape[1]:
    basis[index,index] = avg_rt_sign[index,index]
    index += 1

print("Basis:\n {0}".format(basis))

fov_x = 38.2 * math.pi / 180
fov_y = 29.1 * math.pi / 180
c_x = w / 2
c_y = h / 2

f_x = w * math.tan(fov_x)
f_y = h * math.tan(fov_y)

camera_views = []
output_camera_views = []
max_tz = 0

index = 0
while index < len(input_rt):
    R = np.array(input_rt[index][0:3, 0:3])
    cam_pos = np.array(input_rt[index][0:3, [3]])
    t = -R.dot(cam_pos)

    if index == 0:
        max_tz = t[2]
    max_tz = np.max((max_tz,t[2]))

    print("\nCamera: {0}".format(index))
    print("Cam possition:\n {0}".format(cam_pos))
    print("R:\n {0}".format(R))
    print("t:\n {0}".format(t))

    input_cam = camera.Camera(index)
    input_cam.set_K_elements(u0_px=c_x, v0_px=c_y, fx=f_x,fy=f_x)
    input_cam.set_R(R)
    input_cam.set_t(t)
    camera_views.append(input_cam)

    index += 1

scena = plot_helper.prepare_plot("Scena")

cycol = cycle('bgrcmk')

image_center = np.array([[w/2, h/2]]).T
image_points = [np.array([[0., 0]]).T, np.array([[0., h - 1]]).T, np.array([[w - 1, h - 1]]).T, np.array([[w - 1, 0]]).T]

for cam in camera_views:
    world_center = cam.image_to_world(image_center, z=0)

    world_points = []
    output_points = []
    for point in image_points:
        world_points.append(cam.image_to_world(point, z=0))

    input_cam_pos = -cam.R.T.dot(cam.t)

    x = world_center[0,0]
    y = world_center[1,0]
    z = world_center[2,0]

    output_cam_pos = np.array([[avg_rt[0,3]],[y],[-max_z]])
    output_t = -np.eye(3).dot(output_cam_pos)
    output_t = np.array(output_t)


    R = np.array([[-1, +0, +0],
                  [+0, +1, +0],
                  [+0, +0, -1]])

    output_cam = camera.Camera(cam.id)
    output_cam.set_K(cam.K)
    output_cam.set_R(R)
    output_cam.set_t(output_t)


    for point in world_points:
        output_points.append(output_cam.world_to_image(point))

    output_cam.M = cv2.getPerspectiveTransform(get_2d_points(image_points), get_2d_points(output_points))
    output_cam.world_center = output_cam_pos
    output_cam.world_points = world_points
    output_cam.output_points = output_points
    output_camera_views.append(output_cam)

    line_color = next(cycol)
    plot_helper.plot_camera(scena, input_cam_pos, cam.R, 5)
    plot_helper.plot_point(scena, world_center, line_color)
    plot_helper.plot_point(scena,input_cam_pos, line_color)
    plot_helper.plot_line(scena, world_center, input_cam_pos, line_color,':')

    plot_helper.plot_camera(scena, output_cam_pos, output_cam.R, 5)
    plot_helper.plot_line(scena, world_center, output_cam_pos, line_color, ':')
    plot_helper.plot_point(scena, output_cam_pos, line_color)

    plot_helper.plot_line(scena, world_points[0], world_points[1], line_color, '-')
    plot_helper.plot_line(scena, world_points[1], world_points[2], line_color, '-')
    plot_helper.plot_line(scena, world_points[2], world_points[3], line_color, '-')
    plot_helper.plot_line(scena, world_points[3], world_points[0], line_color, '-')

i = 1
while i < len(output_camera_views):
    j = i
    while j > 0 and output_camera_views[j-1].world_center[1,0] > output_camera_views[j].world_center[1,0]:
        cam = output_camera_views[j-1].copy()
        output_camera_views[j - 1] = output_camera_views[j].copy()
        output_camera_views[j] = cam
        j -= 1
    i += 1

height = 0
max_w = w
max_h = h
index = 0
while index < len(output_camera_views):
    if index == 0:
        cam1 = output_camera_views[index]
    else:
        cam1 = output_camera_views[index-1]

    cam2 = output_camera_views[index]

    poly_points = np.array(((cam2.output_points[0][0:2]),
                            (cam2.output_points[1][0:2]),
                            (cam2.output_points[2][0:2]),
                            (cam2.output_points[3][0:2])), dtype=int)

    px, py, pw, ph = cv2.boundingRect(poly_points)
    max_w = np.max((max_w, pw + px))
    max_h = np.max((max_h, ph + py))

    cam11_center = cam1.world_to_image(cam1.world_center)
    cam12_center = cam1.world_to_image(cam2.world_center)
    cam21_center = cam2.world_to_image(cam1.world_center)

    # height += cam11_center[1] - cam12_center[1]
    height += cam12_center[1]
    # height += cam21_center[1]
    index += 1

height = int(np.round(height + max_h))

print("Height {0}".format(height))

stitch_image_all = np.zeros((height,max_w,3), np.uint8)
stitch_image_poly = np.zeros((height,max_w,3), np.uint8)

height_offset = 0
index = 0
while index < len(output_camera_views):

    if index == 0:
        cam1 = output_camera_views[index]
    else:
        cam1 = output_camera_views[index-1]

    cam0 = output_camera_views[0]
    cam2 = output_camera_views[index]

    cam11_center = cam1.world_to_image(cam1.world_center)
    cam12_center = cam1.world_to_image(cam2.world_center)
    cam21_center = cam2.world_to_image(cam1.world_center)
    cam00_center = cam0.world_to_image(cam0.world_center)
    cam20_center = cam2.world_to_image(cam0.world_center)

    # height_offset += int(cam11_center[1] - cam12_center[1])
    height_offset += int(cam12_center[1])
    # height_offset += int(cam21_center[1])
    points_height_offset = height - height_offset - max_h

    print("Stitch {0}\n{1} {2} {3} {4} {5}".format(index, cam11_center[1], cam12_center[1], cam21_center[1], cam11_center[1] - cam12_center[1], cam11_center[1] - cam21_center[1]))

    poly_points = np.array(((cam2.output_points[0][0:2]),
                            (cam2.output_points[1][0:2]),
                            (cam2.output_points[2][0:2]),
                            (cam2.output_points[3][0:2])), dtype=int)

    i = 0
    while i < len(poly_points):
        poly_points[i][1] += points_height_offset
        i += 1

    lineThickness = 5

    cv2.line(stitch_image_poly, (poly_points[0][0], poly_points[0][1]), (poly_points[1][0], poly_points[1][1]), (0, 255, 0), lineThickness)
    cv2.line(stitch_image_poly, (poly_points[1][0], poly_points[1][1]), (poly_points[2][0], poly_points[2][1]), (0, 255, 0), lineThickness)
    cv2.line(stitch_image_poly, (poly_points[2][0], poly_points[2][1]), (poly_points[3][0], poly_points[3][1]), (0, 255, 0), lineThickness)
    cv2.line(stitch_image_poly, (poly_points[3][0], poly_points[3][1]), (poly_points[0][0], poly_points[0][1]), (0, 255, 0), lineThickness)

    im = cv2.warpPerspective(org_img[cam2.id], cam2.M, (max_w, max_h), borderValue=0)
    cv2.imwrite("out/{0}.jpg".format(cam2.id), im)

    cv2.fillConvexPoly(stitch_image_all, poly_points, (0, 0, 0))
    set_roi(stitch_image_all, im, height, max_w, max_h, height_offset)

    '''
    resize = 0.5
    stitch_image = np.zeros((height, w, 3), np.uint8)
    set_roi(stitch_image, im, height, w, h, height_offset)
    stitch_image = cv2.resize(stitch_image, None, fx=resize, fy=resize)
    cv2.imwrite("out/stitch{0}.jpg".format(index), stitch_image)
    '''
    index += 1

resize = 0.50
stitch_image_all = cv2.resize(stitch_image_all, None, fx=resize, fy=resize)
stitch_image_poly = cv2.resize(stitch_image_poly, None, fx=resize, fy=resize)
cv2.imwrite("out/stitch_image_all.jpg", stitch_image_all)
cv2.imwrite("out/stitch_image_poly.jpg", stitch_image_poly)

plot_helper.set_axes_equal(scena)
plt.show()