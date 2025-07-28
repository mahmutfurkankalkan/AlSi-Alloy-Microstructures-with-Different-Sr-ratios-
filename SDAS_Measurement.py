
import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

folder_names = next(os.walk("/content/x/"), (None, None, []))[1]
folder_names = sorted(folder_names)

area_ranges = list(range(0, 2000, 50))
area_counts = {f"{area_start}-{area_start + 50}": 0 for area_start in area_ranges}

def get_longest_edges(box):
    edges = [box[i] - box[(i + 1) % 4] for i in range(4)]
    lengths = [np.linalg.norm(edge) for edge in edges]
    longest_edge_index = np.argmax(lengths)
    opposite_edge_index = (longest_edge_index + 2) % 4
    return edges[longest_edge_index], edges[opposite_edge_index], longest_edge_index, opposite_edge_index

def get_angle(edge):
    return np.arctan2(edge[1], edge[0]) * 180.0 / np.pi

def are_parallel(angle1, angle2, tolerance=30):
    return np.abs(angle1 - angle2) < tolerance or np.abs(angle1 - angle2 - 180) < tolerance or np.abs(angle1 - angle2 + 180) < tolerance

def is_contour_between(box1, box2, other_boxes):
    center1 = np.mean(box1, axis=0)
    center2 = np.mean(box2, axis=0)
    for box in other_boxes:
        center = np.mean(box, axis=0)
        if min(center1[0], center2[0]) <= center[0] <= max(center1[0], center2[0]) and            min(center1[1], center2[1]) <= center[1] <= max(center1[1], center2[1]):
            return True
    return False

def are_opposing_edges(box1, box2, tolerance_factor=2):
    edge1, edge1_opposite, idx1, opp_idx1 = get_longest_edges(box1)
    edge2, edge2_opposite, idx2, opp_idx2 = get_longest_edges(box2)
    midpoint1 = (box1[idx1] + box1[(idx1 + 1) % 4]) / 2
    midpoint2 = (box2[idx2] + box2[(idx2 + 1) % 4]) / 2
    distance_between_midpoints = np.linalg.norm(midpoint1 - midpoint2)
    return distance_between_midpoints < tolerance_factor * np.linalg.norm(edge1) / 2 and distance_between_midpoints < tolerance_factor * np.linalg.norm(edge2) / 2

def has_opposing_edge(box1, box2, other_boxes, tolerance_factor=2):
    edge1, edge1_opposite, idx1, opp_idx1 = get_longest_edges(box1)
    edge2, edge2_opposite, idx2, opp_idx2 = get_longest_edges(box2)
    angle1 = get_angle(edge1)
    angle2 = get_angle(edge2)
    angle1_opposite = get_angle(edge1_opposite)
    angle2_opposite = get_angle(edge2_opposite)
    if not are_parallel(angle1, angle2) or not are_parallel(angle1_opposite, angle2_opposite):
        return False
    if is_contour_between(box1, box2, other_boxes):
        return False
    if not are_opposing_edges(box1, box2, tolerance_factor):
        return False
    return True

def calculate_sdas_length(center1, center2):
    return np.linalg.norm(np.array(center1) - np.array(center2))

for ri in range(len(folder_names)):
    path_to_the_folders = os.path.join("/content/deneme/", folder_names[ri])
    ima_names = next(os.walk(path_to_the_folders), (None, None, []))[2]

    for i in range(len(ima_names)):
        img = Image.open(os.path.join(path_to_the_folders, ima_names[i])).convert('L')
        img = img.resize((512, 512), Image.LANCZOS)
        test_images = np.array(img)
        test_images_scaled = test_images / 255.0

        predictions_rescaled = test_images_scaled * 255.0
        prediction_image = np.uint8(predictions_rescaled)

        prediction_image = cv2.bitwise_not(prediction_image)

        ret, thresh1 = cv2.threshold(prediction_image, 130, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        backtorgb = cv2.cvtColor(prediction_image, cv2.COLOR_GRAY2RGB)
        clist = []
        box_list = []
        center_list = []

        for l in range(len(contours)):
            carea = cv2.contourArea(contours[l])
            rect = cv2.minAreaRect(contours[l])
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            if carea > 10:
                clist.append(l)
                box_list.append(box)
                M = cv2.moments(contours[l])
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    center_list.append((cx, cy))

        for area_start in area_ranges:
            area_end = area_start + 50
            for zi in range(len(clist)):
                area = cv2.contourArea(contours[clist[zi]])
                if area_start <= area < area_end:
                    area_counts[f"{area_start}-{area_end}"] += 1

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(cv2.imread(os.path.join(path_to_the_folders, ima_names[i])), cmap='gray')
        axes[0].set_title('Original Image')
        axes[1].imshow(thresh1, cmap='gray')
        axes[1].set_title('Predicted Mask')

        parallel_contours = []
        checked_pairs = set()

        for idx1, box1 in enumerate(box_list):
            closest_dist = float('inf')
            closest_idx = None
            for idx2 in range(len(box_list)):
                if idx1 == idx2 or (clist[idx1], clist[idx2]) in checked_pairs or (clist[idx2], clist[idx1]) in checked_pairs:
                    continue
                box2 = box_list[idx2]
                if has_opposing_edge(box1, box2, [box for idx, box in enumerate(box_list) if idx not in [idx1, idx2]]):
                    center1 = center_list[idx1]
                    center2 = center_list[idx2]
                    dist = np.linalg.norm(np.array(center1) - np.array(center2))
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_idx = idx2
            if closest_idx is not None:
                parallel_contours.append((clist[idx1], clist[closest_idx]))
                checked_pairs.add((clist[idx1], clist[closest_idx]))

        image_co = backtorgb.copy()
        sdas_lengths = []
        for contour_pair in parallel_contours:
            idx1, idx2 = contour_pair
            center1 = center_list[clist.index(idx1)]
            center2 = center_list[clist.index(idx2)]
            sdas_length = calculate_sdas_length(center1, center2)
            sdas_lengths.append(sdas_length)
            print(f"SDAS Length between contour {idx1} and {idx2}: {sdas_length}")
            cv2.drawContours(image_co, contours, idx1, (0, 255, 255), 1)
            cv2.drawContours(image_co, contours, idx2, (0, 255, 255), 1)
            cv2.line(image_co, center1, center2, (0, 0, 255), 2)
            cv2.putText(image_co, f'{idx1}', center1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image_co, f'{idx2}', center2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        if sdas_lengths:
            mean_sdas = np.mean(sdas_lengths)
            std_sdas = np.std(sdas_lengths)
            print(f"Mean SDAS Length: {mean_sdas}")
            print(f"SDAS Length Standard Deviation: {std_sdas}")

        axes[2].imshow(cv2.cvtColor(image_co, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Contour of Prediction with IDs')

        if parallel_contours:
            print(f"Parallel Neighboring Contours in {ima_names[i]}: {parallel_contours}")
        else:
            print(f"No Parallel Neighboring Contours found in {ima_names[i]}")

        plt.show()
