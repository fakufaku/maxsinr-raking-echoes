import numpy as np


def get_wall_order_from_images(src_images, mic_pos, room_dim, tolerance=1e-6):
    walls = {
        'west': {'normal': np.array([1, 0, 0]), 'point': np.array([0, 0, 0])},
        'east': {'normal': np.array([-1, 0, 0]), 'point': np.array([room_dim[0], 0, 0])},
        'south': {'normal': np.array([0, 1, 0]), 'point': np.array([0, 0, 0])},
        'north': {'normal': np.array([0, -1, 0]), 'point': np.array([0, room_dim[1], 0])},
        'floor': {'normal': np.array([0, 0, 1]), 'point': np.array([0, 0, 0])},
        'ceil': {'normal': np.array([0, 0, -1]), 'point': np.array([0, 0, room_dim[2]])},
    }

    assert src_images.shape[0] == 3
    assert mic_pos.shape[0] == 3
    assert len(mic_pos.shape) == 1, "for now only one mic is supported"

    def line_plane_intersection(mic_pos, img_pos, wall, tolerance):
        # Direction vector of the line (from mic to image source)
        line_dir = img_pos - mic_pos
        # Point on the plane
        plane_point = wall['point']
        # Normal vector to the plane
        plane_normal = wall['normal']
        # Check if line is parallel to the plane
        denom = np.dot(plane_normal, line_dir)
        if np.abs(denom) < tolerance:  # Consider parallel if within tolerance
            return None
        # Find the intersection point
        t = np.dot(plane_normal, plane_point - mic_pos) / denom
        if t < -tolerance:  # Ignore negative t values with a margin of error
            return None
        intersection_point = mic_pos + t * line_dir
        # Check if the intersection point lies within the room dimensions with tolerance
        if (
            -tolerance <= intersection_point[0] <= room_dim[0] + tolerance and
            -tolerance <= intersection_point[1] <= room_dim[1] + tolerance and
            -tolerance <= intersection_point[2] <= room_dim[2] + tolerance
        ):
            return wall
        return None

    results = []
    for n, img in enumerate(src_images.T):
        found_wall = None
        if n > 0: # first image is the direct path
            for wall_name, wall in walls.items():
                intersecting_wall = line_plane_intersection(mic_pos, img, wall, tolerance)
                print(f"IMG {img}: checking {wall_name} -> {intersecting_wall}")
                if intersecting_wall:
                    found_wall = wall_name
                    break
        results.append(found_wall if found_wall else 'direct')
    
    return results


if __name__ == "__main__":
    import pyroomacoustics as pra
    import matplotlib.pyplot as plt
    
    room_dim = [5, 4, 3]
    mic_pos = np.array([[2.5, 2, 1.5]]).T
    src_pos = np.array([[1, 1, 1]]).T
    tolerance = 1e-6
    
    # create a pyroomacoustics room
    room = pra.ShoeBox(room_dim, fs=16000, max_order=1)
    room.add_microphone_array(pra.MicrophoneArray(mic_pos, room.fs))
    room.add_source(src_pos)
    
    # compute the image sources
    room.image_source_model()
    
    src_images = room.sources[0].images
    print('Images shape', src_images.shape)
    
    walls = get_wall_order_from_images(src_images, mic_pos[:,0], room_dim, tolerance)
    print(walls)
    
    room.plot(img_order=1)
    plt.show()