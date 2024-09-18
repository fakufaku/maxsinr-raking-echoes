import numpy as np
import scipy.optimize
import functools
from tqdm import tqdm
from sympy import Point3D, Plane, Ray3D

def compute_planes(room_dimension):
    room_dimension = np.array(room_dimension)
    if not len(room_dimension) == 3:
        raise ValueError('This is hard coded for Shoebox')
    L, W, H = room_dimension
    planes = {
        'd': None,
        'c': sym_plane_from_points(
                np.transpose(np.array([[0, 0, H], [L, 0, H], [0, W, H], [L, W, H]]))),
        'f': sym_plane_from_points(
                np.transpose(np.array([[0, 0, 0], [L, 0, 0], [0, W, 0], [L, W, 0]]))),
        'w': sym_plane_from_points(
                np.transpose(np.array([[0, 0, 0], [0, W, 0], [0, 0, H], [0, W, H]]))),
        's': sym_plane_from_points(
                np.transpose(np.array([[0, 0, 0], [L, 0, 0], [0, 0, H], [L, 0, H]]))),
        'e': sym_plane_from_points(
                np.transpose(np.array([[L, 0, 0], [L, W, 0], [L, 0, H], [L, W, H]]))),
        'n': sym_plane_from_points(
                np.transpose(np.array([[0, W, 0], [L, W, 0], [0, W, H], [L, W, H]]))),
    }
    return planes

def get_point(x):
    return Point3D(list(x))

def compute_image(x, P):
    pnt = get_point(x)
    prj = P.projection(pnt)
    img = pnt + 2*Ray3D(pnt, prj).direction
    return img


def sym_plane_from_points(points):
    D, N = points.shape
    assert D == 3
    assert N > 3
    # from numpy to Point3D
    pts = [get_point(points[:,n]) for n in range(N)]

    if not Point3D.are_coplanar(*pts):
        raise ValueError('Not complanar')

    pl = Plane(*pts[:3])

    return pl

def plane_from_points(points):
    '''
    regression plane from stackexchange
    https://stackoverflow.com/a/20700063/7580944
    '''

    D, N = points.shape
    assert D == 3
    assert N > 3

    def plane(x, y, params):
        a = params[0]
        b = params[1]
        c = params[2]
        z = a*x + b*y + c
        return z

    def error(params, points):
        result = 0
        for n in range(N):
            [x, y, z] = points[:, n]
            plane_z = plane(x, y, params)
            diff = abs(plane_z - z)
            result += diff**2
        return result

    fun = functools.partial(error, points=points)
    params0 = [0, 0, 0]
    res = scipy.optimize.minimize(fun, params0)

    a = res.x[0]
    b = res.x[1]
    c = res.x[2]

    plane = np.array([a, b, c])

    return plane

def get_wall_order_from_images(src_images, mic_pos, room_dim, tolerance=1e-6):
    walls = {
        'west': {'normal': np.array([1, 0, 0]), 'point': np.array([0, 0, 0])},
        'east': {'normal': np.array([-1, 0, 0]), 'point': np.array([room_dim[0], 0, 0])},
        'south': {'normal': np.array([0, 1, 0]), 'point': np.array([0, 0, 0])},
        'north': {'normal': np.array([0, -1, 0]), 'point': np.array([0, room_dim[1], 0])},
        'floor': {'normal': np.array([0, 0, 1]), 'point': np.array([0, 0, 0])},
        'ceil': {'normal': np.array([0, 0, -1]), 'point': np.array([0, 0, room_dim[2]])},
    }

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
                # print(f"IMG {img}: checking {wall_name} -> {intersecting_wall}")
                if intersecting_wall:
                    found_wall = wall_name
                    break
        results.append(found_wall if found_wall else 'direct')
    
    return results

    # img0 = images[:, np.where(order == 0)].squeeze()
    # source_position = img0
    # D, K = images.shape
    
    # walls_generating_reflections = []
    # num_sources = images.shape[1]  # Number of image sources

    # reflection_walls = []
    # for i in range(num_sources):
    #     img = images[:, i]
    #     if np.allclose(img, source_position):
    #         reflection_walls.append('direct')

    #     # Check reflection on the x-axis (left and right walls)
    #     elif img[0] < 0:
    #         reflection_walls.append('west')
    #     elif img[0] > room_dim[0]:
    #         reflection_walls.append('est')

    #     # Check reflection on the y-axis (front and back walls)
    #     elif img[1] < 0:
    #         reflection_walls.append('south')
    #     elif img[1] > room_dim[1]:
    #         reflection_walls.append('north')

    #     # Check reflection on the z-axis (floor and ceiling)
    #     elif img[2] < 0:
    #         reflection_walls.append('floor')
    #     elif img[2] > room_dim[2]:
    #         reflection_walls.append('ceil')
        
    #     else:
    #         raise ValueError('No wall found for image source')
        
    #     # print(f"Image source # {i} at {img} is generated {reflection_walls[-1]}")
            
    # walls_generating_reflections = reflection_walls

    # return walls_generating_reflections    
 
    return walls

def dist_point_plane(point, plane_point, plane_normal):
    x, y, z = plane_point
    a, b, c = plane_normal
    d = - (a*x + b*y + c*z)
    q, r, s = point
    dist = np.abs(a*q + b*r + c*s + d) / np.sqrt(a**2 + b**2 + c**2)
    return dist


def mesh_from_plane(point, normal):

    def cross(a, b):
        return [a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0]]

    a, b, c = normal

    normal = np.array(cross([1, 0, a], [0, 1, b]))
    d = -point.dot(normal)
    xx, yy = np.meshgrid([-5, 5], [-5, 5])
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

    return xx, yy, z


def square_within_plane(center, normal, size=(5,5)):
    a, b = size[0]/2, size[1]/2
    B = np.array([[-a, -b, 0],
                  [ a, -b, 0],
                  [ a,  b, 0],
                  [-a,  b, 0]]).T # 3xN convention
    assert B.shape == (3, 4)
    # find the rotation matrix that bring [0,0,1] to the input normal
    a = np.array([0, 0, 1])
    b = normal

    R = rotation_matrix(a, b)
    # apply rotation
    B = R @ B
    assert np.allclose(np.mean(B, -1).sum(), 0)
    # translate
    B = B + center[:, None]
    return B

def distance(x, y):
    """
    Computes the distance matrix E.
    E[i,j] = sqrt(sum((x[:,i]-y[:,j])**2)).
    x and y are DxN ndarray containing N D-dimensional vectors.
    """
    assert len(x.shape) == len(y.shape) == 2

    return np.sqrt(np.sum((x[:, :, np.newaxis] - y[:, np.newaxis, :]) ** 2, axis=0))

def rotation_matrix(a, b):
    # https://math.stackexchange.com/questions/180418

    assert a.shape == b.shape
    assert len(a) == len(b) == 3

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    c = np.dot(a, b)

    if np.allclose(c, 1) or np.allclose(c, -1):
        R = np.eye(3)

    else:
        v = np.cross(a, b)
        s = np.linalg.norm(v)
        assert np.allclose(1 - c**2,  s**2)

        sk_v = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

        skw_v2 = np.dot(sk_v, sk_v)
        R = np.eye(3) + sk_v + skw_v2 / (1 - c)

    return R

if __name__ == "__main__":
    a = np.array([0, 1, 0])
    n = np.array([0, 1, 0])
    R = square_within_plane(a, n, size=(5, 5))
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = Axes3D(fig)
    verts = [list(zip(R[0, :], R[1, :], R[2, :]))]
    ax.add_collection3d(Poly3DCollection(verts))
    plt.show()
    pass
