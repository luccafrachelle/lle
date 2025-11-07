import numpy as np
from sklearn.utils.validation import check_random_state

def make_swiss_roll(n_samples=100, *, noise=0.0, random_state=None, hole=False):
    generator = check_random_state(random_state)

    if not hole:
        t = 1.5 * np.pi * (1 + 2 * generator.uniform(size=n_samples))
        y = 21 * generator.uniform(size=n_samples)
    else:
        corners = np.array(
            [[np.pi * (1.5 + i), j * 7] for i in range(3) for j in range(3)]
        )
        corners = np.delete(corners, 4, axis=0)
        corner_index = generator.choice(8, n_samples)
        parameters = generator.uniform(size=(2, n_samples)) * np.array([[np.pi], [7]])
        t, y = corners[corner_index].T + parameters

    x = t * np.cos(t)
    z = t * np.sin(t)

    X = np.vstack((x, y, z))
    X += noise * generator.standard_normal(size=(3, n_samples))
    X = X.T
    t = np.squeeze(t)

    return X, t

def sample_torus(R, r, num_samples):
    
    theta = np.random.uniform(0, 2 * np.pi, num_samples)
    phi = np.random.uniform(0, 2 * np.pi, num_samples)
    radius_section = R + r * np.cos(theta)
    x_coords = radius_section * np.cos(phi)
    y_coords = radius_section * np.sin(phi)
    z_coords = r * np.sin(theta)
    color_feature = phi / (2 * np.pi) 
    points = np.stack((x_coords, y_coords, z_coords, color_feature), axis=-1)

    return points

def make_s_curve(num_samples, scale_factor=3.0, twist_factor=1.0):
    
    t = np.random.uniform(-1, 1, num_samples) * scale_factor  
    h = np.random.uniform(-1, 1, num_samples) * 0.5 
    
    x_coords = np.sin(t) * scale_factor
    y_coords = h
    z_coords = np.sign(t) * (np.cos(t) - 1) * scale_factor + h * twist_factor
    
    color_feature = (t - t.min()) / (t.max() - t.min()) 
    
    points = np.stack((x_coords, y_coords, z_coords, color_feature), axis=-1)

    return points