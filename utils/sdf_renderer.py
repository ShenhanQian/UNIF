import torch


def sphere_tracing(
    signed_distance_function, 
    positions, 
    directions, 
    num_iterations=1000, 
    convergence_threshold=1e-3,
    filter_threshold=1.0,
    priors=None,
):
    assert filter_threshold > convergence_threshold

    for i in range(num_iterations):
        if i:
            valid = torch.logical_and(signed_distances < filter_threshold, ~converged)
            if valid.sum() == 0:
                break

            valid_pos = positions[valid.repeat_interleave(3, dim=-1)].reshape(-1, positions.shape[-1])
            signed_distances[valid] = signed_distance_function(valid_pos.unsqueeze(0), **priors).flatten()
            positions = torch.where(converged, positions, positions + directions * signed_distances)
        else:
            signed_distances = signed_distance_function(positions.unsqueeze(0), **priors).squeeze(0)
            positions = positions + directions * signed_distances

        converged = torch.abs(signed_distances) < convergence_threshold
        
        if torch.all(converged):
            break
    return positions, converged

def phong_shading(positions, normals, textures, cameras, lights, materials):
    light_diffuse_color = lights.diffuse(
        normals=normals, 
        points=positions,
    )
    light_specular_color = lights.specular(
        normals=normals,
        points=positions,
        camera_position=cameras.get_camera_center(),
        shininess=materials.shininess,
    )
    ambient_colors = materials.ambient_color * lights.ambient_color
    diffuse_colors = materials.diffuse_color * light_diffuse_color
    specular_colors = materials.specular_color * light_specular_color
    assert diffuse_colors.shape == specular_colors.shape
    ambient_colors = ambient_colors.reshape(-1, *[1] * len(diffuse_colors.shape[1:-1]), 3)
    # colors = (ambient_colors + diffuse_colors) * textures + specular_colors
    colors = ambient_colors + diffuse_colors * textures + specular_colors
    return colors