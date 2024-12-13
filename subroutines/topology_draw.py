import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap

def display_3d_torus_with_selected_nodes(grid_shape, selected_nodes, link_usage = None, name = "", quantile = 0.5):
    """
    Display a 3D grid and highlight selected nodes using Plotly.

    :param grid_shape: Tuple (x_size, y_size, z_size) specifying the grid dimensions.
    :param selected_nodes: List of tuples [(x1, y1, z1), (x2, y2, z2), ...] representing selected nodes.
    """
    # Extract grid dimensions
    x_size, y_size, z_size = grid_shape

    # Generate the grid points
    x, y, z = np.meshgrid(
        np.arange(x_size),
        np.arange(y_size),
        np.arange(z_size)
    )
    x, y, z = x.flatten(), y.flatten(), z.flatten()  # Flatten the grid for easier plotting

    # Extract selected node coordinates
    selected_x = [node[0] for node in selected_nodes]
    selected_y = [node[1] for node in selected_nodes]
    selected_z = [node[2] for node in selected_nodes]

    # Generate a color gradient for selected nodes
    num_selected = len(selected_nodes)
    colors = np.linspace(0, 1, num_selected)  # Normalize gradient values for color scale

    # Create a 3D scatter plot
    fig = go.Figure()

    # Add all grid points as a background scatter
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=3,
            color='lightgray',  # Light gray for all grid nodes
            opacity=0.2
        ),
        name='Grid Nodes'
    ))


    # Define marker properties dynamically based on link_usage
    marker = dict(
        size=8,
        opacity=1
    )

    if link_usage is None:
        marker.update({
            "color": colors,  # Gradient colors for selected nodes
            "colorscale": 'Viridis',  # Use Viridis colormap
            "colorbar": dict(title="Selected Nodes Rank")  # Add colorbar for context
        })
    else:
        marker["color"] = 'black'  # Solid black color

    # Add the trace
    fig.add_trace(go.Scatter3d(
        x=selected_x, y=selected_y, z=selected_z,
        mode='markers',
        marker=marker,
        name='Selected Nodes'
    ))
    

    if link_usage:
        # Add links with color and width based on usage
        values = np.array(sorted([v for _, v in link_usage]))
        max_value = values[-1]  # Get the max usage for normalization
        q_index = int(quantile * len(link_usage))
        q_value = values[q_index]

        colormap = plt.cm.coolwarm  # Blue for low, red for high
        # Define the custom colors for the colormap
        colors = [
            (0, 1, 0),  # Green (low values)
            (0, 0, 1),  # Blue (medium values)
            (1, 0, 0)   # Red (high values)
        ]
        colormap = LinearSegmentedColormap.from_list("green-blue-red", colors, N=len(link_usage))

        # Sort the links by increasing values so we finish by the highest one
        link_usage = sorted(link_usage)
        for (start, end), value in tqdm(link_usage, total=len(link_usage), desc="Building graph", ncols=150):
            if value < q_value:
                continue

            # Handle wrapping links
            start_wrapped = list(start)
            end_wrapped = list(end)
            wrap = False

            for dim, size in enumerate(grid_shape):
                if abs(start[dim] - end[dim]) > size // 2:  # Wrapping detected
                    wrap = True
                    if start[dim] > end[dim]:  # Wrap down
                        end_wrapped[dim]   = start[dim]+1
                        start_wrapped[dim] = end[dim]-1
                    else:  # Wrap up
                        end_wrapped[dim]   = start[dim]-1
                        start_wrapped[dim] = end[dim]+1


            link_width = 2 + (value / max_value) * 20  # Adjust width dynamically

            # Normalize the value to range [0, 1]
            norm_value = (value - np.min(values)) / (np.max(values) - np.min(values))

            # Map the normalized values to a colormap (e.g., "coolwarm" for blue to red)
            rgba_color = colormap(norm_value)

            link_color = f"rgba({rgba_color[0] * 255:.0f}, {rgba_color[1] * 255:.0f}, {rgba_color[2] * 255:.0f}, {rgba_color[3]:.2f})"
            #rgb_color #"rgba(0,0,0)" #f"rgba({255 - int(255 * value / max_value)}, 0, {int(255 * value / max_value)}, 0.8)",
            if wrap:
                virtual_start = tuple(start_wrapped)
                virtual_end = tuple(end_wrapped)

                # Add the virtual link
                xs, ys, zs = zip(start, virtual_end)
                fig.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode='lines',
                    line=dict(
                        color=link_color,
                        width=link_width
                    ),
                    name=f'Virtual Link {start} -> {end}',
                    showlegend=False
                ))
                xs, ys, zs = zip(virtual_start, end)
                fig.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode='lines',
                    line=dict(
                        color=link_color,
                        width=link_width
                    ),
                    name=f'Virtual Link {start} -> {end}',
                    showlegend=False
                ))
            else:
                # Add the regular link
                xs, ys, zs = zip(start, end)
                fig.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode='lines',
                    line=dict(
                        color=link_color, 
                        width=link_width
                    ),
                    name=f'Link {start} -> {end}',
                    showlegend=False
                ))


    # Update the layout
    fig.update_layout(
        title=f"3D Grid ({x_size}x{y_size}x{z_size}) for {name}",
        scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Z-axis'
        ),
        margin=dict(l=0, r=0, b=0, t=30)  # Minimal margins for better display
    )

    # Show the plot
    fig.show()


def plot_link_usage(shape_3D, coords_3D, link_usage, save_to_html=None):
    """
    Plot the 3D points and links with colors and widths proportional to link usage.

    :param shape_3D: Shape of the 3D torus (dim_x, dim_y, dim_z).
    :param coords_3D: List of 3D coordinates.
    :param link_usage: List of links with usage values [(start, end), value].
    :param save_to_html: Path to save the plot as an HTML file (optional).
    """
    # If the file already exists, open it instead of constructing the figure
    if save_to_html and os.path.exists(save_to_html):
        print(f"File already exists: {save_to_html}. Opening it...")
        os.system(f"open {save_to_html}" if os.name == "posix" else f"start {save_to_html}")
        return

    x, y, z = zip(*coords_3D)  # Extract X, Y, Z for points

    # Create the 3D scatter plot for points
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=5, color='blue', opacity=0.8),
        name='Points'
    ))

    # Add links with color and width based on usage
    max_value = max([v for _, v in link_usage])  # Get the max usage for normalization
    for (start, end), value in tqdm(link_usage, total = len(link_usage), desc = "Building graph", ncols = 150) :
        if(value < max_value/2):
            continue
        xs, ys, zs = zip(start, end)  # Extract coordinates for the link

        # Add the link as a line
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines',
            line=dict(
                color=f"rgba({255 - int(255 * value / max_value)}, 0, {int(255 * value / max_value)}, 0.8)",
                width=1 + (value / max_value) * 5  # Adjust width dynamically
            ),
            name=f'Link {start} -> {end}',
            showlegend=False
        ))

    # Update layout for better visualization
    fig.update_layout(
        title=f"3D Torus Link Usage Visualization",
        scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Z-axis',
            aspectratio=dict(x=shape_3D[0], y=shape_3D[1], z=shape_3D[2])
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Save to HTML if specified
    fig.write_html("tmp/fig.html")
    
    if save_to_html:
        fig.write_html(save_to_html)
        print(f"Plot saved to {save_to_html}")

    # Show the plot
    fig.show()
    


def plot_mat(mat):
    #plot in grayscale
    plt.imshow(mat, cmap='gray_r', interpolation='nearest')

    # Add a colorbar for reference
    plt.colorbar(label="Value")

    # Show the plot
    plt.title("Matrix in Grayscale")
    plt.show()
        
