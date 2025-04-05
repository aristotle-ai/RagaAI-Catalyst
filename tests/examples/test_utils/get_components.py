# Helper function to recursively process components and their children
def process_component(component, all_components):
    # Extract component type and name
    component_type = component.get("type")
    component_name = component.get("name")
    
    # Append the component to the list
    all_components.append({"type": component_type, "name": component_name})
    
    # Process children if they exist
    data = component.get("data", {})
    if isinstance(data, dict):
        children = data.get("children", [])
        for child in children:
            process_component(child, all_components)

# Test function to validate the structure and sequence of components
def get_component_structure_and_sequence(json_data):
    # Initialize an empty list to store all components
    all_components = []
    
    # Extract the spans from the result.json
    span_components = json_data["data"][0]["spans"]
    
    # Process each component and its children
    for component in span_components:
        process_component(component, all_components)
    return all_components

