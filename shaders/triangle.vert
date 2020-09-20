#version 450

// Takes in the vec2 from the buffer for this vertex shader to work on
// Tells it that the attribute is named position
layout(location = 0) in vec2 position;
layout(location = 1) in vec4 color;
//layout(binding = 0) uniform UniformBufferObject {
//    mat4 model;
//    mat4 view;
//    mat4 proj;
//} ubo;

layout(location = 0) out vec4 fragColor;

// Called once for each vertex
void main() {

    // Homogenous Vector
    // X, Y, Z, W
    // if W is 1 then we are working with position in space
    // if W is 0 then we are working with a direction
    // Can be used to do things like, rotating a set of vertices around a point over time
    // We give it a 0.0 since we are only providing a 2D vertex and not 3D
//    gl_Position = ubo.proj * ubo.view * ubo.model *  vec4(position, 0.0, 1.0);
    gl_Position = vec4(position, 0.0, 1.0);
    // Pass through the input color to the frag shader
    fragColor = color;
}
