#version 450

layout(location = 0) in vec4 fragColor;

// The output of the fragment shader
// Can output to multiple images at once
layout(location = 0) out vec4 f_color;

// Is ran on every pixel that the GPU deems to be inside of the shape determined by a vertex shader
void main() {

    // The value to be written to the target image for this pixel
    // ALL VALUES ARE NORMALIZED 1.0 IS 255
    f_color = fragColor;
}
