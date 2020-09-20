pub mod triangle_vert {
    vulkano_shaders::shader!{
        ty: "vertex",
        path: "./shaders/triangle.vert"
    }
}
pub mod triangle_frag {
    vulkano_shaders::shader!{
        ty: "fragment",
        path: "./shaders/triangle.frag"
    }
}