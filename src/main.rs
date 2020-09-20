use log::{Metadata, Record, Level, LevelFilter};
use log::{info, error, debug};
use std::io::Error;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::{
    event::{Event, WindowEvent, DeviceEvent, KeyboardInput, ElementState, VirtualKeyCode},
    event_loop::{ControlFlow, EventLoop},
    window::{WindowBuilder, Window},
    dpi::{LogicalSize},
};
use vulkano_win::VkSurfaceBuild;
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice, QueueFamily
};
use vulkano::device::{Device, DeviceExtensions, Features, QueuesIter, Queue
};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer, DynamicState
};
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::pipeline::{ComputePipeline, GraphicsPipeline, viewport::Viewport};
use vulkano::descriptor::{
    descriptor_set::PersistentDescriptorSet,
    descriptor_set::FixedSizeDescriptorSet,
    descriptor_set::DescriptorSet,
    descriptor_set::collection::DescriptorSetsCollection,
    PipelineLayoutAbstract,
};
use vulkano::format::{Format, ClearValue};
use vulkano::image::{Dimensions, StorageImage, ImageUsage, SwapchainImage};
use vulkano::framebuffer::{Subpass, Framebuffer, RenderPassAbstract, FramebufferAbstract};
use vulkano::swapchain::{Swapchain, SurfaceTransform, PresentMode, ColorSpace, FullscreenExclusive, Surface, SwapchainCreationError, acquire_next_image, AcquireError};
use vulkano::memory::pool::{PotentialDedicatedAllocation, StdMemoryPoolAlloc};
// use nalgebra_glm::{Mat4, mat4, mat3, vec3, Vec3};
use rand::{thread_rng, Rng};

mod shaders;

struct Logger;

impl log::Log for Logger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Trace
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            println!("[{:?}] {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

static LOGGER: Logger = Logger;

// TODO: Make these configurable
const WIDTH: u32 = 1024;
const HEIGHT: u32 = 1024;

// Matrices for coordinate stuff
// struct UniformBufferObject {
//     model: Mat4,
//     view: Mat4,
//     proj: Mat4,
// }

// Define the representation of a Vertex for Vulkano and shaders
// Vulkan has a width and height of 2 units from -1, -1 to 1, 1 where 0, 0 is the center
#[derive(Default, Copy, Clone, Debug)]
struct Vertex {
    // The position of the vertex as an array with 2 floats (X and Y)
    position: [f32; 2],
    color: [f32; 4],
    s_dimensions: [f32; 2],
}

impl Vertex {
    pub fn from_screen_space(x: f32, y: f32, s_dimensions: &[f32; 2], color: &[f32; 4]) -> Self {
        Self {
            position: [x, y],
            color: *color,
            s_dimensions: *s_dimensions
        }.to_view_space()
    }

    fn to_screen_space(self) -> Self {
        let pos_x = {
            if self.position[0] < 0.0 {
                self.s_dimensions[0] * (self.position[0] * 0.5 * -1.0)
            } else if self.position[0] > 0.0 {
                self.s_dimensions[0] * ((self.position[0] + 1.0) * 0.5)
            } else {
                self.s_dimensions[0] / 2.0
            }
        };
        let pos_y = {
            if self.position[1] < 0.0 {
                self.s_dimensions[1] * (self.position[1] * 0.5 * -1.0)
            } else if self.position[1] > 0.0 {
                self.s_dimensions[1] * ((self.position[1] + 1.0) * 0.5)
            } else {
                self.s_dimensions[1] / 2.0
            }
        };
        Self {
            position: [pos_x, pos_y],
            color: self.color,
            s_dimensions: self.s_dimensions,
        }
    }

    fn to_view_space(self) -> Self {
        let mut pos_x = (self.position[0] / self.s_dimensions[0]);
        let mut pos_y = (self.position[1] / self.s_dimensions[1]);
        pos_x = {
            if pos_x == 0.0 {
                -1.0
            } else if pos_x < 0.5 {
                -1.0 - (pos_x / 0.5 * -1.0)
            } else if pos_x > 0.5 {
                pos_x / 0.5 - 1.0
            } else {
                0.0
            }
        };
        pos_y = {
            if pos_y == 0.0 {
                -1.0
            } else if pos_y < 0.5 {
                -1.0 - (pos_y * -1.0 / 0.5)
            } else if pos_y > 0.5 {
                pos_y / 0.5 - 1.0
            } else {
                0.0
            }
        };
        Self {
            position: [pos_x, pos_y],
            color: self.color,
            s_dimensions: self.s_dimensions,
        }
    }
}

// Creates the link between the contents of the buffer and the input of the vertex shader
// I assume you can use this to detail things like rotation too?
vulkano::impl_vertex!(Vertex, position, color);

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Facing {
    NORTH,
    EAST,
    SOUTH,
    WEST,
    NONE
}

impl Facing {
    pub fn find_facing(head_pos: &[u8; 2], prev_head_pos: &[u8; 2]) -> Self {
        if prev_head_pos[0] as i8 - head_pos[0] as i8 == -1 {
            Self::EAST
        } else if prev_head_pos[0] as i8 - head_pos[0] as i8 == 1 {
            Self::WEST
        } else if prev_head_pos[1] as i8 - head_pos[1] as i8 == 1 {
            Self::NORTH
        } else if prev_head_pos[1] as i8 - head_pos[1] as i8 == -1  {
            Self::SOUTH
        } else {
            Self::NONE
        }
    }

    fn get_addition(&self) -> [i8; 2] {
        match self {
            Facing::NORTH => [0, -1],
            Facing::SOUTH => [0, 1],
            Facing::EAST => [1, 0],
            Facing::WEST => [-1, 0],
            _ => [0, 0]
        }
    }

    fn opposite(&self) -> Self {
        match self {
            Facing::NORTH => Facing::SOUTH,
            Facing::SOUTH => Facing::NORTH,
            Facing::EAST => Facing::WEST,
            Facing::WEST => Facing::EAST,
            _ => Facing::NONE,
        }
    }
}

impl From<u8> for Facing {
    fn from(dir: u8) -> Self {
        match dir {
            0 => Self::NORTH,
            1 => Self::EAST,
            2 => Self::SOUTH,
            3 => Self::WEST,
            _ => Self::NONE
        }
    }
}

struct App {
    v_instance: Arc<Instance>,
    v_device: Arc<Device>,
    v_queues: QueuesIter,
    v_queue: Arc<Queue>,
    // An array of 200 arrays of 2 bytes denoting coordinates
    // E.G [2, 2] is the 3rd column and 3rd row
    grid_size: [u8; 2],
    // Grid position
    apple_pos: [u8; 2],
    head_pos: [u8; 2],
    body: Vec<[u8; 2]>
}

impl App {
    pub fn initialize() -> App {
        log::set_logger(&LOGGER)
            .map(|()| log::set_max_level(LevelFilter::Trace)).unwrap();
        // for x in vulkano::instance::layers_list().unwrap() {
        //     println!("{}: {}", x.name(), x.description());
        // }

        // Setup Vulkan communications with the GPU
        let (instance, device, mut queues) =
            App::init_vulkan();
        let queue = queues.next().unwrap();
        let grid_size: [u8; 2] = [50, 50];
        Self {
            v_instance: instance.clone(),
            v_device: device.clone(),
            v_queues: queues,
            v_queue: queue.clone(),
            grid_size,
            apple_pos: [0u8; 2],
            head_pos: [grid_size[0]/2, grid_size[1]/2],
            body: Vec::new(),
        }
    }

    fn init_vulkan() -> (Arc<Instance>, Arc<Device>, QueuesIter) {

        // for x in layer_names {
        //     println!("{}", x);
        // }

        /*
        * An instance of Vulkan
        * Takes an application info with information like the App name, version, engine name and version
        * Takes a list of Vulkan InstanceExtensions
        * Takes a list of layers you want to support EG GOG Galaxy or Steam overlay
        */
        let instance = {
            // The extensions required to use vulkan surfaces
            let extensions = vulkano_win::required_extensions();
            Instance::new(None, &extensions, None)
                .expect("Failed to create Vulkan instance")
        };
        // Returns the first physical device that supports Vulkan
        // EG: Graphics Cards, Integrated Graphics
        let physical = PhysicalDevice::enumerate(&instance).next()
            .expect("No device available");
        // Queues are multithreading for GPU
        // An operation can be pushed to a queue that supports it
        // creating a parallel workorder for the GPU
        /*
        * My GTX 1660 has 3 queue families
        * first with 16 queues
        * second with 2 queues
        * third with 8 queues
        */
        // for family in physical.queue_families() {
        //     info!("Found a queue family with {:?} queue(s)", family.queues_count());
        // }
        // Iterates over the queue families and returns the first one that supports graphics
        let queue_family = physical.queue_families()
            .find(|&q| q.supports_graphics())
            .expect("Could not find a graphical queue family");
        // Creates a Vulkan device that allows us to communicate with the GPU
        // Also returns the queues that allow us to submit operations
        let (device, mut queues) =
            Device::new(physical, &Features::none(),
                        &DeviceExtensions{khr_storage_buffer_storage_class:true, khr_swapchain: true,
                            ..DeviceExtensions::none()},
                        [(queue_family, 0.5)].iter().cloned())
                .expect("Failed to create a device");
        (instance, device, queues)
    }

    fn main_loop(&mut self) {
        let device = self.v_device.clone();
        let queue = self.v_queue.clone();


        let event_loop = EventLoop::new();
        // Creates a vulkan surface
        // Surface.window() returns an object to manipulate the window
        let surface = WindowBuilder::new()
            .with_title("Snake")
            .with_resizable(false)
            .with_inner_size(LogicalSize::new(f64::from(WIDTH),
                                              f64::from(HEIGHT)))
            .build_vk_surface(&event_loop, self.v_instance.clone()).unwrap();

        // Get the capabilities of the surface
        let caps = surface.capabilities(device.physical_device())
            .expect("failed to get surface capabilities");
        // get the Dimensions of the surface or default to our defined width and height
        let dimensions = caps.current_extent.unwrap_or([WIDTH, HEIGHT]);
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;

        // Load the triangle vertex shader into the GPU
        let triangle_vert = shaders::triangle_vert::Shader::load(device.clone())
            .expect("Failed to create vert shader");
        // Load the triangle fragment shader into the GPU
        let triangle_frag = shaders::triangle_frag::Shader::load(device.clone())
            .expect("Failed to create frag shader");

        // Create Swapchain
        // Essentially a grouping of images, the one that is shown, the one that is drawn too, and others
        // Think backbuffer and flipping buffers
        let (mut swapchain, images) = Swapchain::new(device.clone(), surface.clone(),
                                                     caps.min_image_count, format, dimensions,
                                                     1, ImageUsage::color_attachment(),
                                                     &queue, SurfaceTransform::Identity, alpha,
                                                     PresentMode::Fifo,
                                                     FullscreenExclusive::Default, true,
                                                     ColorSpace::SrgbNonLinear
        ).expect("Failed to create Swapchain");

        let render_pass = Arc::new(vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    // Clear: Clears the image when entering the render pass
                    // Load: Replaces something existing
                    load: Clear,
                    // Store the output of our draw commands to the image
                    // Can create Temporary images whose content is only accessible in the render_pass
                    // So would use store: DontCare to do nothing with the image
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }).unwrap()
        );

        let graphics_pipe = Arc::new(GraphicsPipeline::start()
            // What kind of vertex input is expected
            .vertex_input_single_buffer::<Vertex>()
            // .triangle_list()
            // Whether it should source a vertex at the index (false) or start a new primitive at that index (true)
            // .primitive_restart(false)
            // Vertex shader
            .vertex_shader(triangle_vert.main_entry_point(), ())
            // Where to render to the screen
            // Any part of a shape outside of the viewport is discarded
            // Tells the builder we are only using 1 viewport and that the state is dynamic
            // Can change the viewport for each draw command
            // if wasn't dynamic would need to create a new pipeline for every image
            // May be useful for say rendering a screen of part of the level, using more than 1 viewport?
            // Geometry shaders can choose which viewport to draw too
            .viewports_dynamic_scissors_irrelevant(1)
            // Fragment shader
            .fragment_shader(triangle_frag.main_entry_point(), ())
            // This pipeline concerns the first pass of the render pass
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone()).unwrap()
        );

        // let layout = graphics_pipe.layout().descriptor_set_layout(0).unwrap();
        // let set = Arc::new(
        //     FixedSizeDescriptorSet::
        // );

        let mut dynamic_state = DynamicState {
            // Define the viewports
            viewports: Some(vec![Viewport {
                // Sets the origin as 0.0, 0.0
                origin: [0.0, 0.0],
                // Sets the viewports dimensions
                // In this case the entire size
                dimensions: [WIDTH as f32, HEIGHT as f32],
                depth_range: 0.0..1.0,
            }]),
            // Fill the rest of this with none
            .. DynamicState::none()
        };

        // The render pass we created above only describes the layout of our framebuffers. Before we
        // can draw we also need to create the actual framebuffers.
        //
        // Since we need to draw to multiple images, we are going to create a different framebuffer for
        // each image.
        let mut framebuffer = window_size_dependent_setup(&images, render_pass.clone(),
                                                          &mut dynamic_state);

        let indices: Vec<u16> = vec![0, 1, 2, 2, 3, 0];
        let (i_buffer, future) = ImmutableBuffer::from_iter(
            indices.iter().cloned(), BufferUsage::index_buffer(), queue.clone()
        ).unwrap();
        future.then_signal_fence_and_flush().unwrap().wait(None).unwrap();

        // Tries to take ownership of an image to draw on it, returns index of image in images array
        // and a future representing when the image will become available from the GPU
        // let (image_num, subopt, acquire_future) = vulkano::swapchain::acquire_next_image(swapchain.clone(),
        //                                                                          None).unwrap();
        // Whether we should recreate the swapchain the next loop, if it changed from resizing and such
        let mut recreate_swapchain: bool = false;

        // Store the submission of the previous frame
        // Without storing it, the loop will block till the GPU is finished executing
        let mut previous_frame_end = Some(vulkano::sync::now(device.clone()).boxed());


        // let mut coords: Vec3 = vec3(0.0, 0.0, 1.0);
        let mut body = self.body.clone();
        let mut head_pos = self.head_pos.clone();
        let mut previous_head_pos = self.head_pos.clone();
        let mut facing: Facing = Facing::WEST;
        let mut apple_pos = self.apple_pos.clone();
        let mut apple_exists = false;
        let grid_size = self.grid_size.clone();
        let mut alive = true;
        let mut input_buffer: Vec<Facing> = vec![Facing::WEST];

        let mut tick = Instant::now();

        event_loop.run(move |event, _, control_flow| {
            if Instant::now().duration_since(tick).as_millis() >= 100 {
                let i_head_pos: [i8; 2] = [head_pos[0] as i8, head_pos[1] as i8];
                if !input_buffer.is_empty() {
                    if input_buffer[0] == facing.opposite() {
                        input_buffer.remove(0);
                    } else {
                        facing = input_buffer[0].clone();
                        input_buffer.remove(0);
                    }
                }
                head_pos[0] = (i_head_pos[0] + facing.get_addition()[0]) as u8;
                head_pos[1] = (i_head_pos[1] + facing.get_addition()[1]) as u8;
                if head_pos[0] > grid_size[0] - 1 {
                    head_pos[0] = 1;
                } else if head_pos[0] < 1 {
                    head_pos[0] = grid_size[0] - 1;
                } else {}
                if head_pos[1] > grid_size[1] - 1 {
                    head_pos[1] = 1;
                } else if head_pos[1] < 1 {
                    head_pos[1] = grid_size[1]- 1;
                }else {}
                if head_pos != previous_head_pos {
                    let previous_body = &body.clone();
                    for i in (0..body.len()).rev() {
                        if i == 0 {
                            body[i] = previous_head_pos;
                        } else {
                            body[i] = previous_body[i - 1];
                        }
                    }
                    previous_head_pos = head_pos;
                }
                while !apple_exists {
                    let grid_pos_x = thread_rng().gen_range(1, grid_size[0]-1);
                    let grid_pos_y = thread_rng().gen_range(1, grid_size[1]-1);

                    if !(body.contains(&[grid_pos_x, grid_pos_y]) || head_pos == [grid_pos_x, grid_pos_y]) {
                        apple_pos = [grid_pos_x, grid_pos_y];
                        apple_exists = true;
                    }
                }
                if body.contains(&head_pos) {
                    alive = false;
                }
                if apple_exists && head_pos == apple_pos {
                    body.push(apple_pos);
                    apple_pos = [0u8; 2];
                    apple_exists = false;
                }
                if !alive {
                    body = Vec::new();
                    head_pos = [grid_size[0]/2, grid_size[1]/2];
                    previous_head_pos = head_pos;
                    facing = Facing::WEST;
                    let mut apple_exists = false;
                    alive = true;
                }
                // TODO: ADD TOGGLEABLE TELEPORTING
                tick = Instant::now();
            }
            match event {
                Event::DeviceEvent { device_id, event: DeviceEvent::Key(key) } => {
                    match key.state {
                        ElementState::Pressed => {
                            match key.virtual_keycode {
                                Some(VirtualKeyCode::Left) => {
                                    input_buffer.push(Facing::WEST);
                                    // Movement Code for snake manual control
                                    // if head_pos[0] > 1 {
                                    //     previous_head_pos = head_pos;
                                    //     head_pos[0] -= 1;
                                    // }
                                    // Movement Code for anything movement
                                    // coords = mat3(
                                    //     1.0, 0.0, -10.0,
                                    //     0.0, 1.0, 0.0,
                                    //     0.0, 0.0, 1.0) * coords;
                                },
                                Some(VirtualKeyCode::Right) => {
                                    input_buffer.push(Facing::EAST);
                                    // Movement Code for snake manual control
                                    // if head_pos[0] < grid_size[0] - 1 {
                                    //     previous_head_pos = head_pos;
                                    //     head_pos[0] += 1;
                                    // }
                                    // Movement Code for anything movement
                                    // coords = mat3(
                                    //     1.0, 0.0, 10.0,
                                    //     0.0, 1.0, 0.0,
                                    //     0.0, 0.0, 1.0) * coords;
                                }
                                Some(VirtualKeyCode::Up) => {
                                    input_buffer.push(Facing::NORTH);
                                    // Movement Code for snake manual control
                                    // if head_pos[1] > 1 {
                                    //     previous_head_pos = head_pos;
                                    //     head_pos[1] -= 1;
                                    // }
                                    // Movement Code for anything movement
                                    // coords = mat3(
                                    //     1.0, 0.0, 0.0,
                                    //     0.0, 1.0, -10.0,
                                    //     0.0, 0.0, 1.0) * coords;
                                }
                                Some(VirtualKeyCode::Down) => {
                                    input_buffer.push(Facing::SOUTH);
                                    // Movement Code for snake manual control
                                    // if head_pos[1] < grid_size[1] - 1 {
                                    //     previous_head_pos = head_pos;
                                    //     head_pos[1] += 1;
                                    // }
                                    // Movement Code for anything movement
                                    // coords = mat3(
                                    //     1.0, 0.0, 0.0,
                                    //     0.0, 1.0, 10.0,
                                    //     0.0, 0.0, 1.0) * coords;
                                }
                                _ => {}
                            }
                        },
                        ElementState::Released => {}
                    }
                },
                // If we request that window is closed (E.G Pressing the x on the window)
                Event::WindowEvent{event: WindowEvent::CloseRequested, ..} => {
                    *control_flow = ControlFlow::Exit;
                },
                Event::WindowEvent {event: WindowEvent::Resized(_), ..} => {
                    recreate_swapchain = true;
                },
                Event::RedrawEventsCleared => {
                    // Clears resources when they've been used
                    previous_frame_end.as_mut().unwrap().cleanup_finished();

                    if recreate_swapchain {
                        // let dimensions: [u32; 2] = caps.current_extent.unwrap_or([WIDTH, HEIGHT]);
                        let dimensions: [u32; 2] = surface.window().inner_size().into();

                        let (new_swapchain, new_images) =
                            match swapchain.recreate_with_dimensions(dimensions) {
                                Ok(r) => r,
                                Err(SwapchainCreationError::UnsupportedDimensions) => return,
                                Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                            };
                        swapchain = new_swapchain;
                        framebuffer = window_size_dependent_setup(
                            &new_images,
                            render_pass.clone(),
                            &mut dynamic_state,
                        );

                        recreate_swapchain = false;
                    }

                    // Tries to take ownership of an image to draw on it, returns index of image in images array
                    // and a future representing when the image will become available from the GPU
                    let (image_num, subopt, acquire_future) = {
                        match acquire_next_image(swapchain.clone(), None) {
                            Ok(r) => r,
                            Err(AcquireError::OutOfDate) => {
                                recreate_swapchain = true;
                                return;
                            }
                            Err(e) => panic!("Failed to acquire next image: {:?}", e),
                        }
                    };

                    if subopt {
                        recreate_swapchain = true;
                    }

                    let clear_values = vec![[0.0, 0.0, 0.0, 1.0].into()];

                    let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
                        device.clone(), queue.family(),
                    ).unwrap();
                    let dimensions: [f32; 2] = surface.window().inner_size().into();

                    // debug!("{:?}", Vertex::from_screen_space(
                    //         coords.x,
                    //         coords.y,
                    //         &dimensions,
                    //         &[0.0, 1.0, 0.0, 1.0],
                    //     ).to_screen_space());
                    let render_pass = builder
                        .begin_render_pass(framebuffer[image_num].clone(), false, clear_values)
                        .unwrap();
                    // DRAW CALLS FOR RENDER PASS GO HERE
                    for loc in &body {
                        // debug!("DRAWING BODY AT {:#}:{:#}", loc[0], loc[1]);
                        render_pass.draw_indexed(
                            graphics_pipe.clone(),
                            &dynamic_state,
                            draw_body(
                                loc[0],
                                loc[1],
                                &grid_size, &[1.0, 0.0, 0.0, 1.0], &dimensions, device.clone()
                            ),
                            i_buffer.clone(),
                            (),
                            ()
                        ).unwrap();
                    }
                    // Draw a Vertex Buffer using an index buffer
                    // Index Buffer indicates how to draw triangles using the vertices
                    // 0, 1, 2, 2, 3, 0 means TopLeft, TopRight, BottomRight, BottomRight, Bottom Left, TopLeft
                    render_pass.draw_indexed(
                        graphics_pipe.clone(),
                        &dynamic_state,
                        draw_body(
                            head_pos[0],
                            head_pos[1],
                            &grid_size, &[0.0, 0.0, 1.0, 1.0], &dimensions, device.clone()
                        ),
                        i_buffer.clone(),
                        (),
                        ()
                    ).unwrap();
                    render_pass.draw_indexed(
                        graphics_pipe.clone(),
                        &dynamic_state,
                        draw_body(
                            apple_pos[0],
                            apple_pos[1],
                            &grid_size, &[0.0, 1.0, 0.0, 1.0], &dimensions, device.clone()
                        ),
                        i_buffer.clone(),
                        (),
                        ()
                    ).unwrap();
                    // Leave the render pass
                    // can enter next subpass with next_inline or next_secondary
                    render_pass.end_render_pass().unwrap();
                    let comm_buffer = builder.build().unwrap();

                    let future = previous_frame_end
                        .take()
                        .unwrap()
                        .join(acquire_future)
                        .then_execute(queue.clone(), comm_buffer)
                        .unwrap()
                        // The colour output SHOULD contain our triangle
                        // Submits a present command at the end of the queue
                        .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                        .then_signal_fence_and_flush();

                    match future {
                        Ok(future) => {
                            previous_frame_end = Some(future.boxed());
                        }
                        Err(FlushError::OutOfDate) => {
                            recreate_swapchain = true;
                            previous_frame_end = Some(vulkano::sync::now(device.clone()).boxed());
                        }
                        Err(e) => {
                            error!("Failed to flush future: {:?}", e);
                            previous_frame_end = Some(vulkano::sync::now(device.clone()).boxed());
                        }
                    }
                }
                _ => ()
            }
        });
    }
}

fn main() {

    let mut app = App::initialize();
    app.main_loop();

}

fn draw_body(grid_x: u8, grid_y: u8, grid_size: &[u8; 2], color: &[f32; 4], dimensions: &[f32; 2],
             device: Arc<Device>)
             -> Arc<CpuAccessibleBuffer<[Vertex], PotentialDedicatedAllocation<StdMemoryPoolAlloc>>> {
    let scale_perc: [f32; 2] = [(dimensions[0]/1.25)/grid_size[0] as f32, (dimensions[1]/1.25)/grid_size[1] as f32];

    let x = if grid_x == grid_size[0] {
        grid_x as f32 * &scale_perc[0]
    } else {
        (grid_x as f32 * &scale_perc[0]) * 1.25
    };
    let y = if grid_y == grid_size[1] {
        grid_y as f32 * &scale_perc[1] - &scale_perc[1]
    } else {
        (grid_y as f32 * &scale_perc[1]) * 1.25 - &scale_perc[1]
    };
    create_box(
        x - (scale_perc[0]/2.0),
        y,
        scale_perc[0], scale_perc[1], color, &dimensions, device.clone()
    )

}

fn create_box(x: f32, y: f32, width: f32, height: f32, color: &[f32; 4], dimensions: &[f32; 2], device: Arc<Device>)
    -> Arc<CpuAccessibleBuffer<[Vertex], PotentialDedicatedAllocation<StdMemoryPoolAlloc>>> {
    let v_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        vec![
            // TOP LEFT
            Vertex::from_screen_space(
                x,
                y,
                dimensions,
                color,
            ),
            // TOP RIGHT
            Vertex::from_screen_space(
                x + width,
                y,
                dimensions,
                color,
            ),
            // BOTTOM RIGHT
            Vertex::from_screen_space(
                x + width,
                y + height,
                dimensions,
                color,
            ),
            // BOTTOM LEFT
            Vertex::from_screen_space(
                x,
                y + height,
                dimensions,
                color,
            ),
        ].into_iter()
    ).expect("Failed to create buffer");
    v_buffer
}

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0,
    };
    dynamic_state.viewports = Some(vec![viewport]);

    images
        .iter()
        .map(|image| {
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(image.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>()
}