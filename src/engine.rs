pub mod app;
pub(crate) mod memory;
pub(crate) mod renderer;
pub(crate) mod shaders;

pub use app::App;
pub(crate) use memory::DynVertexBuffer;
pub(crate) use renderer::RenderContext;

use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

#[derive(BufferContents, Vertex, Debug)]
#[repr(C)]
pub(crate) struct MyVertex {
    #[format(R32G32_SFLOAT)]
    pos: [f32; 2],
    #[format(R32G32_SFLOAT)]
    vel: [f32; 2],
}
