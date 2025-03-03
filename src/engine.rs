pub mod app;
pub(crate) mod memory;
pub(crate) mod renderer;
pub(crate) mod shaders;

// use std::collections::HashSet;

pub use app::App;
pub(crate) use memory::DynMemoryManager;
pub(crate) use renderer::RenderContext;

use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};
// use winit::event::{ElementState, MouseButton, WindowEvent};

#[derive(BufferContents, Vertex, Debug)]
#[repr(C)]
pub(crate) struct MyVertex {
    #[format(R32G32_SFLOAT)]
    pos: [f32; 2],
    #[format(R32G32_SFLOAT)]
    vel: [f32; 2],
}

// pub(crate) struct InputState {
//     held_buttons: HashSet<MouseButton>,
// }
//
// impl InputState {
//     pub fn new() -> Self {
//         Self {
//             held_buttons: HashSet::new(),
//         }
//     }
//
//     pub fn handle_event(&mut self, event: &WindowEvent) {
//         if let WindowEvent::MouseInput { state, button, .. } = event {
//             match state {
//                 ElementState::Pressed => {
//                     self.held_buttons.insert(*button);
//                 }
//                 ElementState::Released => {
//                     self.held_buttons.remove(button);
//                 }
//             }
//         }
//     }
//
//     pub fn is_held(&self, button: MouseButton) -> bool {
//         self.held_buttons.contains(&button)
//     }
// }
