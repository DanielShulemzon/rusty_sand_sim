use std::{sync::Arc, time::SystemTime};
use vulkano::{
    pipeline::GraphicsPipeline,
    render_pass::{Framebuffer, RenderPass},
    shader::EntryPoint,
    swapchain::Swapchain,
    sync::GpuFuture,
};
use winit::{dpi::PhysicalPosition, window::Window};

use super::MouseState;

pub struct RenderContext {
    pub window: Arc<Window>,
    pub swapchain: Arc<Swapchain>,
    pub render_pass: Arc<RenderPass>,
    pub framebuffers: Vec<Arc<Framebuffer>>,
    pub vs: EntryPoint,
    pub fs: EntryPoint,
    pub pipeline: Arc<GraphicsPipeline>,
    pub recreate_swapchain: bool,
    pub previous_frame_end: Option<Box<dyn GpuFuture>>,
    pub start_time: SystemTime,
    pub last_frame_time: SystemTime,
    pub cursor_pos: PhysicalPosition<f64>,
    pub mouse_state: MouseState,
}

// pub enum GameState {}
//
// pub struct AppData {
//     pub pixel_count: u32,
// }
