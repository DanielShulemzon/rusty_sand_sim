use std::error::Error;

use winit::event_loop::EventLoop;

mod engine;

pub use engine::App;

fn main() -> Result<(), impl Error> {
    // The usual Vulkan initialization. Largely the same as the triangle example until further
    // commentation is provided.

    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}
