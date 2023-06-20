#![allow(
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

use ash::extensions;
use nalgebra::{Vector2, Vector3};
use pretty_env_logger;
use vulkan_tutorial::app::App;
use vulkan_tutorial::error::ApplicationError;
use vulkan_tutorial::vulkan::Vertex;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

fn main() -> Result<(), ApplicationError> {
    // Initializing pretty_env_logger library (used for logging purposes).
    pretty_env_logger::init();

    // Creating event loop and window.
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Vulkan tutorial (Rust)")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)?;

    let vertecies = vec![
        Vertex::new(Vector2::new(0.0, -0.5), Vector3::new(1.0, 1.0, 1.0)),
        Vertex::new(Vector2::new(0.5, 0.5), Vector3::new(0.0, 1.0, 0.0)),
        Vertex::new(Vector2::new(-0.5, 0.5), Vector3::new(0.0, 0.0, 1.0)),
    ];

    // Initilaizeing app struct which holds main logic of this application.
    let mut app = App::new(&window, &vertecies)?;

    // Running event loop and managing the needed states.
    let mut destroying = false;
    let mut minimized = false;
    event_loop.run(move |event, _event_loop_window_target, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::MainEventsCleared if !destroying && !minimized => app
                .render(&window, &vertecies)
                .expect("Error: Rendering failed."),
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                if size.width == 0 || size.height == 0 {
                    minimized = true;
                } else {
                    minimized = false;
                    app.resized = true;
                }
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                destroying = true;
                *control_flow = ControlFlow::Exit;
                unsafe {
                    app.device.device_wait_idle().unwrap();
                }
            }
            _ => {}
        }
    });
}
