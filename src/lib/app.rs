use crate::error::ApplicationError;
use crate::vulkan;
use crate::vulkan::VALIDATION_ENABLED;
use ash::{extensions, vk, Device, Entry, Instance};
use winit::window::Window;

pub struct App {
    entry: Entry,
    instance: Instance,
    app_data: AppData,
    device: Device,
}

impl App {
    pub fn new(window: &Window) -> Result<Self, ApplicationError> {
        // Creating Vulkan entry point. The first thing we need to create before even creating the Vulkan instance.
        let entry = unsafe { Entry::load() }?;

        let mut app_data = AppData::new();

        // Creating Vulkan instance. It is needed for enumerating physical devices and creating logical device.
        let instance = unsafe { vulkan::create_instance(window, &entry, &mut app_data) }?;

        // Creating surface. It should be created before picking up a physical device because it affects this picking.
        app_data.surface = unsafe { vulkan::create_surface(&entry, &instance, &window) }?;

        // Picking physical device. It is needed for creating logical devices. Each physical device has it's own properties and features.
        unsafe { vulkan::pick_physical_device(&entry, &instance, &mut app_data) }?;

        // Creating logical device and storing queue handles in app data by calling this function.
        let device = unsafe { vulkan::create_device(&entry, &instance, &mut app_data) }?;

        unsafe { vulkan::create_swapchain(&entry, &window, &instance, &device, &mut app_data) }?;

        unsafe { vulkan::create_swapchain_image_views(&device, &mut app_data) }?;

        unsafe { vulkan::create_render_pass(&instance, &device, &mut app_data) }?;

        unsafe { vulkan::create_pipeline(&device, &mut app_data) }?;

        Ok(Self {
            entry,
            instance,
            app_data,
            device,
        })
    }

    pub fn render(&mut self, window: &Window) -> Result<(), ApplicationError> {
        Ok(())
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_render_pass(self.app_data.render_pass, None);

            self.device
                .destroy_pipeline_layout(self.app_data.pipeline_layout, None);

            self.app_data
                .swapchain_image_views
                .iter()
                .for_each(|v| self.device.destroy_image_view(*v, None));

            extensions::khr::Swapchain::new(&self.instance, &self.device)
                .destroy_swapchain(self.app_data.swapchain, None);

            extensions::khr::Surface::new(&self.entry, &self.instance)
                .destroy_surface(self.app_data.surface, None);

            self.device.destroy_device(None);

            if VALIDATION_ENABLED {
                extensions::ext::DebugUtils::new(&self.entry, &self.instance)
                    .destroy_debug_utils_messenger(self.app_data.debug_utils_messenger, None);
            }

            self.instance.destroy_instance(None);
        }
    }
}

pub struct AppData {
    pub debug_utils_messenger: vk::DebugUtilsMessengerEXT,
    pub physical_device: vk::PhysicalDevice,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub surface: vk::SurfaceKHR,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub render_pass: vk::RenderPass,
    pub pipeline_layout: vk::PipelineLayout,
}

impl AppData {
    fn new() -> Self {
        Self {
            debug_utils_messenger: Default::default(),
            physical_device: Default::default(),
            graphics_queue: Default::default(),
            present_queue: Default::default(),
            surface: Default::default(),
            swapchain: Default::default(),
            swapchain_format: Default::default(),
            swapchain_extent: Default::default(),
            swapchain_images: Default::default(),
            swapchain_image_views: Default::default(),
            render_pass: Default::default(),
            pipeline_layout: Default::default(),
        }
    }
}
