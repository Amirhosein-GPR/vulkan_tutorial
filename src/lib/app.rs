use crate::error::ApplicationError;
use crate::vulkan;
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

        // Picking physical device. It is needed for creating logical devices. Each physical device has it's own properties and features.
        unsafe { vulkan::pick_physical_device(&instance, &mut app_data) }?;

        // Creating logical device and storing queue handles in app data by calling this function.
        let device = unsafe { vulkan::create_device(&entry, &instance, &mut app_data) }?;

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
            self.device.destroy_device(None);

            extensions::ext::DebugUtils::new(&self.entry, &self.instance)
                .destroy_debug_utils_messenger(self.app_data.debug_utils_messenger, None);

            self.instance.destroy_instance(None);
        }
    }
}

pub struct AppData {
    pub debug_utils_messenger: vk::DebugUtilsMessengerEXT,
    pub physical_device: vk::PhysicalDevice,
    pub graphics_queue: vk::Queue,
}

impl AppData {
    fn new() -> Self {
        Self {
            debug_utils_messenger: Default::default(),
            physical_device: Default::default(),
            graphics_queue: Default::default(),
        }
    }
}
