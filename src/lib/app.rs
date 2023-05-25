use crate::error::ApplicationError;
use crate::vulkan;
use crate::vulkan::MAX_FRAMES_IN_FLIGHT;
use crate::vulkan::VALIDATION_ENABLED;
use ash::{extensions, vk, Device, Entry, Instance};
use winit::window::Window;

pub struct App {
    entry: Entry,
    instance: Instance,
    app_data: AppData,
    pub device: Device,
    frame: u8,
}

impl App {
    pub fn new(window: &Window) -> Result<Self, ApplicationError> {
        // Creating Vulkan entry point. The first thing we need to create before even creating the Vulkan instance.
        let entry = unsafe { Entry::load() }?;

        let mut app_data = AppData::default();

        // Creating Vulkan instance. It is needed for enumerating physical devices and creating logical device.
        let instance = unsafe { vulkan::create_instance(window, &entry, &mut app_data) }?;

        // Creating surface. It should be created before picking up a physical device because it affects this picking.
        app_data.surface = unsafe { vulkan::create_surface(&entry, &instance, &window) }?;

        // Picking physical device. It is needed for creating logical devices. Each physical device has it's own properties and features.
        unsafe { vulkan::pick_physical_device(&entry, &instance, &mut app_data) }?;

        // Creating logical device and storing queue handles in app data by calling this function.
        let device = unsafe { vulkan::create_device(&entry, &instance, &mut app_data) }?;

        unsafe {
            vulkan::create_swapchain(&entry, &window, &instance, &device, &mut app_data)?;

            vulkan::create_swapchain_image_views(&device, &mut app_data)?;

            vulkan::create_render_pass(&instance, &device, &mut app_data)?;

            vulkan::create_pipeline(&device, &mut app_data)?;

            vulkan::create_framebuffers(&device, &mut app_data)?;

            vulkan::create_command_pool(&entry, &instance, &device, &mut app_data)?;

            vulkan::create_command_buffers(&device, &mut app_data)?;

            vulkan::create_sync_objects(&device, &mut app_data)?;
        }

        Ok(Self {
            entry,
            instance,
            app_data,
            device,
            frame: 0,
        })
    }

    pub fn render(&mut self, window: &Window) -> Result<(), ApplicationError> {
        unsafe {
            self.device.wait_for_fences(
                &[self.app_data.in_flight_frame_fences[self.frame as usize]],
                true,
                u64::MAX,
            )?;
        }

        let swapchain_loader = extensions::khr::Swapchain::new(&self.instance, &self.device);

        let image_index = unsafe {
            swapchain_loader.acquire_next_image(
                self.app_data.swapchain,
                // Using u64::MAX value disables the timeout.
                u64::MAX,
                self.app_data.image_available_semaphores[self.frame as usize],
                vk::Fence::null(),
            )
        }?
        .0;

        if self.app_data.in_flight_image_fences[image_index as usize] != vk::Fence::null() {
            unsafe {
                self.device.wait_for_fences(
                    &[self.app_data.in_flight_image_fences[image_index as usize]],
                    true,
                    u64::MAX,
                )?;
            }
        }

        self.app_data.in_flight_image_fences[image_index as usize] =
            self.app_data.in_flight_frame_fences[self.frame as usize];

        // Each entry in wait_dst_stage_masks array corresponds to the semaphores with the same index in wait_semaphores.
        let wait_semaphores = [self.app_data.image_available_semaphores[self.frame as usize]];
        let wait_dst_stage_masks = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [self.app_data.command_buffers[image_index as usize]];
        let signal_semaphores = [self.app_data.render_finished_semaphores[self.frame as usize]];

        let submit_infos = [vk::SubmitInfo::builder()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_dst_stage_masks)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores)
            .build()];

        unsafe {
            self.device
                .reset_fences(&[self.app_data.in_flight_frame_fences[self.frame as usize]])?;

            self.device.queue_submit(
                self.app_data.graphics_queue,
                &submit_infos,
                self.app_data.in_flight_frame_fences[self.frame as usize],
            )?;
        }

        let swapchains = [self.app_data.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe { swapchain_loader.queue_present(self.app_data.graphics_queue, &present_info) }?;

        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            self.app_data
                .image_available_semaphores
                .iter()
                .for_each(|s| self.device.destroy_semaphore(*s, None));
            self.app_data
                .render_finished_semaphores
                .iter()
                .for_each(|s| self.device.destroy_semaphore(*s, None));
            self.app_data
                .in_flight_frame_fences
                .iter()
                .for_each(|f| self.device.destroy_fence(*f, None));

            self.device
                .destroy_command_pool(self.app_data.command_pool, None);

            self.app_data
                .framebuffers
                .iter()
                .for_each(|f| self.device.destroy_framebuffer(*f, None));

            self.device.destroy_pipeline(self.app_data.pipeline, None);

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

#[derive(Default)]
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
    pub pipeline: vk::Pipeline,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub command_pool: vk::CommandPool,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub in_flight_frame_fences: Vec<vk::Fence>,
    pub in_flight_image_fences: Vec<vk::Fence>,
}
