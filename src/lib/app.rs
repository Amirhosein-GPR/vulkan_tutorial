use crate::error::AppError;
use crate::vulkan::{self, UniformBufferObject, Vertex, MAX_FRAMES_IN_FLIGHT, VALIDATION_ENABLED};
use ash::{extensions, vk, Device, Entry, Instance};
use nalgebra::{Matrix4, Point3, Vector3};
use std::f32::consts::FRAC_PI_4;
use std::time::Instant;
use std::{mem, ptr};
use winit::window::Window;

pub struct App {
    entry: Entry,
    instance: Instance,
    app_data: AppData,
    pub device: Device,
    frame: u8,
    pub resized: bool,
    start: Instant,
    pub models: usize,
}

impl App {
    pub fn new(window: &Window) -> Result<Self, AppError> {
        let mut app_data = AppData::default();

        // Creating Vulkan entry point. The first thing we need to create before even creating the Vulkan instance.
        let entry = unsafe { Entry::load() }?;

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

            vulkan::create_descriptor_set_layout(&device, &mut app_data)?;

            vulkan::create_pipeline(&device, &mut app_data)?;

            vulkan::create_command_pools(&entry, &instance, &device, &mut app_data)?;

            vulkan::create_color_objects(&instance, &device, &mut app_data)?;

            vulkan::create_depth_objects(&instance, &device, &mut app_data)?;

            vulkan::create_framebuffers(&device, &mut app_data)?;

            vulkan::create_texture_image(&instance, &device, &mut app_data)?;

            vulkan::create_texture_image_view(&device, &mut app_data)?;

            vulkan::create_texture_sampler(&device, &mut app_data)?;

            vulkan::load_model(&mut app_data)?;

            vulkan::create_vertex_buffer(&instance, &device, &mut app_data)?;

            vulkan::create_index_buffer(&instance, &device, &mut app_data)?;

            vulkan::create_uniform_buffers(&instance, &device, &mut app_data)?;

            vulkan::create_descriptor_pool(&device, &mut app_data)?;

            vulkan::create_descriptor_sets(&device, &mut app_data)?;

            vulkan::create_command_buffers(&device, &mut app_data)?;

            vulkan::create_sync_objects(&device, &mut app_data)?;
        }

        Ok(Self {
            entry,
            instance,
            app_data,
            device,
            frame: 0,
            resized: false,
            start: Instant::now(),
            models: 1,
        })
    }

    pub fn render(&mut self, window: &Window) -> Result<(), AppError> {
        unsafe {
            self.device.wait_for_fences(
                &[self.app_data.in_flight_frame_fences[self.frame as usize]],
                true,
                u64::MAX,
            )?;
        }

        let swapchain_loader = extensions::khr::Swapchain::new(&self.instance, &self.device);

        let result = unsafe {
            swapchain_loader.acquire_next_image(
                self.app_data.swapchain,
                // Using u64::MAX value disables the timeout.
                u64::MAX,
                self.app_data.image_available_semaphores[self.frame as usize],
                vk::Fence::null(),
            )
        };

        let image_index = match result {
            Ok((image_index, _)) => image_index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(error) => return Err(error.into()),
        };

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

        unsafe {
            self.update_command_buffer(image_index as usize)?;
            self.update_uniform_buffer(image_index as usize)?;
        }

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

        let result =
            unsafe { swapchain_loader.queue_present(self.app_data.graphics_queue, &present_info) };

        let changed = result == Ok(true) || result == Err(vk::Result::ERROR_OUT_OF_DATE_KHR);

        if self.resized || changed {
            self.resized = false;
            self.recreate_swapchain(window)?;
        } else if let Err(e) = result {
            return Err(e.into());
        }

        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    pub fn recreate_swapchain(&mut self, window: &Window) -> Result<(), AppError> {
        unsafe {
            self.device.device_wait_idle()?;
            self.destroy_swapchain();

            vulkan::create_swapchain(
                &self.entry,
                window,
                &self.instance,
                &self.device,
                &mut self.app_data,
            )?;
            vulkan::create_swapchain_image_views(&self.device, &mut self.app_data)?;
            vulkan::create_render_pass(&self.instance, &self.device, &mut self.app_data)?;
            vulkan::create_pipeline(&self.device, &mut self.app_data)?;
            vulkan::create_color_objects(&self.instance, &self.device, &mut self.app_data)?;
            vulkan::create_depth_objects(&self.instance, &self.device, &mut self.app_data)?;
            vulkan::create_framebuffers(&self.device, &mut self.app_data)?;
            vulkan::create_uniform_buffers(&self.instance, &self.device, &mut self.app_data)?;
            vulkan::create_descriptor_pool(&self.device, &mut self.app_data)?;
            vulkan::create_descriptor_sets(&self.device, &mut self.app_data)?;
            vulkan::create_command_buffers(&self.device, &mut self.app_data)?;
            self.app_data
                .in_flight_image_fences
                .resize(self.app_data.swapchain_images.len(), vk::Fence::null());
        }

        Ok(())
    }

    unsafe fn destroy_swapchain(&mut self) {
        self.device
            .destroy_image_view(self.app_data.color_image_view, None);

        self.device
            .free_memory(self.app_data.color_image_memory, None);

        self.device.destroy_image(self.app_data.color_image, None);

        self.device
            .destroy_descriptor_pool(self.app_data.descriptor_pool, None);

        self.app_data
            .uniform_buffers
            .iter()
            .for_each(|b| self.device.destroy_buffer(*b, None));

        self.app_data
            .uniform_buffers_memories
            .iter()
            .for_each(|m| self.device.free_memory(*m, None));

        self.device
            .destroy_image_view(self.app_data.depth_image_view, None);

        self.device.destroy_image(self.app_data.depth_image, None);

        self.device
            .free_memory(self.app_data.depth_image_memory, None);

        self.app_data
            .framebuffers
            .iter()
            .for_each(|f| self.device.destroy_framebuffer(*f, None));

        self.device.destroy_pipeline(self.app_data.pipeline, None);

        self.device
            .destroy_pipeline_layout(self.app_data.pipeline_layout, None);

        self.device
            .destroy_render_pass(self.app_data.render_pass, None);

        self.app_data
            .swapchain_image_views
            .iter()
            .for_each(|v| self.device.destroy_image_view(*v, None));

        extensions::khr::Swapchain::new(&self.instance, &self.device)
            .destroy_swapchain(self.app_data.swapchain, None);
    }

    unsafe fn update_uniform_buffer(&self, image_index: usize) -> Result<(), AppError> {
        let view = Matrix4::look_at_rh(
            &Point3::<f32>::new(6.0, 0.0, 2.0),
            &Point3::<f32>::new(0.0, 0.0, 0.0),
            &Vector3::<f32>::new(0.0, 0.0, 1.0),
        );

        let mut project = Matrix4::new_perspective(
            self.app_data.swapchain_extent.width as f32
                / self.app_data.swapchain_extent.height as f32,
            FRAC_PI_4,
            0.1,
            10.0,
        );

        project[(1, 1)] *= -1.0;

        let ubo = UniformBufferObject { view, project };

        let memory = self.device.map_memory(
            self.app_data.uniform_buffers_memories[image_index],
            0,
            mem::size_of::<UniformBufferObject>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        ptr::copy_nonoverlapping(&ubo, memory.cast(), 1);

        self.device
            .unmap_memory(self.app_data.uniform_buffers_memories[image_index]);

        Ok(())
    }

    unsafe fn update_command_buffer(&mut self, image_index: usize) -> Result<(), AppError> {
        let command_pool = self.app_data.command_pools[image_index];
        self.device
            .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())?;

        let command_buffer = self.app_data.command_buffers[image_index];

        // Only relevant for secondary command buffers.
        let command_buffer_inheritance_info = vk::CommandBufferInheritanceInfo::builder();
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            .inheritance_info(&command_buffer_inheritance_info);
        self.device
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)?;

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(self.app_data.swapchain_extent)
            .build();

        let clear_color_depth_stencil_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];

        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.app_data.render_pass)
            .framebuffer(self.app_data.framebuffers[image_index])
            .render_area(render_area)
            .clear_values(&clear_color_depth_stencil_values);

        self.device.cmd_begin_render_pass(
            command_buffer,
            &render_pass_begin_info,
            vk::SubpassContents::SECONDARY_COMMAND_BUFFERS,
        );

        let secondary_command_buffers = (0..self.models)
            .map(|i| self.update_secondary_command_buffer(image_index, i))
            .collect::<Result<Vec<_>, _>>()?;

        self.device
            .cmd_execute_commands(command_buffer, &secondary_command_buffers);

        self.device.cmd_end_render_pass(command_buffer);

        self.device.end_command_buffer(command_buffer)?;

        Ok(())
    }

    unsafe fn update_secondary_command_buffer(
        &mut self,
        image_index: usize,
        model_index: usize,
    ) -> Result<vk::CommandBuffer, AppError> {
        self.app_data
            .secondary_command_buffers
            .resize_with(image_index + 1, Vec::new);

        let command_buffers = &mut self.app_data.secondary_command_buffers[image_index];

        while model_index >= command_buffers.len() {
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.app_data.command_pools[image_index])
                .level(vk::CommandBufferLevel::SECONDARY)
                .command_buffer_count(1);

            let command_buffer = self
                .device
                .allocate_command_buffers(&command_buffer_allocate_info)?[0];

            command_buffers.push(command_buffer);
        }

        let command_buffer = command_buffers[model_index];

        let y = (((model_index % 2) as f32) * 2.5) - 1.25;
        let z = (((model_index / 2) as f32) * -2.0) + 1.0;

        let model = Matrix4::<f32>::new_translation(&Vector3::new(0.0, y, z));

        let time = self.start.elapsed().as_secs_f32();

        let model = Matrix4::<f32>::from_axis_angle(&Vector3::z_axis(), time * FRAC_PI_4) * model;
        let (_, model_bytes, _) = model.as_slice().align_to::<u8>();

        let opacity = (model_index + 1) as f32 * 0.25;
        let opacity_bytes = &opacity.to_ne_bytes()[..];

        let command_buffer_inheritance_info = vk::CommandBufferInheritanceInfo::builder()
            .render_pass(self.app_data.render_pass)
            .subpass(0)
            .framebuffer(self.app_data.framebuffers[image_index]);

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
            .inheritance_info(&command_buffer_inheritance_info);

        self.device
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)?;

        self.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.app_data.pipeline,
        );
        self.device.cmd_bind_vertex_buffers(
            command_buffer,
            0,
            &[self.app_data.vertex_buffer],
            &[0],
        );
        self.device.cmd_bind_index_buffer(
            command_buffer,
            self.app_data.index_buffer,
            0,
            vk::IndexType::UINT32,
        );
        self.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.app_data.pipeline_layout,
            0,
            &[self.app_data.descriptor_sets[image_index]],
            &[],
        );
        self.device.cmd_push_constants(
            command_buffer,
            self.app_data.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            model_bytes,
        );
        self.device.cmd_push_constants(
            command_buffer,
            self.app_data.pipeline_layout,
            vk::ShaderStageFlags::FRAGMENT,
            64,
            opacity_bytes,
        );
        self.device.cmd_draw_indexed(
            command_buffer,
            self.app_data.indices.len() as u32,
            1,
            0,
            0,
            0,
        );

        self.device.end_command_buffer(command_buffer)?;

        Ok(command_buffer)
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            self.destroy_swapchain();

            self.app_data
                .command_pools
                .iter()
                .for_each(|p| self.device.destroy_command_pool(*p, None));

            self.device
                .destroy_sampler(self.app_data.texture_sampler, None);

            self.device
                .destroy_image_view(self.app_data.texture_image_view, None);

            self.device.destroy_image(self.app_data.texture_image, None);

            self.device
                .free_memory(self.app_data.texture_image_memory, None);

            self.device
                .destroy_descriptor_set_layout(self.app_data.descriptor_set_layout, None);

            self.device
                .destroy_buffer(self.app_data.vertex_buffer, None);

            self.device
                .free_memory(self.app_data.vertex_buffer_memory, None);

            self.device.destroy_buffer(self.app_data.index_buffer, None);

            self.device
                .free_memory(self.app_data.index_buffer_memory, None);

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
    pub msaa_samples: vk::SampleCountFlags,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub surface: vk::SurfaceKHR,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub render_pass: vk::RenderPass,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub command_pool: vk::CommandPool,
    pub command_pools: Vec<vk::CommandPool>,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub secondary_command_buffers: Vec<Vec<vk::CommandBuffer>>,
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub in_flight_frame_fences: Vec<vk::Fence>,
    pub in_flight_image_fences: Vec<vk::Fence>,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub vertex_buffer: vk::Buffer,
    pub vertex_buffer_memory: vk::DeviceMemory,
    pub index_buffer: vk::Buffer,
    pub index_buffer_memory: vk::DeviceMemory,
    pub uniform_buffers: Vec<vk::Buffer>,
    pub uniform_buffers_memories: Vec<vk::DeviceMemory>,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub mip_levels: u32,
    pub texture_image: vk::Image,
    pub texture_image_memory: vk::DeviceMemory,
    pub texture_image_view: vk::ImageView,
    pub texture_sampler: vk::Sampler,
    pub depth_image: vk::Image,
    pub depth_image_memory: vk::DeviceMemory,
    pub depth_image_view: vk::ImageView,
    pub color_image: vk::Image,
    pub color_image_memory: vk::DeviceMemory,
    pub color_image_view: vk::ImageView,
}
