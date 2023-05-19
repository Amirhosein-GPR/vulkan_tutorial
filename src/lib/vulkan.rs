use crate::app::AppData;
use crate::error::{ApplicationError, SuitabilityError};
use ash::{extensions, util, vk, Device, Entry, Instance};
use log::{debug, error, info, trace, warn};
use std::collections::HashSet;
use std::ffi::{c_void, CStr, CString};
use winit::platform::x11::WindowExtX11;
use winit::window::Window;

// This version number is important for platforms that not fully conform to the Vulkan API specificaiton.
const PORTABILITY_MACOS_VERSION: u32 = vk::make_api_version(0, 1, 3, 216);
pub const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
const VALIDATION_LAYER_NAME: &CStr =
    unsafe { &CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };
const DEVICE_EXTENSIONS_NAMES: &[&CStr] = &[extensions::khr::Swapchain::name()];

struct QueueFamilyIndices {
    graphics_queue_family: u32,
    present_queue_family: u32,
}

impl QueueFamilyIndices {
    // This function checks the given physical device for required queue families.
    unsafe fn get(
        entry: &Entry,
        instance: &Instance,
        app_data: &mut AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self, ApplicationError> {
        let queue_family_properties =
            instance.get_physical_device_queue_family_properties(physical_device);

        // Checking for a graphics queue family.
        let graphics_queue_family = queue_family_properties
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        // Checking for a present queue family.
        let mut present_queue_family = None;
        let surface_loader = extensions::khr::Surface::new(entry, instance);
        for (index, _queue_family_properties) in queue_family_properties.into_iter().enumerate() {
            if surface_loader.get_physical_device_surface_support(
                physical_device,
                index as u32,
                app_data.surface,
            )? {
                present_queue_family = Some(index as u32);
                break;
            }
        }

        if let (Some(graphics_queue_family), Some(present_queue_family)) =
            (graphics_queue_family, present_queue_family)
        {
            Ok(Self {
                graphics_queue_family,
                present_queue_family,
            })
        } else {
            Err(SuitabilityError(
                "Missing required queue families.".to_string(),
            ))?
        }
    }
}

struct SwapchainSupport {
    surface_capabilites: vk::SurfaceCapabilitiesKHR,
    surface_formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    // Gets physical device surface properties.
    unsafe fn get(
        entry: &Entry,
        instance: &Instance,
        app_data: &mut AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self, ApplicationError> {
        let surface_loader = extensions::khr::Surface::new(entry, instance);

        Ok(Self {
            surface_capabilites: surface_loader
                .get_physical_device_surface_capabilities(physical_device, app_data.surface)?,
            surface_formats: surface_loader
                .get_physical_device_surface_formats(physical_device, app_data.surface)?,
            present_modes: surface_loader
                .get_physical_device_surface_present_modes(physical_device, app_data.surface)?,
        })
    }
}

// Creates a Vulkan instance.
pub unsafe fn create_instance(
    window: &Window,
    entry: &Entry,
    app_data: &mut AppData,
) -> Result<Instance, ApplicationError> {
    let application_name = CString::new("Vulkan tutorial").unwrap();
    let engine_name = CString::new("Vulkan engine").unwrap();
    let application_info = vk::ApplicationInfo::builder()
        .application_name(&application_name)
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_3);

    let mut instance_layers_names = Vec::new();
    // These extensions are needed for surface creation. Surface is our link to window for preseneting images on it.
    let mut instance_extensions_names = vec![
        extensions::khr::Surface::name().as_ptr(),
        extensions::khr::XlibSurface::name().as_ptr(),
    ];

    if VALIDATION_ENABLED {
        // Enumerating instance layer properties. First getting them in [i8] format, then creating a slice of CStr from them.
        let available_instance_layer_names = entry
            .enumerate_instance_layer_properties()?
            .into_iter()
            .map(|l| l.layer_name)
            .collect::<Vec<_>>();
        let available_instance_layer_names = available_instance_layer_names
            .iter()
            .map(|l| CStr::from_ptr(l.as_ptr()))
            .collect::<Vec<_>>();

        if !available_instance_layer_names.contains(&VALIDATION_LAYER_NAME) {
            return Err(ApplicationError::EngineError(
                "Validation layer requested but not supported.".to_string(),
            ));
        }

        instance_layers_names.push(VALIDATION_LAYER_NAME.as_ptr());
        instance_extensions_names.push(extensions::ext::DebugUtils::name().as_ptr());
    }

    /* Starting from Vulkan version 1.3.216, platforms that not fully conform to the Vulkan API specification should enable
    these extensions and pass them with this flag to instance create info struct. */
    let instance_create_flags = if cfg!(target_os = "macos")
        && entry.try_enumerate_instance_version()?.unwrap() >= PORTABILITY_MACOS_VERSION
    {
        info!("Enabling extensions for macOS portability.");

        instance_extensions_names
            .push(extensions::khr::GetPhysicalDeviceProperties2::name().as_ptr());
        instance_extensions_names.push(vk::KhrPortabilityEnumerationFn::name().as_ptr());

        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::empty()
    };

    let mut instance_create_info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_layer_names(&instance_layers_names)
        .enabled_extension_names(&instance_extensions_names)
        .flags(instance_create_flags);

    // Creating debug utils messenge which is used to enable validation layers.
    let instance;
    if VALIDATION_ENABLED {
        let mut debug_utils_messenger_create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(debug_callback));

        instance_create_info =
            instance_create_info.push_next(&mut debug_utils_messenger_create_info);

        instance = entry.create_instance(&instance_create_info, None)?;

        app_data.debug_utils_messenger = extensions::ext::DebugUtils::new(entry, &instance)
            .create_debug_utils_messenger(&debug_utils_messenger_create_info, None)?;
    } else {
        instance = entry.create_instance(&instance_create_info, None)?;
    }

    Ok(instance)
}

extern "system" fn debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    let message = unsafe { CStr::from_ptr((*callback_data).p_message) }.to_string_lossy();

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => error!("({:?}) {}", message_type, message),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => warn!("({:?}) {}", message_type, message),
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => info!("({:?}) {}", message_type, message),
        _ => {
            trace!("({:?}), {}", message_type, message)
        }
    }

    vk::FALSE
}

// This function picks the required physical device.
pub unsafe fn pick_physical_device(
    entry: &Entry,
    instance: &Instance,
    app_data: &mut AppData,
) -> Result<(), ApplicationError> {
    for physical_device in instance.enumerate_physical_devices()? {
        let physical_device_name = CStr::from_ptr(
            instance
                .get_physical_device_properties(physical_device)
                .device_name
                .as_ptr(),
        );

        if let Err(error) = check_physical_device(entry, instance, app_data, physical_device) {
            warn!(
                "Skipping physical device ('{:?}'): {}",
                physical_device_name, error
            );
        } else {
            info!("Selected physical device ('{:?}').", physical_device_name);
            app_data.physical_device = physical_device;
            return Ok(());
        }
    }

    Err(SuitabilityError(
        "Failed to find suitable physical device.".to_string(),
    ))?
}

// This function checks the given physical device for required properties and features.
unsafe fn check_physical_device(
    entry: &Entry,
    instance: &Instance,
    app_data: &mut AppData,
    physical_device: vk::PhysicalDevice,
) -> Result<(), ApplicationError> {
    let physical_device_properties = instance.get_physical_device_properties(physical_device);
    if physical_device_properties.device_type != vk::PhysicalDeviceType::DISCRETE_GPU {
        return Err(SuitabilityError(
            "Only discrete GPUs are supported.".to_string(),
        ))?;
    }

    let physical_device_features = instance.get_physical_device_features(physical_device);
    if physical_device_features.geometry_shader != vk::TRUE {
        return Err(SuitabilityError(
            "Missing geometry shader support.".to_string(),
        ))?;
    }

    QueueFamilyIndices::get(entry, instance, app_data, physical_device)?;

    check_physical_device_extensions(instance, physical_device)?;

    // If there isn't any surface formats or present modes available, then there's no sufficient support for swapchain in current physical device.
    let swapchain_support = SwapchainSupport::get(entry, instance, app_data, physical_device)?;
    if swapchain_support.surface_formats.is_empty() || swapchain_support.present_modes.is_empty() {
        return Err(SuitabilityError(
            "Insufficient swapchain support.".to_string(),
        ))?;
    }

    Ok(())
}

// This function creates a logical device and retrieves queue handles that are automatically created with the logical device.
pub unsafe fn create_device(
    entry: &Entry,
    instance: &Instance,
    app_data: &mut AppData,
) -> Result<Device, ApplicationError> {
    let queue_family_indices =
        QueueFamilyIndices::get(entry, instance, app_data, app_data.physical_device)?;

    // Using hash set to avoid possible duplicated queue family indices.
    let mut unique_indices = HashSet::new();
    unique_indices.insert(queue_family_indices.graphics_queue_family);
    unique_indices.insert(queue_family_indices.present_queue_family);

    let queue_priorities = [1.0];
    let device_queue_create_infos = unique_indices
        .into_iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(i)
                .queue_priorities(&queue_priorities)
                .build()
        })
        .collect::<Vec<_>>();

    let mut device_layers_names = Vec::new();
    let mut device_extensions_names = DEVICE_EXTENSIONS_NAMES
        .into_iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();
    let device_features = vk::PhysicalDeviceFeatures::builder();
    if VALIDATION_ENABLED {
        device_layers_names.push(VALIDATION_LAYER_NAME.as_ptr());
    }

    /* Starting from Vulkan version 1.3.216, platforms that not fully conform to the Vulkan API specification should enable
    this extension and pass it to device create info struct. */
    if cfg!(target_os = "macos")
        && entry.try_enumerate_instance_version()?.unwrap() >= PORTABILITY_MACOS_VERSION
    {
        device_extensions_names.push(vk::KhrPortabilitySubsetFn::name().as_ptr());
    }

    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&device_queue_create_infos)
        .enabled_layer_names(&device_layers_names)
        .enabled_extension_names(&device_extensions_names)
        .enabled_features(&device_features);

    let device = instance.create_device(app_data.physical_device, &device_create_info, None)?;

    app_data.graphics_queue =
        device.get_device_queue(queue_family_indices.graphics_queue_family, 0);
    app_data.present_queue = device.get_device_queue(queue_family_indices.present_queue_family, 0);

    Ok(device)
}

// Creates a surface with the help of winit and vulkan extensions.
pub unsafe fn create_surface(
    entry: &Entry,
    instance: &Instance,
    window: &Window,
) -> Result<vk::SurfaceKHR, ApplicationError> {
    let xlib_surface_create_info = vk::XlibSurfaceCreateInfoKHR::builder()
        .dpy(window.xlib_display().unwrap() as *mut vk::Display)
        .window(window.xlib_window().unwrap());

    let surface_loader = extensions::khr::XlibSurface::new(entry, instance);
    let surface = surface_loader.create_xlib_surface(&xlib_surface_create_info, None)?;

    Ok(surface)
}

// Checks physical device to support logical device extensions.
unsafe fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<(), ApplicationError> {
    let available_device_extensions_names = instance
        .enumerate_device_extension_properties(physical_device)?
        .into_iter()
        .map(|e| e.extension_name)
        .collect::<Vec<_>>();
    let available_device_extensions_names = available_device_extensions_names
        .iter()
        .map(|e| CStr::from_ptr(e.as_ptr()))
        .collect::<Vec<_>>();

    if DEVICE_EXTENSIONS_NAMES
        .iter()
        .all(|e| available_device_extensions_names.contains(e))
    {
        Ok(())
    } else {
        Err(SuitabilityError(
            "Missing required device extensions.".to_string(),
        ))?
    }
}

// Picks one of many possible surface formats.
unsafe fn get_swapchain_surface_format(
    surface_fomrats: &[vk::SurfaceFormatKHR],
) -> vk::SurfaceFormatKHR {
    surface_fomrats
        .into_iter()
        .cloned()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or_else(|| surface_fomrats[0])
}

// Picks one of many possible surface present modes.
unsafe fn get_swapchain_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    present_modes
        .into_iter()
        .cloned()
        .find(|m| *m == vk::PresentModeKHR::MAILBOX)
        .unwrap_or_else(|| vk::PresentModeKHR::FIFO)
}

// Gets surface extent from surface capabilites or creates a new one if we can decide on it's size.
unsafe fn get_swapchain_extent(
    window: &Window,
    surface_capabilities: vk::SurfaceCapabilitiesKHR,
) -> vk::Extent2D {
    /* u32::MAX is the maximum u32 number value. Here it has a meaning. If current extent width is equal to u32::MAX,
    it means we can pick a resolution that best matches the window within min image extent and max image extent.*/
    if surface_capabilities.current_extent.width != u32::MAX {
        surface_capabilities.current_extent
    } else {
        let window_size = window.inner_size();
        let clamp = |min: u32, max: u32, value: u32| min.max(max.min(value));
        vk::Extent2D::builder()
            .width(clamp(
                surface_capabilities.min_image_extent.width,
                surface_capabilities.max_image_extent.width,
                window_size.width,
            ))
            .height(clamp(
                surface_capabilities.min_image_extent.height,
                surface_capabilities.max_image_extent.height,
                window_size.height,
            ))
            .build()
    }
}

// Creates swapchain and retrieves handles to images in it and store them in app data.
pub unsafe fn create_swapchain(
    entry: &Entry,
    window: &Window,
    instance: &Instance,
    device: &Device,
    app_data: &mut AppData,
) -> Result<(), ApplicationError> {
    let queue_family_indices =
        QueueFamilyIndices::get(entry, instance, app_data, app_data.physical_device)?;
    let swapchain_support =
        SwapchainSupport::get(entry, instance, app_data, app_data.physical_device)?;

    let surface_format = get_swapchain_surface_format(&swapchain_support.surface_formats);
    let present_mode = get_swapchain_present_mode(&swapchain_support.present_modes);
    let extent = get_swapchain_extent(window, swapchain_support.surface_capabilites);

    let mut image_count = swapchain_support.surface_capabilites.min_image_count + 1;
    // '0' for max image count is a special value which means there is no maximum for image count.
    if swapchain_support.surface_capabilites.max_image_count != 0
        && image_count > swapchain_support.surface_capabilites.max_image_count
    {
        image_count = swapchain_support.surface_capabilites.max_image_count;
    }

    let mut seperate_queue_family_indices = Vec::new();
    // Concurrent image sharing mode is used for at least 2 different queue families.
    let image_sharing_mode = if queue_family_indices.graphics_queue_family
        != queue_family_indices.present_queue_family
    {
        seperate_queue_family_indices.push(queue_family_indices.graphics_queue_family);
        seperate_queue_family_indices.push(queue_family_indices.present_queue_family);
        vk::SharingMode::CONCURRENT
    } else {
        vk::SharingMode::EXCLUSIVE
    };

    let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
        .surface(app_data.surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(image_sharing_mode)
        .queue_family_indices(&seperate_queue_family_indices)
        .pre_transform(swapchain_support.surface_capabilites.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    let swapchain_loader = extensions::khr::Swapchain::new(instance, device);
    app_data.swapchain = swapchain_loader.create_swapchain(&swapchain_create_info, None)?;
    app_data.swapchain_images = swapchain_loader.get_swapchain_images(app_data.swapchain)?;
    app_data.swapchain_format = surface_format.format;
    app_data.swapchain_extent = extent;

    Ok(())
}

// Creates image views, which are needed to work with images in the swapchain.
pub unsafe fn create_swapchain_image_views(
    device: &Device,
    app_data: &mut AppData,
) -> Result<(), ApplicationError> {
    app_data.swapchain_image_views = app_data
        .swapchain_images
        .iter()
        .map(|i| {
            let component_mapping = vk::ComponentMapping::builder()
                .r(vk::ComponentSwizzle::IDENTITY)
                .g(vk::ComponentSwizzle::IDENTITY)
                .b(vk::ComponentSwizzle::IDENTITY)
                .a(vk::ComponentSwizzle::IDENTITY)
                .build();
            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build();
            let image_view_create_info = vk::ImageViewCreateInfo::builder()
                .image(*i)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(app_data.swapchain_format)
                .components(component_mapping)
                .subresource_range(subresource_range);

            device.create_image_view(&image_view_create_info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(())
}

pub unsafe fn create_render_pass(
    instance: &Instance,
    device: &Device,
    app_data: &mut AppData,
) -> Result<(), ApplicationError> {
    let color_attachment_descriptions = [vk::AttachmentDescription::builder()
        .format(app_data.swapchain_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .build()];

    let color_attachment_refrences = [vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build()];

    let subpass_descriptions = [vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_refrences)
        .build()];

    let render_pass_create_info = vk::RenderPassCreateInfo::builder()
        .attachments(&color_attachment_descriptions)
        .subpasses(&subpass_descriptions);

    app_data.render_pass = device.create_render_pass(&render_pass_create_info, None)?;

    Ok(())
}

// Creates pipeline by specifying it's required states and stages.
pub unsafe fn create_pipeline(
    device: &Device,
    app_data: &mut AppData,
) -> Result<(), ApplicationError> {
    let vertex_shader_spv = include_bytes!("../../shaders/vert.spv");
    let fragment_shader_spv = include_bytes!("../../shaders/frag.spv");

    let vertex_shader_module = create_shader_module(device, vertex_shader_spv)?;
    let fragment_shader_module = create_shader_module(device, fragment_shader_spv)?;

    let vertex_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vertex_shader_module)
        .name(CStr::from_bytes_with_nul_unchecked(b"main\0"));
    let fragment_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(fragment_shader_module)
        .name(CStr::from_bytes_with_nul_unchecked(b"main\0"));

    let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo::builder();

    let input_assembly_state_create_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let viewports = [vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(app_data.swapchain_extent.width as f32)
        .height(app_data.swapchain_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0)
        .build()];
    let scissor_rectangles = [vk::Rect2D::builder()
        .offset(vk::Offset2D::builder().x(0).y(0).build())
        .extent(app_data.swapchain_extent)
        .build()];

    let viewport_state_create_info = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(&viewports)
        .scissors(&scissor_rectangles);

    let rasterization_state_create_info = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE)
        .depth_bias_enable(false);

    let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let depth_stenci_state_create_info = vk::PipelineDepthStencilStateCreateInfo::builder();

    // This color blending configuration is per attached framebuffer
    let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(false)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ZERO)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD)
        .build()];

    // This color blending configuration is global
    let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(&color_blend_attachment_states)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    // If we want to use dynamic states, we should mention them here.
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::LINE_WIDTH];
    let dynamic_state_create_info =
        vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

    // Pipeline layout is used for creating uniform values which are used in shaders and their values can be chaned at drawing times.
    let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder();

    app_data.pipeline_layout = device.create_pipeline_layout(&pipeline_layout_create_info, None)?;

    device.destroy_shader_module(vertex_shader_module, None);
    device.destroy_shader_module(fragment_shader_module, None);

    Ok(())
}

// Creates shader module which is needed to link each shader to pipeline.
unsafe fn create_shader_module(
    device: &Device,
    bytecode: &[u8],
) -> Result<vk::ShaderModule, ApplicationError> {
    // Shader module create info needs a u32 slice but what we read is a u8 slice. So we align our u8 slice to a u32 slice.
    let (prefix, code, suffix) = bytecode.align_to::<u32>();
    // We also check if our slice is aligned properly which means there shouldn't be any prefix or suffix for the aligned slice.
    if !prefix.is_empty() || !suffix.is_empty() {
        return Err(ApplicationError::EngineError(
            "Shader bytecode is not properly aligned.".to_string(),
        ));
    }

    let shader_module_create_info = vk::ShaderModuleCreateInfo::builder().code(&code);

    Ok(device.create_shader_module(&shader_module_create_info, None)?)
}
